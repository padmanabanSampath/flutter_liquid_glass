# Post-training cross-layer weight/basis sharing for pretrained transformers (calibration-only or light finetune)

*How these notes were gathered (2026-10-12): the session's egress policy blocked arxiv.org, openreview.net, proceedings.iclr.cc, huggingface.co and every other paper host (HTTP 403 on CONNECT). Only GitHub (raw.githubusercontent.com) and the web-search tool were reachable. So: (a) the numbers marked "primary (GitHub)" were read directly from official repo files; (b) all other paper numbers come from search-engine excerpts of the cited arXiv/OpenReview/ACL pages and should be spot-checked against the PDFs before final publication. Convention: "X% compression" = X% of parameters removed (keep ratio = 1 − X). Dense LLaMA-1-7B WikiText-2 PPL is 5.67–5.68.*

---

## Q1. Basis Sharing (Wang et al., ICLR 2025): shared SVD basis across groups of layers

### Takeaway
Basis Sharing is the reference method for training-free cross-layer sharing. It fits one SVD basis for a group of consecutive layers and gives each layer its own coefficient matrix, closed form after SVD-LLM-style activation whitening, from 256 WikiText calibration sequences. It beats per-layer SVD (SVD-LLM) at equal parameter counts, especially at high compression. In absolute terms it still loses ~2 PPL at only 20% compression on LLaMA-1/2 and ~9 PPL on Llama-3.1-8B, so it does not come close to 4-bit quantization's quality-per-byte.

### Cited Findings
- **Paper:** "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression", Jingcun Wang, Yu-Guang Chen, Ing-Chao Lin, Bing Li, Grace Li Zhang. arXiv 2410.03765 (Oct 2024), ICLR 2025 poster (published 22 Jan 2025). — [OpenReview](https://openreview.net/forum?id=gp32jvUquq); [arXiv](https://arxiv.org/abs/2410.03765); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)
- **Core idea:** weight matrices in several layers are written in a shared set of basis vectors. Each layer keeps unique coefficients, so the layers keep different functions. The paper argues that naive full sharing degrades badly and that training shared-weight LLMs from scratch is impractical. — [arXiv abstract via search](https://arxiv.org/abs/2410.03765); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)
- **What is shared vs. per-layer (official config, primary GitHub):** for LLaMA-7B at 20% compression the config sets `group_size: 2`, `share_part: [v, k, q, up, gate]` and `private_part: [down, o]`. So Q/K/V/up/gate share a basis within each 2-layer group, while o_proj and down_proj get ordinary per-layer low-rank SVD. Calibration is `dataset_name: wikitext`, `calibration_size: 256`, `context_length: 2048`, i.e. ~0.5M tokens. — [Basis_Sharing config share_llama_7b_20.yaml](https://raw.githubusercontent.com/TUDa-HWAI/Basis_Sharing/main/tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml)
- **Optional healing:** the same config defines LoRA healing with r=8, alpha=32, lr 1e-4 and 2 epochs. The repo ships `lora.py`, `test_adapter.py` (reasoning tasks) and `test_throughput.py`. — [Basis_Sharing README](https://github.com/TUDa-HWAI/Basis_Sharing) (raw README read)
- **Group size vs. ratio:** "4" means every four consecutive layers share a basis matrix, assigned from first to last layer. Per the paper, grouping 4–5 layers gives the lowest PPL when compression is below 30%, and 2-layer groups are better above 30%. — [ICLR PDF via search](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)
- **Paper's own results, 20% compression, WikiText-2 PPL (Basis Sharing vs SVD-LLM):** OPT-6.7B 11.79 vs 16.04; LLaMA-2-7B 7.70 vs 8.5; Mistral-7B 7.57 vs 10.21. That is up to 25% lower PPL than SVD-LLM. LLaMA-7B/13B/30B are also evaluated, and Basis Sharing reportedly has the lowest PPL among the SVD methods at 50%. — [OpenReview/ICLR via search](https://openreview.net/forum?id=gp32jvUquq)
- **Independent standardized evaluation (LowRankArena, Aug 2026; primary GitHub table; matched lm-eval protocol):**
  - **Llama-1-7B** (dense WT2 5.67, C4 7.20, 7-task MCQ avg 0.649):
    - 80% keep: Basis Sharing WT2 7.74 / C4 15.51 / MCQ 0.574; SVD-LLM v1 7.88 / 16.42 / 0.501.
    - 60% keep: Basis Sharing 12.42 / 41.13 / 0.465; SVD-LLM v1 13.74 / 55.03 / 0.399.
  - **Llama-3.1-8B** (dense WT2 6.24, C4 9.10, MCQ 0.716):
    - 80% keep: Basis Sharing 15.61 / 54.36 / 0.561.
    - 60% keep: Basis Sharing 82.96 / 461.21 / 0.396.
  - **Qwen3-8B-Base** (dense WT2 7.00, MCQ 0.704):
    - 80% keep: Basis Sharing 11.05 / 0.588.
    - 60% keep: Basis Sharing 18.59 / 0.469.
  - Within-layer methods beat it at the same keep ratio, e.g. MoDeGPT on Llama-3.1-8B at 80% keep: WT2 9.01, MCQ 0.590.
  - — [LowRankArena README](https://github.com/Zishan-Shao/lowrankarena) (raw README tables read); [paper arXiv 2608.26389](https://arxiv.org/abs/2608.26389)
- **LowRankArena conclusions:**
  - No SVD method leads on every backbone, and close orderings are calibration-sensitive.
  - Speedups show up mainly in compute-bound prefill, not in full generation.
  - At 70B, Basis Sharing "crash[ed] in CUDA eigensolvers".
  - — [LowRankArena arXiv 2608.26389 via search](https://arxiv.org/html/2608.26389v1)
- **Third-party numbers conflict:**
  - The D-Rank paper reports Basis Sharing on LLaMA-2-7B at 20% as PPL 7.57, against D-Rank's 7.51, and claims equal or higher accuracy than Basis Sharing at 20–50% on LLaMA-7B. — [D-Rank arXiv 2509.25622](https://arxiv.org/pdf/2509.25622)
  - Basis Sharing's own paper reports 7.70 for LLaMA-2-7B (7.57 is its Mistral-7B number), so one of these was likely transcribed wrongly.
- **Third-party healing data point:** a community checkpoint compresses Llama-2-7b-chat with the official code at keep 0.5999 and LoRA-heals only the coefficients. It reports WikiText-2 PPL 10.82. This is not the paper's LoRA recipe. — [HF model card Jeesup/svd-safety-llama2_7b_chat_up_basis_coeff…](https://huggingface.co/Jeesup/svd-safety-llama2_7b_chat_up_basis_coeff_jbbsft_finetuned_keep_0p60)
- **Follow-ups that reuse Basis Sharing's grouping protocol:** CoSpaDi uses the exact grouping on LLaMA-2-7B and claims it substantially outperforms Basis Sharing at all budgets (competitor claim). — [CoSpaDi arXiv 2509.22075](https://arxiv.org/html/2509.22075v4)
- **Negative 2026 analysis:**
  - A May 2026 paper places SVD-LLM, SVD-LLM V2 and Basis Sharing in one "cross-layer subspace coupling" optimization problem. Its best coupled setting lowers Frobenius reconstruction error vs SVD-LLM by 46.4% (Pythia-70M) and 37.2% (Pythia-1.4B; only attention out-proj compressed, d/D=0.25).
  - Downstream it is worse: Pythia-1.4B PPL 159.14 vs 50.17 for SVD-LLM V1. Total activation error was 5.76e8 vs 1.93e8.
  - The author's conclusion: Frobenius gains from cross-layer coupling do not carry over to language-modeling quality.
  - — [arXiv 2605.30836](https://arxiv.org/abs/2605.30836)

### Inferences
- **Math, as implied by the config and paper description:**
  - For a group G of n layers and a shared projection type (e.g. q_proj), stack the n matrices W^(i) (d_out×d_in each). Whiten with calibration activations as in SVD-LLM, then take a truncated SVD of the stacked whitened matrix.
  - Result: W^(i) ≈ B · C^(i). B (rank k) is shared by all n layers; C^(i) is the per-layer coefficient block, un-whitened by S^(-1).
  - Storage per group: d_out·k + n·k·d_in, vs n·d_out·d_in dense. Fitting is one eigendecomposition/Cholesky plus one SVD per group, with no gradients.
  - Precisely which side carries the basis, and how whitening combines different layers' inputs, was not verified; that is in Gaps.
- **Inference-cost caveat (my arithmetic, not from the paper):**
  - At batch-1 decode the shared basis B is still read once per layer per token, so sharing cuts storage and VRAM but not bytes read per token.
  - Example, square d×d matrices with n=2 at 20% compression: 1.6d² = dk + 2kd, so k ≈ 0.53d. Each layer then reads B + C^(i) = 2·0.53d² ≈ 1.07d², more than the dense d².
  - With n=4: k = 0.64d and each layer reads ≈1.28d².
  - So Basis Sharing at low compression can be slower than dense in memory-bound decode. This fits LowRankArena's finding that speedups are concentrated in prefill.
  - Quantization, by contrast, cuts storage and per-token bandwidth together.
- **Quality against the target:** Basis Sharing at 20% removal (1.25×) is about +2.1 PPL on LLaMA-1-7B and +9.4 PPL on Llama-3.1-8B (LowRankArena). Newer, more heavily trained models tolerate cross-layer basis sharing much worse than LLaMA-1/2.

### Gaps
- I could not open the paper's main tables, so these are missing:
  - LLaMA-7B PPL at 30/40/50% (paper's own numbers);
  - zero-shot table;
  - LoRA-healed numbers;
  - throughput numbers;
  - exact whitening formula for a shared group (one Cholesky factor per layer applied before concatenation vs. a joint one);
  - the paper's stated reason for keeping o_proj/down_proj private. My recollection is that sharing them gave larger loss; unverified.
- No sourced 4-bit stacking result for Basis Sharing specifically.

---

## Q2. DeltaLLM (2025) and the "shared base + per-layer low-rank delta" family (Relaxed Recursive Transformers, FlexiGPT, LiSA)

### Takeaway
Writing W_l ≈ W_shared + B_l·A_l is the closest existing post-hoc version of "looped transformer + per-loop LoRA". Every published instance needs some training of the deltas, from tens of millions of tokens (DeltaLLM) to 15–60B tokens (Relaxed Recursive Transformers). In return it gets either larger unique-parameter reductions (2× for RRT) or better retention than training-free sharing at modest ratios (12–24% for DeltaLLM).

### Cited Findings
- **DeltaLLM** (Mikaelyan et al., arXiv 2501.18596, Jan 2025):
  - Method: later Transformer blocks reuse weights from earlier blocks and add a low-rank difference matrix ("delta") per shared layer. The authors call it a "post-training compression technique".
  - Training: the low-rank modules are trained on ~30M–40M tokens with progressive module replacement, a probability scheduler that swaps original blocks for DeltaLLM blocks.
  - Released models are DeltaLLAMA and DeltaPHI, with a 12% parameter reduction that retains 90% of base performance on knowledge/reasoning benchmarks.
  - They outperform JointDrop, SliceGPT, ShortGPT and LaCo at equal removed parameters.
  - — [arXiv abs](https://arxiv.org/abs/2501.18596); [HTML v1](https://arxiv.org/html/2501.18596v1)
- **DeltaLLM details:**
  - Applied to Phi-3.5 and Llama-3.2. The replacement rate varies per block, because later layers are more redundant.
  - Variants: delta-only tuning (base frozen), or deltas plus LoRA on the shared base.
  - DeltaPhi 2.9B (24% reduction) scores like DeltaPhi 3.35B (12% reduction) on MMLU and WinoGrande.
  - The delta layers in DeltaPhi 2.9B total only 90 MB.
  - On UltraChat perplexity (paper's Table 7), DeltaPhi beats the original Phi-3.5. One summary instead reports Alpaca PPL 3.34 (DeltaPhi 3.35B) vs 2.96 (Phi-3.5); datasets differ.
  - — [arXiv HTML](https://arxiv.org/html/2501.18596v1); [Moonlight review](https://www.themoonlight.io/en/review/deltallm-compress-llms-with-low-rank-deltas-between-shared-weights)
- **Relaxed Recursive Transformers (RRT)** (Bae, Fisch, Harutyunyan, Ji, Kim, Schuster; arXiv 2410.20672; ICLR 2025):
  - Converts a pretrained LLM into a recursive model with a single block of unique layers repeated in a loop. Initialization: "Stepwise" (pick layers at fixed intervals), "Average", or "Lower".
  - The relaxed variant adds depth-wise (per-loop) LoRA.
  - With 15B tokens of uptraining, a recursive Gemma 1B (built from Gemma 2B) beats TinyLlama 1.1B and Pythia 1B.
  - It gains up to 13.5 points of few-shot accuracy over a same-size Gemma 1B trained from scratch (22% error reduction).
  - With KD and 60B uptraining tokens it approaches full Gemma 2B (pretrained on 3T tokens). A secondary summary reports 58.4% vs 58.6% average accuracy; not verified in the paper.
  - Stepwise init is best for strict tying. Average init is best once LoRA relaxation is added.
  - — [arXiv 2410.20672](https://arxiv.org/abs/2410.20672); [mlanthology ICLR 2025](https://mlanthology.org/iclr/2025/bae2025iclr-relaxed/); [Liner review](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora)
- **RRT closed-form initialization** (replication repo, primary GitHub):
  - Residual: R^l = W^l − W'_{((l−1) mod L/B)+1}.
  - LoRA init by truncated SVD: B^l = U_r Σ_r, A^l = V_rᵀ.
  - Both W' and the LoRA factors are then trained.
  - — [avramdj/rrt-lora README](https://github.com/avramdj/rrt-lora)
- **RRT throughput:** a secondary blog says recursive Gemma kept 98.5% of accuracy with half the parameters and ran >2× faster, using continuous depth-wise batching with early exit. This is unverified against the paper. — [Ajith Prabhakar blog](https://ajithp.com/2024/10/29/recursive-transformers/)
- **Follow-up:** "Improving Recursive Transformers with Mixture of LoRAs" (arXiv 2512.12880, Dec 2025). Found but not read. — [arXiv 2512.12880](https://arxiv.org/pdf/2512.12880)
- **FlexiGPT** (Samsung Research America, NAACL 2025, arXiv 2501.14713):
  - Prunes blocks by importance and replaces each with a weight-shared copy of an unpruned block plus block-specific low-rank adapters (SVD-initialized) and output normalization.
  - SOTA on 5/6 benchmarks at 30% compression and 6/6 at 40% on LLaMA-2-7B.
  - A secondary summary gives, at 30%: PPL 6.55 and 62.68% avg zero-shot after 1B recovery tokens, vs 6.71 / 61.40% for the best pruning+LoRA baseline.
  - Llama-3-8B is much more sensitive to pruning than Llama-2-7B.
  - — [ACL Anthology](https://aclanthology.org/2025.naacl-long.31/); [arXiv](https://arxiv.org/html/2501.14713v1)
- **LiSA, "Cross-layer Attention Sharing for Pre-trained LLMs"** (Mu et al., arXiv 2408.01890; TACL 2026):
  - Replaces self-attention in trained LLMs: tiny FFNs align attention heads between adjacent layers, and low-rank matrices approximate the layer-to-layer differences in attention weights.
  - Applied in 53–84% of layers, with Q and K compressed 6×.
  - Max throughput gains: +19.5% (LLaMA3-8B), +32.3% (LLaMA2-7B), +40.1% (LLaMA2-13B).
  - Training: uptraining on 4.2B tokens sampled from RedPajama subsets, updating only 0.46–1.64% of parameters.
  - Direct sharing without head re-arrangement "proves to be ineffective", and shallow layers are vulnerable.
  - — [LiSA README (primary GitHub)](https://github.com/takagi97/lisa); [ACL Anthology TACL](https://aclanthology.org/2026.tacl-1.30/)

### Inferences
- The family differs mainly in what is trained afterwards:
  - DeltaLLM: deltas only, ~3e7 tokens.
  - FlexiGPT: 1e9 tokens.
  - LiSA: 4e9 tokens.
  - RRT: 1.5e10–6e10 tokens.
- The more aggressive the tying (RRT ties every layer to one block, giving ~2× fewer unique params), the more uptraining is required.
- RRT's SVD-of-residual init is a training-free starting point. W_l ≈ W_shared + SVD_r(W_l − W_shared) is exactly the "looped model + per-loop low-rank correction" estimator.
- A post-training "looped quantizer" could use this init but with activation-whitened truncation (SVD-LLM style) instead of plain SVD. No surveyed paper reports that combination's quality without uptraining.

### Gaps
- No released code found for DeltaLLM. Its exact per-model tables were not retrieved: zero-shot deltas, which layers share, delta ranks.
- RRT's quality at initialization, before uptraining, was not found. It is the number most relevant to a "no-training" looped compressor.
- RRT official code: only an unofficial replication was found.

---

## Q3. MoS (Mixture of Shards) and ShareLoRA: adapter sharing, not base-weight compression

### Takeaway
Both methods share LoRA adapter parameters across layers to make fine-tuning or serving cheaper; neither compresses the pretrained base weights. They matter here only as design patterns for the per-layer delta part of a shared-base scheme: global shard pools, routing, and A/B-only sharing.

### Cited Findings
- **MoS** ("MoS: Unleashing Parameter Efficiency of Low-Rank Adaptation with Mixture of Shards", arXiv 2410.00938, ICLR 2025 poster):
  - Mixes inter-layer and intra-layer sharing of LoRA parameters.
  - Shards are selected from global pools by an MoE-like router and concatenated into the low-rank matrices.
  - Four "nearly cost-free differentiation strategies": subset selection, pair dissociation, vector sharding, shard privatization.
  - ~8× parameter savings vs standard LoRA. The authors find that "differentiation" is indispensable for reversing the harm of pure sharing.
  - Evaluated on LLaMA2-7B/13B and LLaMA3.2-3B fine-tuning.
  - — [MoS README (primary GitHub)](https://github.com/Forence1999/MoS)
- **ShareLoRA** (arXiv 2406.10785, UC Irvine): shares the LoRA A matrix, the B matrix, or both across layers. It reports 44%–96% fewer trainable parameters than LoRA, with savings growing with model size. Tested on RoBERTa, GPT-2 and LLaMA 1/2/3, with claimed robustness/transfer benefits. — [arXiv 2406.10785](https://arxiv.org/abs/2406.10785v2)

### Inferences
- MoS's lesson that pure sharing needs cheap differentiation mechanisms matches the base-weight literature: Basis Sharing has per-layer coefficients, LiSA aligns heads, RRT has per-loop LoRA.
- MoS-style shard pools are a candidate parameterization for the per-layer residuals of a looped compressor.

### Gaps
- Neither paper tests compressing the frozen base, so no base-weight compression numbers exist for them.

---

## Q4. Dictionary learning / sparse coding, cross-layer attention sharing, tensor decompositions across layers, hashing

### Takeaway
- **Training-free and calibration-guided:** dictionary methods (CoSpaDi, GeoPair) are the most active 2025–2026 line. They claim to beat Basis Sharing, but only in competitor-reported numbers.
- **Trained-from-scratch atom sharing:** MASA cuts attention parameters 66.7% at parity.
- **Tensor decompositions over stacked layers:** applied post-hoc, these consistently underperform matrix methods and even RTN quantization (2026 evidence).
- **Hashing:** HashedNets-style hashing is a training-time technique only.

### Cited Findings
- **MASA, "Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning"** (arXiv 2508.04581, Aug 2025; AAAI 2026):
  - Q/K/V/O projections across layers are expressed as combinations of shared matrix "atoms", with per-layer mixing coefficients.
  - Main results are trained from scratch at 100M–700M parameters: 66.7% fewer attention parameters (e.g. 226.5M → 75M in a ~700M model) at on-par quality.
  - Beats GQA, low-rank baselines and "Repeat-all-over/Sequential" sharing at similar budgets.
  - A QKV-only variant (50% reduction) slightly beats the full model on average reasoning accuracy. ViTs match baseline with 66.7% fewer attention params.
  - O-projection is the bottleneck: compressing it hurts LM quality more.
  - For pretrained LLMs there is a post-hoc path (Matrix PCA for shared atoms plus local refinement), but no Llama numbers were retrievable.
  - Code reportedly at github.com/mts-ai/MASA.
  - — [arXiv abs](https://arxiv.org/abs/2508.04581); [HTML](https://arxiv.org/html/2508.04581v1); [paper note](https://en.papernotes.org/AAAI2026/interpretability/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear/)
- **CoSpaDi** (arXiv 2509.22075, Sep 2025, revised through v5):
  - Training-free. Each weight is factorized into a dense dictionary times column-sparse coefficients, so different columns use different atom subsets at the same storage.
  - Calibration turns functional (activation-space) reconstruction into a standard dictionary-learning problem.
  - Supports cross-layer dictionary sharing within groups of similar projections, reusing Basis Sharing's grouping protocol on LLaMA-2-7B.
  - Claims better accuracy/PPL vs compression than SVD-based and structured-pruning baselines at 20–40% (another source says 20–50%) on Llama and Qwen. Claims it "substantially outperforms Basis Sharing".
  - — [arXiv HTML v4](https://arxiv.org/html/2509.22075v4); [v5](https://arxiv.org/html/2509.22075v5)
- **GeoPair, "Geometry-Preserving Cross-Layer Factorization for Training-Free Transformer Compression"** (Mohammad, Ali, Lefkimmiatis; arXiv 2609.25963, Sep 2026):
  - Pairs layers by maximum-weight matching, solved with Edmonds' Blossom algorithm.
  - Shared-dictionary updates are closed-form, via a generalized Sylvester equation. Coefficients are structured-sparse via Hard Thresholding Pursuit with a CG solver.
  - Claims >90% of baseline accuracy "at high compression ratios without any fine-tuning", beating independent per-layer decompositions and other pairwise factorizations across architectures and modalities.
  - Limitation: pairwise grouping only.
  - — [arXiv 2609.25963](https://arxiv.org/abs/2609.25963v1)
- **FiPS, "Learning Fine-grained Parameter Sharing via Sparse Tensor Decomposition"** (later retitled "Learning Parameter Sharing with Tensor Decompositions and Sparsity"; arXiv 2411.09816):
  - A shared base plus sparse factors represent neurons across MLP modules in multiple layers.
  - Initialized by SVD, then optimized by block-wise reconstruction-error minimization (gradient-based, calibration).
  - DeiT-B and Swin-L MLPs compressed to 25–40% of original parameters within 1 pp accuracy.
  - Gemma-2 and Llama-3 MLP budgets reduced 40–50%.
  - Stacked with QAT: 3-bit FiPS on Gemma-2-2B gets lower PPL than 2-bit QAT alone at the same 8× compression.
  - — [arXiv 2411.09816](https://arxiv.org/abs/2411.09816); [FiPS repo](https://github.com/cemuyuk/FiPS)
- **Head-wise Shareable Attention (DirectShare/PostShare)** (Cao, Yang, Zhao; EMNLP 2024 Findings, arXiv 2402.11819):
  - DirectShare scores candidate head pairs by weight similarity and ties the top pairs to hit a target sharing ratio. It reuses pretrained weights with no retraining.
  - PostShare first post-trains with a similarity penalty.
  - A secondary review reports DirectShare kept ~99.5% of Llama2-7B reasoning performance at a 30% sharing ratio (unverified in paper tables).
  - Unofficial "init" code exists.
  - — [arXiv](https://arxiv.org/abs/2402.11819); [Liner review](https://liner.com/review/headwise-shareable-attention-for-large-language-models); [DirectShare repo](https://github.com/zouyingcao/DirectShare)
- **DSFormer** (Chand, Prabhu, Kumar; ICLR 2024, arXiv 2312.13211):
  - Per-matrix factorization into a small dense matrix times a semi-structured sparse matrix, tuned with a "Straight-Through Factorizer" for task accuracy. That is training, not calibration-only.
  - Up to 40% better compression than prior low-rank factorizers; argues that low-rank assumptions are too restrictive.
  - Not cross-layer. A third-party summary says it is orthogonal to (stackable with) layer-sharing and quantization.
  - — [arXiv 2312.13211](https://arxiv.org/abs/2312.13211)
- **MiniViT** (2022) shares weights across every K adjacent ViT layers (except LayerNorm) and adds unshared per-layer transforms ("weight multiplexing"). It is trained with distillation, so not post-hoc. — [arXiv 2204.07154](https://arxiv.org/pdf/2204.07154)
- **Tensor decompositions across layers:**
  - TRAWL stacks related matrices across layers and applies CP/Tucker as a training-free denoising intervention (as described in a 2026 paper's related work).
  - TensorLLM (arXiv 2501.15674) Tuckerizes multi-head attention, sharing factor matrices across heads within a layer.
  - "Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression" (arXiv 2606.03465, Jun 2026) finds that tensor decompositions (Tucker for MHA, TT for FFN) "consistently underperform matrix-based methods" and are "significantly worse than RTN quantization". Its explanation: Tucker/TT keep Frobenius mass but distort the head/neuron/expert-specific spectral subspaces. Early layers are the most sensitive.
  - Saten (arXiv 2505.14871) notes that directly TT-compressing pretrained weights causes huge accuracy drops, because many layers are not low-rank; it adds a sparse term.
  - — [arXiv 2606.03465](https://arxiv.org/pdf/2606.03465); [Saten](https://arxiv.org/pdf/2505.14871); [TensorLLM summary](https://pith.science/paper/2501.15674)
- **TensorGPT** (arXiv 2307.00526): training-free TT decomposition of the embedding layer only, up to 38.4× embedding compression on GPT-2. — [arXiv 2307.00526](https://arxiv.org/pdf/2307.00526)
- **LoRETTA** (NAACL 2024, arXiv 2402.11417): tensor-train adapters for PEFT, up to 100× fewer trainable params than common PEFT on LLaMA-2-7B. It is not base-weight compression. — [arXiv 2402.11417](https://arxiv.org/abs/2402.11417v1)
- **HashedNets** (Chen et al., ICML 2015): a hash function randomly groups connections into buckets that share one parameter, trained with backprop. There is no post-hoc variant for pretrained LLMs. — [PMLR](https://proceedings.mlr.press/v37/chenc15.html)

### Inferences
- The dictionary/sparse-coding line (CoSpaDi, GeoPair, FiPS) is the strongest post-hoc evolution of Basis Sharing. All three put sparsity in the per-layer coefficients and fit against activation-weighted error (calibration), not raw weights.
- This agrees with the 2026 negative result that Frobenius-optimal cross-layer coupling hurts perplexity. The objective must live in activation space.
- Hashing and tensor-on-layer-axis approaches look like dead ends for post-hoc compression of modern LLMs.

### Gaps
- No exact PPL/accuracy tables were retrieved for CoSpaDi's cross-layer variant, GeoPair, MASA's post-hoc Llama path, or FiPS's LLM tables.
- I found no paper titled "Cross-layer dictionary" as such, and no post-hoc "DictFormer"-style result. DictFormer (shared dictionary across layers, trained) did not surface in search.

---

## Q5. SVD-based post-training baselines (SVD-LLM v1/v2, ASVD, FWSVD, Dobi-SVD) and why whitening makes truncation closed form

### Takeaway
Activation-aware SVD baselines remove 20% of LLaMA-7B parameters at about +2.2 PPL (SVD-LLM 7.94 vs 5.68). They degrade sharply at 40% and beyond, and dramatically on Llama-3.1-8B. The whitening step is what lets truncation be solved in closed form against activation-space error. Every cross-layer method surveyed builds on it.

### Cited Findings
- **SVD-LLM** (Wang, Zheng, Wan, Zhang; ICLR 2025):
  - Pipeline: (1) truncation-aware data whitening + SVD compression; (2) optional "parameter update with sequential low-rank approximation", which LoRA-finetunes U then V on Alpaca (r=8, 2 epochs, lr 1e-4, batch 64). Recommended ratio ≤0.3. Also integrates with GPTQ.
  - — [SVD-LLM README (primary GitHub)](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- **SVD-LLM, LLaMA-7B at 20% compression:** WikiText-2 7.94, PTB 18.05, C4 15.93; ASVD 11.14 on WikiText-2. — [SVD-LLM ICLR PDF via search](https://proceedings.iclr.cc/paper_files/paper/2025/file/3104e1ab39875cf54fe1eb4473e7c5a1-Paper-Conference.pdf); [arXiv 2403.07378](https://arxiv.org/pdf/2403.07378)
- **SVD-LLM V2** (NAACL 2025, arXiv 2503.12340):
  - Assigns a per-matrix compression ratio from theoretical truncation loss (heterogeneous allocation) and optimizes truncation.
  - Evaluated on LLaMA-7B, OPT-6.7B and LLaMA-3-8B at 20% across 10 datasets.
  - A secondary summary claims up to 42% lower PPL and +9% accuracy vs SVD-LLM on LLaMA-3-8B.
  - — [arXiv 2503.12340](https://arxiv.org/pdf/2503.12340); [Liner review](https://liner.com/review/svdllm-v2-optimizing-singular-value-truncation-for-large-language-model)
- **ASVD** (arXiv 2312.05821):
  - Training-free. Scales weights by the activation distribution to handle outliers, and calibrates layer sensitivity iteratively ("several hours" to compute sensitivities).
  - Claims 10–20% compression "without losing reasoning capacities". Released Llama-2-7B/13B at 95/90/85% param ratios, quantizable with bnb 4/8-bit.
  - — [ASVD README (primary GitHub)](https://github.com/hahnyuan/ASVD4LLM)
- **Dobi-SVD** (ICLR 2025, arXiv 2502.02723; differentiable truncation positions):
  - The ICLR poster reports WikiText-2 PPL 9.07 on LLaMA-7B at 40% parameter compression, "78.7% improvement" over prior SVD SOTA.
  - Another excerpt reports 9.70 at "ratio 0.4" vs 43,104 (ASVD) and 458 (SVD-LLM). The ratio convention and the exact figure conflict between sources.
  - Stacked with 4-bit: on Llama-2-7B, 4-bit BnB = 3.2 GB / PPL 6.97, and 4-bit BnB + Dobi-SVD(0.8) = 3.0 GB / PPL 6.91.
  - Official repo says "Full code coming soon".
  - — [ICLR poster](https://iclr.cc/virtual/2025/poster/28553); [arXiv 2502.02723](https://arxiv.org/pdf/2502.02723); [Dobi-SVD README](https://github.com/wangqinsi1/Dobi-SVD)
- **Matched-protocol reality check (LowRankArena, primary GitHub):**
  - Llama-1-7B, 80% keep, WT2 / MCQ avg:
    - ASVD 8.60 / 0.609
    - SVD-LLM v1 7.88 / 0.501
    - DoBi-SVD 9.23 / 0.526
    - Basis Sharing 7.74 / 0.574
    - MoDeGPT 6.92 / 0.552
    - AA-SVD 6.89 / 0.583
    - ZS-SVD 6.69 / 0.603
  - Llama-3.1-8B at 80% keep:
    - ASVD collapses (2011 PPL) and DoBi-SVD collapses (556.6).
    - SVD-LLM v1 14.83; Basis Sharing 15.61; ZS-SVD 9.78; MoDeGPT 9.01 (dense 6.24).
  - LowRankArena also could not reproduce SAES-SVD's paper numbers on LLaMA-7B.
  - — [LowRankArena README](https://github.com/Zishan-Shao/lowrankarena)
- **SVD + quantization stacking (SVD-LLM paper, LLaMA-7B WikiText-2):**
  - GPTQ-4bit: 3.9 GB, PPL 6.21.
  - GPTQ-3bit: 2.8 GB, PPL 16.28.
  - SVD-LLM(20%) + GPTQ-4bit: 2.1 GB, PPL 13.29. That beats GPTQ-3bit, but is ~2× worse PPL than GPTQ-4bit alone.
  - SVD-LLM V2(30%) + GPTQ-4bit beats GPTQ-3bit at the same memory.
  - — [SVD-LLM arXiv 2403.07378](https://arxiv.org/pdf/2403.07378); [SVD-LLM V2](https://arxiv.org/pdf/2503.12340)

### Inferences
- **Background on whitening** (from prior knowledge of the SVD-LLM paper; not re-fetched because arXiv was blocked):
  - With calibration inputs X, compute S such that S·Sᵀ = X·Xᵀ (Cholesky), then take the SVD of W·S.
  - Truncating the smallest singular values of W·S gives the optimal rank-k approximation under the activation loss ‖(W − W')X‖_F, and each dropped singular value's contribution to that loss equals that singular value. That is the "truncation-aware" closed form.
  - Weights: W' = U_k Σ_k (V_kᵀ S⁻¹).
  - ASVD instead uses a diagonal activation-magnitude scaling, which is not exact.
  - FWSVD (Hsu et al., ICLR 2022) weights rows by Fisher information, which needs gradients on task data.
  - Basis Sharing applies the same whitening to the stacked group.
- Combining any SVD-family method with 4-bit quantization gave at best a few-percent memory reduction at equal PPL (Dobi-SVD + BnB: 3.2 → 3.0 GB, PPL 6.97 → 6.91). SVD-LLM + GPTQ4 instead traded large PPL increases for ~1.9× smaller memory.

### Gaps
- FWSVD numbers were not retrieved; its LLM-scale numbers appear only as baselines inside Basis Sharing/SVD-LLM tables, which I could not open.
- SVD-LLM V2's exact LLaMA-3-8B table was not retrieved.
- The Dobi-SVD figure conflicts (9.07 vs 9.70) and its ratio convention was not resolved.

---

## Q6. Post-hoc averaging/merging of layer weights measured without retraining (brief; layer merging such as LaCo, MKA and FuseGPT is covered elsewhere)

### Takeaway
Permutation-aligned merging of FFN sublayers can remove a third of FFN parameters at modest cost in small models. RRT's "Average" initialization is useful mainly as a starting point for uptraining. No surveyed work shows naive averaging of LLM layers without alignment surviving without retraining.

### Cited Findings
- **"Merging Feed-Forward Sublayers for Compressed Transformers"** (Verma et al., arXiv 2501.06126, Jan 2025):
  - Uses the first FFN sublayer as an anchor, computes permutations aligning the other sublayers to it, then merges (averages) them.
  - Reports removal of 1/3, 1/2 and (n−1)/n of FFN sublayers, plus a "Vanilla" no-permutation variant. Removes >21% of total ViT parameters at 99% of original performance.
  - A third-party review says merging 12 of 36 GPT-2 sublayers raises test PPL by ~1 point.
  - — [arXiv 2501.06126](https://arxiv.org/abs/2501.06126); [Pith review](https://pith.science/paper/2501.06126)
- **RRT:** strict tying works best with "Stepwise" init (pick every k-th layer). With LoRA relaxation, "Average" init is best, which shows that averaged weights plus low-rank residuals are a good basis. Both still require uptraining. — [Liner review](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora); [arXiv 2410.20672](https://arxiv.org/abs/2410.20672)
- **LiSA:** directly sharing attention weights without re-arranging heads "proves to be ineffective". Permutation symmetry has to be resolved before sharing. — [LiSA README](https://github.com/takagi97/lisa)
- **Retrofitted recurrence** (McLeish et al., arXiv 2511.07384, NeurIPS 2025):
  - Converts pretrained LMs into prelude / recurrent block / coda, removing middle layers, then trains with a recurrence curriculum.
  - Aimed at better math performance per compute, not compression. Code: github.com/mcleish7/retrofitting-recurrence.
  - — [arXiv 2511.07384](https://arxiv.org/html/2511.07384v1)

### Inferences
- Alignment of permutation symmetries (FFN neurons, attention heads) before sharing is a recurring prerequisite: Verma et al., LiSA, MCWC (Q7). A post-hoc looped compressor should include a permutation-alignment step before computing shared blocks.

### Gaps
- Whether Verma et al. fine-tune after merging, and their LLM-scale (>1B) numbers, were not confirmed.

---

## Q7. 2025–2026 work that frames cross-layer sharing as a quantization/codebook problem, or stacks it with int4/int8

### Takeaway
The most "quantization-like" cross-layer results come from 2026 preprints:
- MCWC treats depth as video: align layers, predict each from earlier decoded layers, then quantize and entropy-code the residuals. It beats GPTQ-W4 at ~4.2 bits/param.
- LegoLM uses a single global (model-wide) codebook and claims 4.41× on Mistral-7B at +0.03% PPL.

Both are unreviewed single-group preprints. Stacking classic cross-layer/SVD methods on top of 4-bit gives small or negative returns.

### Cited Findings
- **MCWC, Motion-Compensated Weight Compression** (Lamaakal, arXiv 2605.24754, May 2026):
  - A weight-only codec. It first aligns permutation-symmetric blocks (hidden units, attention heads) to maximize cross-layer correspondence.
  - "Keyframe" layers are coded directly; later layers are predicted from already-decoded layers. Only quantized prediction residuals are coded, under a learned entropy model with a rate-distortion objective.
  - Ablations show alignment, prediction, entropy model and keyframe scheduling are each necessary.
  - Bitstream: keyframes ~25.8% (6.20e8 bits), residuals ~64.6% (1.55e9 bits).
  - Transformer LM results (model unnamed in the excerpt):
    - PPL 15.33 at 16.6 bits/param.
    - PPL 15.95 at ~4.2–4.3 bits, vs 16.85 uniform INT4 PTQ and 16.40 GPTQ-W4.
    - PPL 16.40 at 2.8 bits.
    - Prune+Quant baseline: 18.10 at 2.7 bits.
  - Vision: 81.72% at 16.5 bits → 80.15% at 2.8 bits (excerpt partly corrupted).
  - — [arXiv HTML 2605.24754](https://arxiv.org/html/2605.24754v1)
- **LegoLM, "Structured Weight Sharing for LLMs"** (J. Bingham, arXiv 2608.08652, 9 Aug 2026):
  - "Global weight sharing" with a single model-wide scalar codebook (Lloyd-Max centroids, K up to 128).
  - Three data-free fixes:
    - scalar-block encoding (d=1);
    - percentile-selective replacement, keeping outliers verbatim (~1/K of weights lie beyond the outer threshold);
    - boundary-layer protection, leaving the first and last blocks alone.
  - Mistral-7B results:
    - +0.03% PPL at 4.41×, beating PTQ-8bit in both quality and ratio;
    - −0.02% at 2.67×;
    - downstream LAMBADA/HellaSwag within noise at 5.12× (K=64, p=99%);
    - +14.24% PPL at 9.74× with 1% outliers kept.
  - Full replacement at K=128: +23% PPL on GPT-2 small but +1,134,279% on Mistral-7B. Outlier dominance grows with scale.
  - Data-free, under 30 min on one GPU for Mistral-7B. A third-party review calls the theory internally inconsistent and the headline number test-set-selected.
  - — [arXiv 2608.08652](https://arxiv.org/abs/2608.08652); [Pith review](https://pith.science/paper/2608.08652)
- **FiPS + QAT:** 3-bit FiPS (cross-layer shared base + sparse factors) on Gemma-2-2B beats 2-bit QAT alone at the same 8× compression. This needs QAT training. — [arXiv 2411.09816](https://arxiv.org/abs/2411.09816)
- **SWSC** (Zeng et al., arXiv 2501.08631, Jan 2025):
  - K-means clusters similar channels within a matrix (not across layers) into representative vectors, plus an SVD error-compensation term.
  - A secondary review reports Llama-2-7B Q/K projections at WikiText-2 PPL ~6.5–10.9 at average 3- and 2-bit budgets, vs RTN in the thousands or NaN.
  - — [arXiv 2501.08631](https://arxiv.org/abs/2501.08631); [Pith review](https://pith.science/paper/2501.08631)
- **Search for shared VQ codebooks across layers:** I found no 2025–2026 paper that shares a vector-quantization codebook across layers. Closest:
  - QAM-W (2026): one Lloyd-Max codebook for all MLP rows.
  - XFP (2026): a per-layer library of 32 codebooks.
  - AQLM / CRVQ: per-layer multi-codebooks.
  - — [QAM-W arXiv 2605.26339](https://arxiv.org/html/2605.26339v1); [XFP arXiv 2605.14844](https://arxiv.org/html/2605.14844v1); [CRVQ TACL](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.45/133863/CRVQ-Channel-Relaxed-Vector-Quantization-for)
- **Stacking with 4-bit:**
  - SVD-LLM(20%) + GPTQ-4bit on LLaMA-7B: 2.1 GB / PPL 13.29, vs GPTQ-4bit 3.9 GB / 6.21 and GPTQ-3bit 2.8 GB / 16.28. — [SVD-LLM arXiv 2403.07378](https://arxiv.org/pdf/2403.07378)
  - Dobi-SVD(0.8) + BnB-4bit on Llama-2-7B: 3.0 GB / 6.91, vs BnB-4bit 3.2 GB / 6.97. — [Dobi-SVD arXiv 2502.02723](https://arxiv.org/pdf/2502.02723)
  - ASVD releases are designed to be loaded with bnb NF4/8-bit. — [ASVD README](https://github.com/hahnyuan/ASVD4LLM)

### Inferences
- The codebook/predictive-coding framing turns "looped weights" into "a keyframe layer plus quantized cross-layer residuals". This is the most promising way to make cross-layer redundancy additive with quantization rather than competing with it.
- The catch: MCWC/LegoLM-style codecs reduce storage but must decode weights at load or run time. Unless decode is fused into the GEMM kernels, they do not by themselves reduce per-token bandwidth the way int4 GEMM does.
- LegoLM's "global weight sharing" is effectively non-uniform scalar quantization with one model-wide codebook. It is not layer reuse in the looped-transformer sense, and its compression comes from bit-width, not depth sharing.

### Gaps
- No primary tables were retrieved for MCWC (which LM, which bit accounting) or LegoLM (baseline PTQ configuration).
- No independent reproductions of either.
- No paper found that stacks Basis Sharing, CoSpaDi or GeoPair with GPTQ/AWQ int4.

---

## Q8. Realistic compression from cross-layer sharing alone at <1–2 PPL loss without finetuning, compared with 4-bit quantization

### Takeaway
Training-free cross-layer sharing alone does not reach 1.25× (20% removal) within 2 PPL on any model with sourced numbers. It is +2.1 PPL on LLaMA-1-7B and +2.2–2.3 PPL on LLaMA-2-7B/Mistral-7B (paper), and +9.4 PPL on Llama-3.1-8B (independent benchmark). A realistic training-free target at <1–2 PPL is ~1.1–1.2× on older models and less on modern ones. 4-bit GPTQ gives ~3.5–4× at ~+0.5 PPL (6.21 vs 5.68 on LLaMA-7B). Sharing therefore makes sense only as an additional factor on top of quantization, or with healing: DeltaLLM 12–24% with ~30–40M tokens; RRT ~2× unique params with 15–60B tokens.

### Cited Findings
- 4-bit reference, LLaMA-7B WikiText-2: GPTQ-4bit 3.9 GB, PPL 6.21. — [SVD-LLM arXiv 2403.07378](https://arxiv.org/pdf/2403.07378)
- Dense LLaMA-1-7B is 5.67 (WT2) / 7.20 (C4) under the LowRankArena protocol. — [LowRankArena](https://github.com/Zishan-Shao/lowrankarena)
- Best training-free cross-layer method in the matched benchmark (Basis Sharing):
  - 80% keep: 7.74 on LLaMA-1-7B (+2.07), 15.61 on Llama-3.1-8B (+9.37), 11.05 on Qwen3-8B (+4.05).
  - 60% keep: 12.42 / 82.96 / 18.59.
  - Best within-layer methods at 80% keep on Llama-3.1-8B: MoDeGPT 9.01, ZS-SVD 9.78.
  - — [LowRankArena](https://github.com/Zishan-Shao/lowrankarena)
- Basis Sharing's own 20% numbers: LLaMA-2-7B 7.70, Mistral-7B 7.57, OPT-6.7B 11.79. — [OpenReview](https://openreview.net/forum?id=gp32jvUquq)
- LegoLM claims 4.41× at +0.03% PPL on Mistral-7B, but via a global scalar codebook (quantization), not layer reuse; unreviewed. — [arXiv 2608.08652](https://arxiv.org/abs/2608.08652)
- MCWC (cross-layer predictive coding + quantized residuals) beats GPTQ-W4 at ~4.2 bits (15.95 vs 16.40, dense 15.33). — [arXiv 2605.24754](https://arxiv.org/html/2605.24754v1)
- Healed sharing:
  - DeltaLLM: 12% reduction at 90% retained performance, ~30–40M training tokens. — [arXiv 2501.18596](https://arxiv.org/abs/2501.18596)
  - RRT: 2B → 1B-unique recursive Gemma, 15B–60B uptraining tokens. — [arXiv 2410.20672](https://arxiv.org/abs/2410.20672)
  - FlexiGPT: 30% at PPL 6.55 with 1B tokens (secondary). — [ACL 2025](https://aclanthology.org/2025.naacl-long.31/)
  - LiSA: 6× Q/K in 53–84% of layers with 4.2B tokens. — [LiSA](https://github.com/takagi97/lisa)
- Cross-layer coupling optimized for Frobenius error can hurt badly (Pythia-1.4B PPL 159 vs 50). — [arXiv 2605.30836](https://arxiv.org/abs/2605.30836)

### Inferences
- **Byte-for-byte, quantization dominates.** 1.25× from sharing at +2 PPL is ~20× worse than 4-bit's ~4× at +0.5 PPL.
- **"Looped weights as quantization" is more viable when:**
  1. it is applied on top of int4, as a residual/delta coder across depth (MCWC style), not instead of int4;
  2. the shared component is fitted in whitened activation space with permutation alignment;
  3. per-layer residuals are low-rank or sparse and themselves quantized;
  4. a short healing budget (1e7–1e9 tokens) is allowed. DeltaLLM shows 3e7 tokens suffices at 12–24%.
- **Bandwidth caveat:** pure sharing reduces storage, not per-token weight bandwidth at batch-1 decode, unless the shared block stays cache-resident or tokens are batched across depth (RRT's continuous depth-wise batching). So its speed benefit is not comparable to int4 GEMM kernels.
- **Model vintage matters:** more heavily trained models (Llama-3.1, Qwen3) are much less tolerant of all low-rank/sharing methods than LLaMA-1/2. Claims made on LLaMA-1-7B overstate what is achievable on current models.

### Gaps
- No paper found that reports a training-free cross-layer method stacked on GPTQ/AWQ int4 for a modern 7–8B model with matched-protocol PPL. That is the experiment that would settle whether sharing adds a meaningful factor beyond 4×.
- No ViT post-training cross-layer results with exact numbers were retrieved, beyond FiPS's MLP-only figures and the ViT FFN-merging result.
