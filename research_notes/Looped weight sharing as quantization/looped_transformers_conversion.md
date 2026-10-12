# Looped / recursive / weight-shared transformers: converting pretrained models and how much retraining it costs

Source-access note (applies to every section): arxiv.org, openreview.net, iclr.cc and huggingface.co could not be fetched from this environment (DNS and proxy 403). Paper numbers come from three kinds of source, and each bullet says which:
- **(search excerpt)**: a search engine's quotation or summary of the primary paper page (arXiv HTML/PDF). These are usually exact quotes, but I could not open the full table myself.
- **(repo)**: read directly from the authors' official GitHub repositories (READMEs, conversion scripts, released evaluation data).
- **(catalog)**: the Awesome-Loop-Models catalog (https://github.com/huskydoge/Awesome-Loop-Models, 299 papers, auto-built 2026-10-12). Each entry has a one-line TL;DR, and some have primary-source audit records.

Terminology used below:
- "from scratch": pretraining a looped model.
- "uptraining": continued pretraining of a converted pretrained model on generic tokens.
- "distillation": KD from the original model.
- "zero-training": no gradient updates at all.

## 1. Relaxed Recursive Transformers (Bae et al., KAIST/Google DeepMind; arXiv 2410.20672, Oct 2024; ICLR 2025)

### Takeaway
RRT converts a pretrained LLM into a CYCLE-shared looped model by default: one block of L/B layers repeated B times, with B=2 (e.g. Gemma 2B's 18 layers become one 9-layer block looped twice). It then needs real uptraining: 15B tokens for the initialization comparisons and 60B tokens plus KD for the best results. With per-layer LoRA initialized by truncated SVD of (original − shared) and rank 512, relaxed Gemma reaches 58.4% vs 58.6% for the original 2B. I found no published number for the converted model at zero uptraining.

### Cited Findings
- **Architecture.** The default "Recursive Transformer" with L layers and B blocks computes h^l = f(h^{l−1}; Φ'_{((l−1) mod L/B)+1}). Index (l−1) mod L/B is the CYCLE pattern A→B→…→A→B…. This is the same pattern as the user's A→B→A→B→A→B. — [rrt-lora replication README (repo)](https://github.com/avramdj/rrt-lora); [RRT paper](https://arxiv.org/abs/2410.20672)
- **Default loop factor.** B=2: "Gemma 2B with 18 layers can be converted to a recursive variant with 2 blocks by storing weights for only the first 9 layers". Results for Gemma with B=3 are in the appendix, but I could not retrieve them (search excerpt). — [arXiv HTML v1](https://arxiv.org/html/2410.20672v1)
- **Base models.** Gemma 2B, TinyLlama 1.1B and Pythia 1B. Uptraining used 15B to 60B SlimPajama tokens, and evaluation used seven few-shot benchmarks (search excerpt). — [arXiv PDF](https://arxiv.org/pdf/2410.20672)
- **Initialization of the shared block (three strategies).**
  - **Stepwise** "selects intermediate layers at specific intervals while keeping the first and last layer fixed" (search excerpt). The author's own summary says Stepwise selects pretrained layers at fixed intervals with the first and last kept. — [arXiv HTML](https://arxiv.org/html/2410.20672v1); [S. Bae on X](https://x.com/raymin0223/status/1851216065831076292)
  - **Average** initializes each shared layer as the mean of the layers tied to it (search excerpt/summary). — [themoonlight review](https://www.themoonlight.io/en/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layer-wise-lora)
  - **Lower** "directly uses weights from the first K layers of the unshared model" (secondary summary). — [themoonlight review](https://www.themoonlight.io/en/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layer-wise-lora)
  - **Winners.** After a short 15B-token uptraining, Stepwise is best for plain (non-relaxed) recursive models. Average is best for relaxed models, and Average-initialized relaxed models "even outperform[ed] the non-relaxed model initialized with Stepwise" (search excerpt; Figure 5 of the paper). — [arXiv HTML](https://arxiv.org/html/2410.20672v1)
- **Per-layer LoRA initialization.**
  - For every tied position l, compute the residual R^l = W^l − W'_{shared}. The truncated SVD at rank r gives B^l = U_r Σ_r and A^l = V_r^T. The forward pass is h = W'x + B^l A^l x, and both W' and the LoRA factors are trained. — [rrt-lora README (repo)](https://github.com/avramdj/rrt-lora); confirmed by the paper text "initialized by computing truncated SVD on the residual matrices between original layer weights and shared weights" (search excerpt) — [arXiv PDF](https://arxiv.org/pdf/2410.20672)
  - The authors say approximating these residuals "appears readily achievable using truncated SVD with low ranks" (search excerpt). — [arXiv HTML](https://arxiv.org/html/2410.20672v1)
- **LoRA ranks tried:** 64, 128, 256 and 512 ("We show the performance of four different rank values", search excerpt). One secondary summary reports SVD initialization giving up to +6.5 points over zero-initialized LoRA. — [arXiv PDF](https://arxiv.org/pdf/2410.20672); [Liner review](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora)
- **Headline quality numbers.**
  - Relaxed Gemma with rank 512, uptrained on 60B tokens: 58.4% few-shot average vs 58.6% for the original Gemma 2B pretrained on 3T tokens (search excerpt). — [arXiv HTML](https://arxiv.org/html/2410.20672v1)
  - Recursive Gemma 1B beats reduced-size non-recursive baselines with equal parameters by up to 13.5 points (51.7% vs 38.2%), a "22% error reduction" (search excerpt). — [arXiv PDF](https://arxiv.org/pdf/2410.20672)
  - Recursive Gemma 1B "outperforms both similar-sized vanilla pretrained models (such as TinyLlama 1.1B and Pythia 1B) and knowledge distillation baselines" and "can even recover most of the performance of the original 'full-size' model". Secondary summaries add that this holds with only 15B uptraining tokens, even though Pythia was pretrained on 300B tokens. — [arXiv HTML](https://arxiv.org/html/2410.20672v1); [DeepMind publication page](https://deepmind.google/research/publications/relaxed-recursive-transformers-effective-parameter-sharing-with-layer-wise-lora/)
  - A recursive TinyLlama uptrained on 15B tokens "yields few-shot performance comparable to the original model pretrained on 105 billion tokens" (secondary summary; not verified in the table). — [Liner review](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora)
- **Distillation.** The best Gemma results combine extended uptraining (60B tokens) with knowledge distillation, and the text says these "yielded substantial accuracy gains". I could not retrieve the exact KD-vs-no-KD delta (search excerpt was truncated). — [arXiv PDF](https://arxiv.org/pdf/2410.20672)
- **Throughput.** "Continuous depth-wise batching" combined with early exit is projected to give 2–3× throughput. This comes from a simulation with oracle exits, not a measured deployment speedup (catalog primary-source audit, verified 2026-07-10). — [Awesome-Loop-Models entry 2410.20672](https://github.com/huskydoge/Awesome-Loop-Models)

### Inferences
- The uptraining budget relative to the original pretraining is about 0.5%–2% of Gemma's 3T tokens (15–60B). That is cheap compared with pretraining, but it is billions of tokens, not post-training-quantization (PTQ)-scale calibration.
- RRT's SVD-of-residual initialization is a closed-form, data-free step that could be the "stage 0" of a near-post-training pipeline. The paper reports results only after uptraining, so how good the relaxed model is right after SVD initialization is unknown. At rank r, the SVD init exactly reproduces each original layer only if the residual has rank ≤ r. Average plus a full-rank residual means no compression, so quality right after initialization depends on how fast the residual's singular values decay.
- The "Average for relaxed, Stepwise for strict" finding matters for the user's design:
  - With per-pass relaxation (LoRA, or LN/FiLM per pass), averaging the tied layers is a better center.
  - With no relaxation, picking actual pretrained layers (Stepwise) is better than averaging them.

### Gaps
- I could not get the exact RRT Table/Figure 5 values (Stepwise vs Average vs Lower), the B=3 appendix numbers, KD-vs-no-KD deltas, or any "0 tokens of uptraining" row. The arXiv full text was unreachable. If a 0-token point exists, it would be in the initialization figures or the appendix.
- I found no RRT experiment on vision or multimodal models.

## 2. Mixture-of-Recursions (MoR; Bae, Kim, Bayat et al., KAIST/Mila/Google; arXiv 2507.10524, Jul 2025; NeurIPS 2025) and follow-ups

### Takeaway
MoR is trained from scratch, not converted. It establishes three things:
1. **Middle-Cycle sharing** is the best sharing pattern: first and last layers unique, middle layers cycled.
2. **Recursion-wise KV caching** cuts KV memory and IO to about (N_r+1)/(2N_r) of the baseline.
3. **Recursive KV sharing** (reuse the first recursion's KV) costs little with token-choice routing but a lot with expert-choice routing.

At an equal training-FLOP budget, a 167M MoR with 2 recursions beats a 315M vanilla Transformer (43.1% vs 42.3% few-shot).

### Cited Findings
- **Training regime.** "Our models are pretrained on a deduplicated subset of the FineWeb-Edu dataset" on a modified Llama architecture, so there is no pretrained-model conversion in MoR. — [MoR GitHub README (repo)](https://github.com/raymin0223/mixture_of_recursions)
- **Sharing strategies compared:** Cycle, Sequence, Middle-Cycle and Middle-Sequence, on 135M and 360M recursive models. The section heading names Middle-Cycle the most effective (search excerpt). Middle-Cycle keeps the first and last layers unique and shares the middle (explainer). — [arXiv HTML](https://arxiv.org/html/2507.10524v1); [Turing Post explainer](https://www.turingpost.com/p/mixtureofrecursions); [alphaXiv summary](https://www.alphaxiv.org/abs/2507.10524)
- **Routers:** expert-choice and token-choice; the implementations are `expert_choice_router.py` and `token_choice_router.py`. — [MoR README (repo)](https://github.com/raymin0223/mixture_of_recursions)
- **Recursion-wise KV caching:**
  - Each recursion step stores KV only for the tokens routed to it, and attention at that step is restricted to those cached tokens.
  - KV memory and IO drop to about (N_r+1)/(2N_r) of the baseline under the paper's capacity schedule, i.e. 0.75× for N_r=2 and about 0.67× for N_r=3 (search excerpt).
  - **Recursive KV sharing** instead caches only at the first recursion and reuses that cache later. It "leads to quite large performance degradation in expert-choice routing under a fixed token setting", but only a slight drop elsewhere. The stated reason is that shared-parameter depths show "highly consistent magnitude patterns and high cosine similarity" in KV (search excerpts; Table 4 uses a 360M model with 3 recursions).
  - Sources: [arXiv HTML](https://arxiv.org/html/2507.10524v1); [arXiv PDF](https://arxiv.org/pdf/2507.10524)
- **Quality at equal training FLOPs (16.5e18):**
  - A 167M MoR with 2 recursions (expert-choice) scores 43.1% few-shot average with lower validation loss, vs 42.3% for a 315M vanilla Transformer.
  - A 118M Middle-Cycle MoR with 3 recursions (expert-choice plus cache) reaches NLL 2.7925 and a 42.6% average at 30B tokens, and NLL 2.8466 / 41.5% at 20B tokens.
  - Sources (search excerpts from the table): [arXiv PDF](https://arxiv.org/pdf/2507.10524); [arXiv HTML](https://arxiv.org/html/2507.10524v1)
- **Throughput.** Up to 2.06× inference throughput (MoR-4) vs vanilla sequence-wise batching, and "up to 2× greater inference throughput … at similar accuracy". The KV-sharing throughput in the reference code uses a simplified shared-cache implementation. — [alphaXiv summary](https://www.alphaxiv.org/abs/2507.10524); [MoR README (repo)](https://github.com/raymin0223/mixture_of_recursions)
- **MoR README positioning:**
  - The earlier RRT required "two separate training processes for integrating parameter sharing and early-exiting [which] degraded performance, and this model still required handling the missing KV cache".
  - MoR unifies these with an end-to-end trained router.
  - Source: [MoR README (repo)](https://github.com/raymin0223/mixture_of_recursions)
- **Follow-ups (2025–2026; catalog TL;DRs unless noted):**
  - **T-LoopFormer** (arXiv 2609.15160, Sep 2026): token-level elastic depth with recursion-wise KV caches. — [arXiv](https://arxiv.org/abs/2609.15160)
  - **Soft MoR / SR-ViT** (arXiv 2607.00774, Jul 2026): token-wise mixtures over outputs of repeated applications of a shared ViT block. — [arXiv](https://arxiv.org/abs/2607.00774)
  - **MoR-MLLM** (arXiv 2610.08830, Oct 2026): MoR for multimodal LLMs. — [arXiv](https://arxiv.org/abs/2610.08830)
  - **"The Surprising Effectiveness of Shared Memory in Looped Transformers"** (arXiv 2610.02383): only the first recursion writes KV, and later recursions read it plus a short local window. At 150M–1B this improves quality, and a gated-DeltaNet hybrid with 5 recursions lowers FineWeb-Edu perplexity by 1.12–1.82 vs a same-size Transformer. — [arXiv](https://arxiv.org/abs/2610.02383)
  - **"Enabling Dynamic Computation in Looped LMs"** (arXiv 2610.09013): "Ragged KV" reuses each earlier token's exit-depth KV, cutting FLOPs and KV memory by up to 30% at full-depth quality on Ouro-style models. — [arXiv](https://arxiv.org/abs/2610.09013)
  - **Hybrid Latent Attention** (arXiv 2610.07940): uptrained on frozen Ouro-1.4B/2.6B (4 loops), it shrinks KV 10.7× per token and speeds decoding up to 7.4× at 16K context. — [arXiv](https://arxiv.org/abs/2610.07940)
  - **ResidualQuant** (arXiv 2610.10381): stores the final-loop KV as an anchor and the other loops as 2-bit residuals. On Ouro-1.4B and Huginn-3.5B it stays near BF16 accuracy with 80.7% less KV storage and up to 2.73× decode throughput. — [arXiv](https://arxiv.org/abs/2610.10381)
  - **FlashLoop** (arXiv 2609.29812): a training-free method using cross-loop redundancy (token-sparse updates, KV-residual quantization). It reports lossless accuracy with up to 1.64× speedup and 6× KV reduction on Ouro and Huginn. — [arXiv](https://arxiv.org/abs/2609.29812)
  - **Mixture-of-Recursions-for-Vision** community repo (image/multimodal generation). — [GitHub](https://github.com/enricguaschmesia-lab/Mixture-of-Recursions-for-Vision)

### Inferences
- For encoder-style or per-frame video use (no autoregressive KV), MoR's KV machinery matters less. Its transferable lessons are Middle-Cycle sharing and the high KV/activation similarity across shared depths. That similarity is also what ResidualQuant and FlashLoop exploit at inference.
- MoR's quality claims are from-scratch, iso-FLOP results. They do not show that converting a pretrained model into MoR is cheap.

### Gaps
- I could not retrieve MoR's per-strategy numbers (Cycle vs Sequence vs Middle-Cycle vs Middle-Sequence at 135M/360M), the exact KV-sharing deltas (Table 4), or any MoR uptraining-from-pretrained experiment. If one exists, it would be in the appendix.

## 3. Other parameter-sharing and conversion work (ALBERT → 2026 post-hoc looping)

### Takeaway
There is a clear divide between two kinds of work:
- **From-scratch weight sharing** (ALBERT, Universal Transformers, Subformer, Takase & Kiyono, Huginn, Ouro, Saunshi): looped models can match deeper or larger models at equal FLOPs, especially on reasoning.
- **Conversion of pretrained models** (RRT, MiniViT, Retrofitted Recurrence, Closing the Loop, LoopUS, CHERRY, DeltaLLM, RecursiveVLM): always needs gradient training. Budgets range from about 30–40M tokens for small deltas (DeltaLLM, 12–24% compression) to tens of billions of tokens for real looping (RRT 15–60B; Retrofitted Recurrence about 26–50B).

The only "zero-training" looping results use loops to add test-time compute on top of the full frozen model. They do not compress it.

### Cited Findings
**Classic from-scratch sharing**
- **ALBERT** (ICLR 2020), cross-layer sharing ablation on ALBERT-base:
  - E=768: not-shared 108M params / avg 82.3; shared-attention 83M / 81.6; shared-FFN 57M / 79.5; all-shared 31M / 79.8.
  - E=128: not-shared 89M / 81.6; shared-attention 64M / 81.7; shared-FFN 38M / 80.2; all-shared 12M / 80.1.
  - The paper says sharing attention costs nothing at E=128 (+0.1) and −0.7 at E=768; most of the loss comes from sharing the FFN.
  - Source (search excerpt of Table 4 plus a wiki table for the E=128 rows): [ALBERT arXiv](https://arxiv.org/abs/1909.11942); [aiwiki ALBERT](https://aiwiki.ai/wiki/albert)
- **Universal Transformers** (ICLR 2019): one shared layer applied recurrently with adaptive halting. It is the canonical "foundation" entry in the loop-model catalog. — [arXiv 1807.03819](https://arxiv.org/abs/1807.03819); [Awesome-Loop-Models](https://github.com/huskydoge/Awesome-Loop-Models)
- **Subformer** (Findings of EMNLP 2021): "sandwich-style" sharing, i.e. unshared first and last layers with shared middle (prelude-core-coda), plus factorized embeddings (catalog). — [arXiv 2101.00234](https://arxiv.org/abs/2101.00234)
- **Takase & Kiyono, "Lessons on Parameter Sharing across Layers in Transformers"** (arXiv 2104.06022, Apr 2021; SustaiNLP 2023):
  - Three strategies: Sequence, Cycle, and Cycle (rev). Cycle (rev) stacks the lower layers in reverse order for the remaining layers.
  - All three are efficient in parameters and compute, and they remain effective at WMT-scale data.
  - The official repo trains a 12-layer cycle_reverse model (`--share-type cycle_reverse`).
  - An aggregator lists 35.14 BLEU on WMT14 En-De for Cycle (Rev), which I have not verified.
  - Sources: [arXiv](https://arxiv.org/abs/2104.06022); [official repo](https://github.com/takase/share_layer_params); [hyper.ai (unverified figure)](https://hyper.ai/en/papers/2104.06022)
- **MiniViT** (Zhang et al., Microsoft, CVPR 2022), weight multiplexing:
  - Consecutive blocks share weights, and per-layer weight transformations keep the layers diverse. Per a secondary summary, these are linear transforms before and after the attention softmax and depth-wise convolution in the MLP.
  - Weight distillation (logits, self-attention and hidden states) transfers knowledge from a large pretrained teacher.
  - Results:
    - DeiT-B compressed 9.7× (86M → 9M) using one layer of parameters: Mini-DeiT-B gets 79.8% top-1 vs 81.8% for DeiT-B.
    - Mini-Swin-B with 12-layer parameters beats the 24-layer Swin-B, i.e. 48% smaller and +1.0% top-1.
    - Per the supplement, with sharing plus multiplexing, Swin-S and Swin-B lose only 1.2% and 0.3%.
    - The paper says naive sharing makes "training become unstable, and even causes training collapse as the number of shared layers increases".
  - Sources: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html); [CVPR supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_MiniViT_Compressing_Vision_CVPR_2022_supplemental.pdf); [arXiv 2204.07154](https://www.arxiv.org/pdf/2204.07154)

**Recurrent-depth / looped LLMs trained from scratch**
- **Huginn-0125** (Geiping et al., arXiv 2502.05171, Feb 2025; NeurIPS 2025):
  - 3.5B params trained on about 800B tokens on 4,096 AMD MI250X GPUs (Frontier).
  - Layout: 2 prelude, 4 recurrent-core and 2 coda blocks.
  - Training samples the recurrence count from a Poisson–lognormal distribution with mean 32, and backpropagates through only the last 8 passes.
  - The model card says fewer than 4 steps gives "very coarse answers", and benchmarks improve up to about 64 steps.
  - Sources: [arXiv](https://arxiv.org/abs/2502.05171); [HF model card](https://huggingface.co/tomg-group-umd/huginn-0125/raw/refs%2Fpr%2F6/README.md); [official repo (repo)](https://github.com/seal-rg/recurrent-pretraining)
- **Saunshi et al., "Reasoning with Latent Thoughts"** (arXiv 2502.17416; ICLR 2025):
  - A k-layer block looped L times "can be competitive to, if not better than" a kL-layer model on many downstream reasoning tasks, and T loops can simulate T steps of CoT.
  - A third-party critique says that on LM perplexity, looped models close only about 34–50% of the gap between the shallow and deep baselines. I did not verify this against the paper.
  - Sources: [arXiv](https://arxiv.org/abs/2502.17416); [critique, GitHub issue](https://github.com/microprediction/latentreasoning/issues/28)
- **Ouro / LoopLM** (Zhu et al., ByteDance; arXiv 2510.25741, Oct 2025):
  - 1.4B and 2.6B looped LMs pretrained on 7.7T tokens with an entropy-regularized learned-exit objective.
  - The 1.4B is comparable to 4B models on most benchmarks, and the 2.6B beats models up to 8B on reasoning-heavy tasks.
  - The gains are attributed to knowledge manipulation, not knowledge capacity.
  - Later papers describe Ouro as a 24-layer stack run for 4 loops (1.4B).
  - Sources: [arXiv](https://arxiv.org/abs/2510.25741); [project page](https://ouro-llm.github.io/); [SanSi (catalog)](https://arxiv.org/abs/2610.07730); [Hybrid Latent Attention (catalog)](https://arxiv.org/abs/2610.07940)
- **"Closing the Loop: Practical Training Recipes for Looped LMs"** (arXiv 2610.00673, Sep 30, 2026):
  - A from-scratch recipe that cuts Ouro's 7.7T tokens to 310B. The resulting 1.4B LoopLM beats a parameter-matched dense model on all 12 benchmarks (+14 GSM8K).
  - A minimal conversion recipe turns Qwen3-1.7B-Base into a looped model with "one learned input-mixing scalar" and a smoothed exit loss; step-specific LoRA is optional.
  - The converted model beats an identically continued dense baseline on every benchmark, with clear gains on GSM8K, MATH and MMLU-Pro.
  - Note that this conversion keeps all of Qwen's layers. Looping adds depth; it does not remove parameters.
  - Sources: [arXiv](https://arxiv.org/abs/2610.00673); [catalog briefing](https://github.com/huskydoge/Awesome-Loop-Models)
- **Parcae** (arXiv 2604.12946, ICLR 2026 workshop): stable looped LMs via spectral-norm constraints on the injection, plus iso-FLOP scaling laws. **Sparse Layers are Critical…** (arXiv 2605.09165): MoE layers make looped LMs scale better, because routing diverges across repeated shared layers (catalog). — [Parcae](https://arxiv.org/abs/2604.12946); [Sparse layers](https://arxiv.org/abs/2605.09165)

**Converting pretrained LLMs into looped/recurrent models**
- **Retrofitted Recurrence** (McLeish, …, Geiping, Goldstein, Goldblum; arXiv 2511.07384, Nov 10, 2025):
  - **Models.** TinyLlama-1.1B-3T, Llama-3.2-1B and OLMo-2-1B are converted to Huginn-style prelude/core/coda models. A curriculum ramps the mean recurrence up during training. At matched compute, the converted models beat continued post-training of the original on math. — [arXiv](https://arxiv.org/abs/2511.07384)
  - **Conversion recipe (repo).** For TinyLlama (22 layers): prelude = layers [0–3], core = layers [10–17] (8 layers), coda = [18–21]. Layers 4–9 are dropped.
  - **Adapter (repo).** A new Linear(2d→d) concatenates the recurrent state with the prelude output ("input injection").
  - **Conversion check (repo).** The scripts verify hidden states match exactly at 1 recurrence before training ("you should see a lot of `True, 0.000`").
  - Repo sources: [convert_llama.py](https://github.com/mcleish7/retrofitting-recurrence/blob/main/convert_pretrained_model/convert_llama.py); [README](https://github.com/mcleish7/retrofitting-recurrence)
  - **Training scripts (repo).** 25k–50k optimizer steps on 1,024-token sequences; learning rate 5e-5 (Muon 1e-3 in some runs); `mean_recurrence_schedule` warmup 0.25–0.75 with "1-sqrt" ramps. — [shells/](https://github.com/mcleish7/retrofitting-recurrence/tree/main/shells)
  - **Training tokens (my computation from the authors' released data).** The logged FLOPs for the non-recurrent baselines equal 6·N·(steps × 2^20 tokens). So 25k steps ≈ 26B tokens (TinyLlama general-data runs) and 48k steps ≈ 50B tokens (Llama math runs). — [paper_plots/data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots)
  - **Quality vs recurrence, TinyLlama, about 26B tokens of general data (repo data, `loop_whole_model.jsonl`, checkpoint 25k).** Benchmarks are HellaSwag / LAMBADA / ARC-E.

    | Model (unique layers) | Test recurrences r | Layer passes | HellaSwag | LAMBADA | ARC-E |
    |---|---|---|---|---|---|
    | 4-8-4 retrofit (16) | 1 | 16 | 41.9 | 29.6 | 49.2 |
    | 4-8-4 retrofit (16) | 2 | 24 | 52.5 | 44.0 | 54.8 |
    | 4-8-4 retrofit (16) | 4 | 40 | 60.5 | 54.6 | 60.2 |
    | 4-8-4 retrofit (16) | 32 | 264 | 62.6 | 58.2 | 63.1 |
    | Static TinyLlama, same continued training (22) | – | 22 | 61.4 | 58.1 | 64.5 |
    | 2-4-2 retrofit (8) | 32 | 132 | 42.7 | 35.9 | 50.7 |
    | 6-10-6 retrofit (22, all layers) | 32 | 332 | 65.0 | 60.4 | 66.2 |

    The 4-8-4 model was trained with mean recurrence 32, so the r=1/2/4 rows are off its training distribution. — [paper_plots/data/loop_whole_model.jsonl](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)
  - **Math post-training (repo data, about 50B tokens):**
    - Llama-3.2-1B 4-6-4 (14 of 16 layers unique), trained at r=4: GSM8K-CoT 39.4 at r=4, 36.2 at r=2 (20 passes) and 26.4 at r=1, vs 37.1 for the static model trained the same way.
    - OLMo-2-1B 4-6-4, trained at r=4: 38.7 at r=4, 37.4 at r=2 and 24.3 at r=1, vs 35.3 static.
    - TinyLlama 4-8-4, trained at r=32: GSM8K 45.0 at r=32 vs 26.6 static.
    - Source: [paper_plots/data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)
  - **Pretrained vs random init (repo data, 2-4-2 Llama-3.2-1B at r=32, about 15B tokens).** Llama-initialized vs random-initialized: HellaSwag 55.6 vs 48.3, LAMBADA 45.5 vs 38.5, ARC-E 60.4 vs 56.1. — [tinyllama_long_run_table.jsonl](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)
- **LoopUS** (arXiv 2605.11011, May 2026):
  - Recasts a pretrained LLM into encoder / looped reasoning block / decoder, with block boundaries chosen from "staged representation dynamics".
  - Uses a Mamba-style input-dependent exponential-decay gate across depth, random deep supervision, and a confidence-head early exit.
  - I found no token budgets or numbers.
  - Sources: [arXiv HTML](https://arxiv.org/html/2605.11011v1); [Pith review](https://pith.science/paper/2605.11011)
- **"Retrofitting Recurrent Depth into a Pretrained LM"** (arXiv 2608.11233, Jul 2026):
  - Qwen2.5-0.5B-Instruct is split into a 6-layer prelude, a 12-layer weight-tied recurrent block and a 6-layer coda, with a learned "re-entry bridge". One loop reproduces the original dense path.
  - Two installation budgets are compared: full-block adaptation, and a frozen base with rank-16 LoRA.
  - Source (catalog briefing): [arXiv](https://arxiv.org/abs/2608.11233)
- **CHERRY** (Kwon & Park; arXiv 2606.31796, Jun 2026):
  - Compresses 48 layers to 6 unique blocks by averaging each group of k adjacent layers into one, then restores depth with prelude/recurrent core/coda unrolling.
  - A 227M compressed model reaches loss 2.934 vs 2.926 for a 566M dense model.
  - The authors note the averaged model "starts with high loss before training recovers it". The numbers are self-reported, and the frontier-scale claims are projections.
  - Sources (search excerpt): [arXiv](https://arxiv.org/abs/2606.31796); [alphaXiv](https://www.alphaxiv.org/abs/2606.31796)
- **DeltaLLM** (Mikaelyan et al.; arXiv 2501.18596, Jan 2025):
  - Shares weights between blocks plus low-rank delta matrices, trained by progressive module replacement on about 30–40M tokens (37M in the intro), with base weights frozen.
  - DeltaLLaMA/DeltaPhi: 12% fewer parameters while keeping 90% of base performance.
  - DeltaPhi 2.9B (24% reduction) matches a recovery-fine-tuned SlicedPhi 3.3B (12% reduction).
  - Sources (search excerpt): [arXiv](https://arxiv.org/abs/2501.18596); [arXiv HTML](https://arxiv.org/html/2501.18596v1)
- **RecursiveVLM, "Looping Back to Move Forward"** (Xu et al., PKU; arXiv 2602.09080, Feb 2026):
  - Naive recursion fails in VLMs because hidden-state scale grows with depth.
  - A "Recursive Connector" (RMSNorm plus a mapping back toward input-embedding space) and a "Monotonic Recursion Loss" (supervise every recursion) give about +3% over standard Transformers and +7% over vanilla recursive baselines.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.09080v1)
- **Vision looped models (all from scratch unless noted):**
  - **LoopViT** (arXiv 2602.02156): an 18M model reaches 65.8% on ARC-AGI-1.
  - **RD-ViT** (arXiv 2605.03999): recurrent-depth ViT for segmentation with depth-wise LoRA.
  - **ELT** (arXiv 2604.09168): elastic looped transformers for image and video generation, with intra-loop self-distillation for any-time inference.
  - **MergeOver** (arXiv 2608.13141): post-training token merging on the weight-shared Sliced Recursive Transformer, with no retraining. It cuts about 37% of peak activation memory at a 1.47-point ImageNet top-1 cost and reduces latency on a Raspberry Pi 5.
  - **WeCoLoRA** (arXiv 2404.09326, WACV 2025): copies every other layer of a pretrained ViT into a shallower student and recovers it with LoRA plus few-shot distillation. This is layer skipping, not weight sharing.
  - Sources (catalog TL;DRs): [LoopViT](https://arxiv.org/abs/2602.02156); [RD-ViT](https://arxiv.org/abs/2605.03999); [ELT](https://arxiv.org/abs/2604.09168); [MergeOver](https://arxiv.org/abs/2608.13141); [WeCoLoRA](https://arxiv.org/abs/2404.09326)

**Zero-training looping of frozen models (adds compute; does not compress)**
- **"Transformer Layers as Painters"** (Sun et al., AAAI 2025):
  - On frozen models, replacing a block of middle layers with copies of the middle-most layer ("Middle Repeat") is the worst variant, "quickly degrading to random baseline".
  - Running the middle layers in parallel and looping them degrades gracefully, and the best loop count is roughly proportional to the number of parallelized layers.
  - First and last layers differ from the middle layers, which are fairly uniform.
  - Source: [arXiv](https://arxiv.org/abs/2407.09298)
- **"Training-Free Looped Transformers"** (Chen, Li, Liang, Lao, Liu; arXiv 2605.23872, May 2026):
  - A wrapper re-applies a contiguous mid-stack block of a frozen checkpoint (Qwen3, Llama-3.2, DeepSeek-V2-Lite, Moonlight, Qwen1.5-MoE) as damped Euler sub-steps, at a cost of (b−a+1)(K−1) extra layer passes.
  - "Naive block reapplication usually degrades performance."
  - Reported gains are about +1–2.6 pp on MMLU-Pro, CommonsenseQA and OpenBookQA (blog summary, unverified).
  - Sources: [arXiv HTML](https://arxiv.org/html/2605.23872v1); [blog summary](https://alanhou.org/blog/arxiv-training-free-looped-transformers/)
- **CoLa, "Skip a Layer or Loop It?"** (arXiv 2507.07996, Jul 2025): per-sample MCTS over skip/repeat programs of frozen pretrained layers, with no training. A June 2026 follow-up (arXiv 2606.06574) learns a predictor for such programs. — [HF paper page](https://huggingface.co/papers/2507.07996); [follow-up](https://arxiv.org/abs/2606.06574)

### Inferences
- Every result that reduces unique parameters by looping a pretrained model includes gradient training. The only exception is DeltaLLM-scale tiny training (tens of millions of tokens), and that achieves only 12–24% compression, not 3×.
- Training-free looping papers keep the full model and add passes. They show that pretrained mid-stack blocks tolerate damped re-application, not that one block can stand in for others.
- **The TinyLlama retrofit data is the most quantitative public evidence on "loop factor vs quality" after conversion:**
  - At about equal compute, a 16-unique-layer model run as 24 passes (4-8-4 at r=2) after about 26B tokens trails the 22-layer model by about 9–14 points on HellaSwag/LAMBADA. That model was trained for r≈32, so a model trained at r=2 would likely do better.
  - Matching the original needs r≥4, i.e. about 2× the compute.
  - Cutting to 8 unique layers (2-4-2, about 36% of the layers) cannot be rescued even by 132 passes (HellaSwag 42.7 vs 61.4).
  - This suggests that for language modeling, unique-parameter count (knowledge capacity) is the binding constraint. That is consistent with Ouro's "knowledge manipulation, not capacity" finding and with "What Makes Recurrence Effective" (extra loops help reasoning but degrade knowledge).

### Gaps
- I found no Ouro conversion-from-pretrained result. Ouro is from scratch.
- I found no LoopUS or Closing-the-Loop numeric tables or token budgets (abstracts only).
- I found no Subformer numbers, no exact Takase BLEU table, and no MiniViT sharing-only ablation numbers.

## 4. Depth conditioning in looped models (FiLM, per-iteration embeddings, step encodings), and what "LARM" is

### Takeaway
"LARM" is the **Loop Audio Recurrent Model** from Idiap: Kaloga, Kumar, Sheikh, Khalil, Motlicek and Kodrasi, "Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers", arXiv 2606.04678, June 2026.
- It applies one shared Transformer encoder K times. Its FiLM layer acts on the recurrent state between loops: h^(k) = γ(d̄(k)) ⊙ â^(k) + β(d̄(k)), with normalized depth d̄(k) = (k−1)/(K−1), plus a supervision-clock embedding.
- It is trained from scratch (LibriSpeech), not converted.

Other per-pass specialization mechanisms in the literature: Universal Transformer timestep embeddings; MiniViT per-layer weight transforms; RRT per-layer LoRA; RD-ViT shared LoRA with a per-loop scale vector; LoopFormer time/step-size conditioning; RecursiveVLM's RMSNorm connector; and a learned input-injection adapter or scalar (Huginn / Retrofitted Recurrence / Closing the Loop).

### Cited Findings
- **LARM recurrence (repo README):**
  - h^(0) = φ(x) (acoustic frontend); z^(k) = F_θ(h^(k−1)) with a shared encoder and a shared CTC head ψ.
  - Delayed soft-posterior feedback: r̄_t^(k) = p_{t−1}^(k) W_ρ.
  - State aggregation: a^(k) = z^(k) + β h^(0) + α r̄^(k), with learnable α and β.
  - Clock: â^(k) = a^(k) + W_c[(k−1) mod c].
  - FiLM: h^(k) = γ_film(d̄(k)) ⊙ â^(k) + β_film(d̄(k)), where d̄(k) = (k−1)/(K−1).
  - CTC loss only at checkpoint loops {c, 2c, …, K}.
  - The flag `--depth_mode film` selects FiLM, and there is a `scripts/ablative_depth/` set of depth-conditioning ablations.
  - Source: [idiap/LARM README (repo)](https://github.com/idiap/LARM)
- **LARM results (repo README), LibriSpeech greedy test-clean / test-other WER:**
  - 100h: LARM d=384, K=12, 7.7M params: 11.34 / 31.84, vs 14.43 / 37.23 for a 16-block unshared encoder (28.9M) and 12.24 / 34.06 for a 48-block encoder (85.7M).
  - 960h: LARM 7.7M: 4.59 / 11.75 vs 16-block 4.79 / 13.26.
  - 960h: LARM d=768 (28.9M): 3.45 / 9.44, beating the 48-block 85.7M encoder (3.87 / 10.58).
  - The README says the reference LARM "matches or beats a 16-block unshared encoder at ~4× fewer parameters".
  - Source: [idiap/LARM README](https://github.com/idiap/LARM)
- **LARM ablation:** the full model lowers 100h test-clean WER from 12.70 (naive looping) to 11.34 (search-excerpt summary of the arXiv paper). Intermediate (unsupervised) loops can be non-monotonic, while supervised checkpoints improve monotonically (README). — [arXiv HTML 2606.04678v2](https://arxiv.org/html/2606.04678v2); [alphaXiv overview](https://www.alphaxiv.org/overview/2606.04678); [README](https://github.com/idiap/LARM)
- **LoopFormer** (arXiv 2602.11451; ICLR 2026): an elastic-depth looped Transformer that conditions each iteration on normalized time and step size. It uses a shortcut-consistency objective for budget-conditioned reasoning "without retraining". — [arXiv HTML](https://arxiv.org/html/2602.11451v1); [GitHub](https://github.com/armenjeddi/loopformer)
- **"What Makes Recurrence Effective in Looped LMs?"** (arXiv 2609.36636, Sep 2026): "channel-wise history-state injection with timestep conditioning" is more robust across inference budgets than initial-state injection. Non-recurrent output layers make models more robust to running too few loops, and extra loops help reasoning but hurt knowledge (catalog briefing). — [arXiv](https://arxiv.org/abs/2609.36636)
- **RD-ViT** (arXiv 2605.03999): depth-wise LoRA built from shared down/up projections plus a per-loop scale vector, giving each loop its own behavior for about 1.5K extra params in the Tiny model (search excerpt). — [arXiv PDF](https://arxiv.org/pdf/2605.03999)
- **RecursiveVLM** (arXiv 2602.09080): an RMSNorm-based "Recursive Connector" fixes the hidden-state scale growth that breaks naive recursion. — [arXiv HTML](https://arxiv.org/html/2602.09080v1)
- **Input injection in conversions:**
  - Retrofitted Recurrence adds Linear(2d→d) over [state; embeddings]. — [repo](https://github.com/mcleish7/retrofitting-recurrence)
  - Closing the Loop uses a single learned input-mixing scalar. — [arXiv](https://arxiv.org/abs/2610.00673)
  - The Qwen2.5 retrofit uses a "re-entry bridge". — [arXiv](https://arxiv.org/abs/2608.11233)
- **Loop Dropout** (arXiv 2609.34218): LoRA adapters shared across loops end up biased toward late loop positions. Randomly masking per-loop adapter applications with inverse-survival rescaling helps (catalog). — [arXiv](https://arxiv.org/abs/2609.34218)
- **LoopMoEVR** (arXiv 2610.05109): video restoration with one parameter-shared loop, "degradation-conditioned low-rank loop embeddings", and about 0.884M params (catalog). — [arXiv](https://arxiv.org/abs/2610.05109)

### Inferences
- The user's per-pass LN scale/shift is functionally close to LARM's FiLM. LARM conditions the inter-loop state (after the block), whereas per-pass LN conditions the normalized input inside the block. RD-ViT's "shared LoRA + per-loop scale vector" sits between FiLM and full per-pass LoRA in parameter cost.
- In LARM, FiLM is a smooth function of normalized depth, so it can interpolate to unseen K. Per-pass lookup tables (per-pass LN, per-pass LoRA) cannot extrapolate to more loops than were trained.

### Gaps
- I could not find LARM's per-component ablation table (FiLM alone vs clock alone vs feedback alone) beyond the 12.70 → 11.34 full-vs-naive number.
- Whether the user's "LARM's FiLM" refers to this ASR paper is my best inference. I found no other looped model called LARM; "LARM" also names unrelated Large Reconstruction/robotics models, per [emergentmind topic page](https://www.emergentmind.com/topics/larm).

## 5. Which layers can be shared, and the quality cost per loop factor (with/without relaxation, with/without distillation)

### Takeaway
The evidence consistently says to keep the first and last layers unique and loop the middle:
- Stepwise keeping first and last; MoR Middle-Cycle; the Subformer sandwich.
- Huginn 2/4/2; retrofits 4/8/4, 4/6/4 and 6/12/6.
- "Painters": middle layers are near-uniform.

In the Llama-1B retrofit, the best core was the later middle layers just before the coda. In ALBERT, attention is cheap to share and FFN is expensive. Quality per loop factor depends mostly on how much gradient recovery you do:
- 2× loop: with relaxation, KD and 60B tokens it reaches near parity (RRT).
- 3–4× in vision: about −0.3 to −1.2% with weight transforms plus distillation (MiniViT Swin).
- 12× in ViT: about −2% top-1 with weight transforms plus distillation (MiniViT DeiT-B).
- 8× in LMs: near parity only after substantial training, and only per one self-reported preprint (CHERRY).

### Cited Findings
- **Positional evidence:**
  - RRT Stepwise keeps the first and last layers. — [arXiv HTML](https://arxiv.org/html/2410.20672v1)
  - MoR: Middle-Cycle (unique first and last) is the best strategy. — [arXiv HTML](https://arxiv.org/html/2507.10524v1)
  - Subformer uses sandwich sharing. — [arXiv](https://arxiv.org/abs/2101.00234)
  - "Painters": middle layers are surprisingly uniform, while first and last differ. — [arXiv](https://arxiv.org/abs/2407.09298)
- **Which middle layers to loop, Llama-3.2-1B 2-4-2 (repo data, final training loss).**
  - Core layer choice (prelude [0,1], coda [14,15] throughout):
    - [10,11,12,13]: 2.91.
    - Evenly spaced [4,7,10,13]: 2.97.
    - Contiguous early-middle [2,3,4,5]: 3.15–3.18.
    - Evenly spaced [2,5,8,11]: 3.84.
    - From-scratch 2-4-2 (labelled "Takase" init): 3.67.
  - For the TinyLlama 4-8-4 core, the default "last 8 layers before the coda" reached loss 0.729, vs 0.772 for a ShortGPT-importance-selected core.
  - Source: [paper_plots/which_layers_llama_1b.csv, shortpgt.csv](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots)
- **Which sub-modules to share (ALBERT, from scratch):**
  - Sharing only attention: −0.7 (E=768) / +0.1 (E=128).
  - Sharing only the FFN: −2.8 / −1.4.
  - Sharing all: −2.5 / −1.5.
  - Source: [ALBERT](https://arxiv.org/abs/1909.11942); [aiwiki table](https://aiwiki.ai/wiki/albert)
- **Non-recurrent output layers** make looped LMs more robust to under-looping. — [arXiv 2609.36636](https://arxiv.org/abs/2609.36636)
- **Data points by loop factor (unique-parameter reduction):**
  - **2×, pretrained LLM → recursive, uptrained:**
    - Recursive Gemma 1B (2 × 9-layer block) recovers "most" of Gemma 2B. Relaxed with rank 512 and 60B tokens (with KD): 58.4 vs 58.6. — [arXiv](https://arxiv.org/abs/2410.20672)
  - **About 1.4× in layers, LM retrofit, about 26B tokens:**
    - TinyLlama 22 → 16 unique layers (4-8-4) matches the static model only when run at r≥4–32 (40–264 layer passes). At r=2 (24 passes), HellaSwag is 52.5 vs 61.4 (trained for r≈32).
    - 22 → 8 unique layers (2-4-2): HellaSwag 42.7 even at 132 passes. — [repo data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)
  - **About 1.14× in layers (Llama/OLMo 16 → 14 unique, 4-6-4, trained at r=4, about 50B math tokens):**
    - At r=2 (20 passes vs 16), GSM8K is 36.2 vs 37.1 (Llama) and 37.4 vs 35.3 (OLMo). — [repo data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)
  - **2× from scratch, at equal FLOPs:** MoR 167M with 2 recursions beats a 315M vanilla model (43.1 vs 42.3). — [arXiv PDF](https://arxiv.org/pdf/2507.10524)
  - **Up to 12× from scratch (ALBERT all-shared):** −1.5 to −2.5 GLUE-avg points. — [ALBERT](https://arxiv.org/abs/1909.11942)
  - **2× ViT, pretrained teacher, weight transforms + distillation:** Mini-Swin-B with 12-layer parameters beats 24-layer Swin-B (+1.0% top-1 at −48% size). — [MiniViT](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html)
  - **12× ViT (DeiT-B, one shared layer) + transforms + distillation:** 81.8 → 79.8 top-1. — [MiniViT arXiv](https://www.arxiv.org/pdf/2204.07154)
  - **8× LM (48 → 6 unique blocks), layer averaging + recurrent unrolling + training:** loss 2.934 (227M) vs 2.926 (566M dense). This is self-reported, and the training-token budget was not retrieved. — [CHERRY](https://arxiv.org/abs/2606.31796)
  - **About 4× audio, from scratch with FiLM:** 7.7M LARM ≈ or better than a 28.9M 16-block encoder. — [LARM](https://github.com/idiap/LARM)
- **Relaxation vs no relaxation:**
  - RRT: relaxed (LoRA, Average init) beats strict (Stepwise), and rank 512 approaches the original. — [arXiv HTML](https://arxiv.org/html/2410.20672v1)
  - MiniViT: transforms are needed because pure sharing "drops significantly" and can collapse in training. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.pdf)
- **Distillation vs none:**
  - RRT's best Gemma numbers use KD during 60B-token uptraining. — [arXiv PDF](https://arxiv.org/pdf/2410.20672)
  - MiniViT's weight distillation (attention, hidden states and logits) is part of the method, and the paper reports no sharing-only number. — [MiniViT](https://www.arxiv.org/pdf/2204.07154)
  - I found no clean "same budget, KD vs no-KD" table for looped conversion.

### Inferences
- For the user's 2-unique-block × 3-pass student (unique-parameter reduction 3× relative to a 6-block teacher):
  - The closest LM analogue (RRT, B=2) needed 15–60B tokens. The closest vision analogue (MiniViT) used full ImageNet-scale training with distillation.
  - A 3× loop with only per-pass LN/FiLM (no LoRA) is the "strict" regime, where both RRT and MiniViT report larger losses than relaxed variants.
- In the retrofit data, more loops recover reasoning and "manipulation" but not lost capacity. For video and image encoders, where per-frame features may be more "procedural" than knowledge-heavy, the trade may be better, but I found no conversion study that measures this for a pretrained ViT or video model.

### Gaps
- No source gives a systematic 2×/3×/4× sweep after conversion with the same budget. RRT's B=3 appendix is the closest candidate, but I could not retrieve it.
- I found no data on Jetson-class latency for converted looped models. MergeOver reports a Raspberry Pi 5 latency reduction for a recursive ViT, without precise numbers retrieved.

## 6. Is there any looped model obtained with ONLY calibration data / closed-form fitting (no gradient training) that retains usable quality?

### Takeaway
No. I found no published result where a pretrained Transformer is converted into a looped / weight-tied model with fewer unique weights using only calibration data or closed-form fitting, and still retains usable quality. The evidence that does exist points to failure:
- Middle-layer repetition on frozen models drops to random-baseline performance.
- Freshly converted recurrent models start with training loss near or above uniform-random.
- MiniViT reports training collapse with naive sharing.
- CHERRY's averaged model "starts with high loss".

The closest training-free results either (a) keep all weights and add loops, or (b) share SVD bases across layers without looping (Basis Sharing: 20–50% compression, training-free, with mixed independent replications).

### Cited Findings
- **Frozen middle-layer repetition fails.** Replacing middle layers with copies of the middle-most layer, with no training, is the worst variant and "quickly degrad[es] to random baseline". — [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298)
- **Naive re-application of pretrained blocks "usually degrades performance".** Training-free gains require damped sub-steps, and the model keeps all of its weights. — [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1)
- **Freshly converted recurrent models start near or above uniform-random loss (repo data, first logged training step):**
  - Retrofitted Llama-3.2-1B 2-4-2 runs start at 12.4–14.3 nats (ln(128k vocab) ≈ 11.8) and fall to about 2.9–3.8 after 11k–19k steps, depending on which layers form the core.
  - TinyLlama 4-8-4 starts at 9.7–12.0 nats (ln(32k) ≈ 10.4).
  - Simple ShortGPT layer removal (no looping) of 6 TinyLlama layers starts at 2.8–4.9 nats.
  - Caveat: the first step runs at the sampled training recurrence (mean up to 32), not at r=1. At r=1 the conversion is exact by construction.
  - Source: [paper_plots CSVs](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots)
- **Layer averaging needs training.** CHERRY's average-k-adjacent-layers model "starts with high loss before training recovers it". — [CHERRY](https://arxiv.org/abs/2606.31796)
- **Naive sharing collapses in ViTs.** Weight sharing across layers "causes training collapse as the number of shared layers increases". — [MiniViT](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.pdf)
- **Basis Sharing** (arXiv 2410.03765; ICLR 2025):
  - Method: concatenate a weight type across n layers, run SVD once, and represent each layer as shared basis vectors times layer-specific coefficients. Which matrix types and layer groups to share is chosen by compression error.
  - Results: training-free at 20–50% compression, up to 25% lower perplexity and up to 4% higher reasoning accuracy than SVD baselines.
  - Independent 2026 evaluations surfaced by search report collapse in some settings: Pythia-1.4B at PPL 758 and HellaSwag 26.8%; Llama-3.1-8B at 60% keep with C4 PPL 461. I could not confirm which 2026 papers report which figure. Candidates are [Cross-Layer Subspace Coupling](https://arxiv.org/pdf/2605.30836) and [LowRankArena](https://arxiv.org/pdf/2608.26389).
  - This is cross-layer *basis* sharing with per-layer coefficients, not a looped architecture, and it does not reduce compute.
  - Sources: [arXiv](https://arxiv.org/abs/2410.03765); [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)
- **FFN-sublayer merging** (Verma et al.; arXiv 2501.06126): post-training align-average-tie of more than a third of FFN sublayers keeps performance comparable and beats layer pruning. A third-party review says retention relies on recovery fine-tuning, and I could not confirm no-fine-tuning numbers. — [arXiv](https://arxiv.org/abs/2501.06126); [Pith review](https://pith.science/paper/2501.06126)
- **DeltaLLM** is the minimal-training end: about 30–40M tokens to train low-rank deltas between shared blocks, for 12–24% parameter reduction. — [arXiv](https://arxiv.org/abs/2501.18596)
- **PTQ for looped models must calibrate across loops:**
  - "Calibration blindness": GPTQ Hessians from step 0 only underperform Hessians accumulated across recurrence steps, which recover bf16-level accuracy on Huginn. — [arXiv 2609.30820](https://arxiv.org/abs/2609.30820)
  - Loopy picks one shared rotation and scaling by the final loss at the deployment depth. — [arXiv 2610.05265](https://arxiv.org/abs/2610.05265)
  - LoopQ corrects "role shifts, cross-loop state mismatch, and recursive quantization-error accumulation". — [arXiv 2605.16343](https://arxiv.org/abs/2605.16343)
  - Error accumulates only for loops that "never settle"; for settling loops, a fixed weight error just shifts the fixed point. — [arXiv 2609.39277](https://arxiv.org/abs/2609.39277)
  - Weight-tied recursive reasoners: per-tensor 4-bit activations take Sudoku from 84.1% to 0%, while MXInt4 block scaling restores it. — [arXiv 2607.16237](https://arxiv.org/abs/2607.16237)

### Inferences
- A near-post-training looping pipeline would combine closed-form initialization with very light training:
  - Closed-form initialization: Average or Stepwise for the shared block, plus SVD of the residuals into per-pass LoRA (RRT), or SVD bases with per-layer coefficients (Basis Sharing).
  - Very light training: DeltaLLM-scale (about 10⁷–10⁸ tokens), feature-matching distillation against the teacher's internal outputs, which is the user's stage A.
  - There is no published evidence for this combination at a 3× loop factor. DeltaLLM's 30–40M tokens reached only 12–24% compression. At 2× and above, every published success used 10¹⁰-token-scale uptraining or full training.
- Any calibration-based fitting of the shared block (least squares, GPTQ-style) should collect statistics from every pass of the loop, not from the teacher's per-layer inputs alone. The looped-PTQ papers show that one-step calibration misses the error feedback through the loop.
- Pretrained middle blocks tolerate re-application only when it is damped. So FiLM or per-pass scale that can shrink each pass's residual update (close to zero at initialization) is a plausible way to make a closed-form starting point stable before any training. This is an inference from the damped sub-step results, not a tested method.

### Gaps
- I found no paper reporting the accuracy of an RRT-style SVD-initialized relaxed recursive model before any uptraining. That single number would directly answer the zero-training question, and it is missing from everything I could access.
- I found no closed-form or calibration-only looped conversion for ViTs or video models.
- I could not identify which specific 2026 papers report the Basis Sharing collapse figures.
