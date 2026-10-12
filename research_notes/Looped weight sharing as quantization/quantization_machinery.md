# Quantization machinery for framing cross-layer weight sharing / looping as a post-training quantization method

Scope: the quantization, codebook, delta-compression and low-rank+quantization literature that supplies the machinery for "shared base + small per-layer low-bit residual, fitted from calibration data". Research date 2026-10-12. arXiv dates are given as YYYY-MM from the arXiv ID.

Access note: arxiv.org, openreview.net, aclanthology.org, proceedings.neurips.cc and pmc.ncbi.nlm.nih.gov could not be reached from this sandbox (proxy 403 / DNS failure). Numbers below come from search-engine extracts of those papers, from GitHub READMEs that were fetched directly, and in a few marked places from my recall of well-known papers ("[recalled, not re-fetched]"). Where extracts disagreed, both values are given.

Notation used throughout: layer weight W (m x n), calibration inputs X, layer Hessian H = X X^T, bits per weight = bpw. "ctx" = evaluation context length; Llama-2 Wikitext2 FP16 perplexity is 5.47 at ctx 2048 and 5.12 at ctx 4096, so numbers from papers that use different ctx are not comparable.

## 1. Closed-form layer-wise PTQ, and how the objective generalizes when one tensor must serve several layers

### Takeaway
All of GPTQ, OBQ, AWQ, QuIP#, QTIP, CALDERA and QERA minimize the per-layer proxy ||WX - ŴX||² = tr((W-Ŵ) H (W-Ŵ)^T). That objective summed over the layers that share one tensor collapses exactly into a single standard GPTQ problem with the summed Hessian H_Σ = Σ H_i and a Hessian-weighted mean target W* = (Σ W_i H_i) H_Σ^-1. Existing solvers can therefore be reused unchanged. The only published GPTQ-style solve for a weight used at several depths is in looped models: "Quantizing Looped Transformers" (2026-09) accumulates the GPTQ Hessian across recurrence steps and beats both one-step GPTQ and RTN on all 9 checkpoints. I found no paper that does a joint multi-layer GPTQ solve for distinct original layers W_i mapped onto one shared tensor.

### Cited Findings
- **GPTQ (2022-10, ICLR 2023):** a one-shot method that uses approximate second-order information. It quantizes 175B-parameter GPT models in about 4 GPU hours down to 3 or 4 bpw with negligible accuracy loss. It is built on Optimal Brain Quantization, quantizing column by column and updating the not-yet-quantized columns with the inverse Hessian [recalled, not re-fetched] — [GPTQ arXiv](https://arxiv.org/abs/2210.17323)
- **The Hessian is per layer and shared across rows:** GPTQ computes H = X X^T from that layer's calibration inputs and processes the model layer by layer. Because one Hessian serves every output row, all weights in a column are quantized in parallel — [ZeroEntropy GPTQ explainer](https://zeroentropy.dev/concepts/gptq/); [EmergentMind GPTQ](https://www.emergentmind.com/topics/generative-pretrained-transformer-quantization-gptq)
- **GuidedQuant** keeps a separate Hessian for each group of output rows, weighted by saliency. This is a published precedent for weighting several Hessians in one solve. It comes from a search-engine summary; the GuidedQuant paper itself was not read — [search extract](https://www.emergentmind.com/topics/generative-pretrained-transformer-quantization-gptq)
- **OBQ/OBC (2022-08, NeurIPS 2022):** the exact greedy Optimal Brain Quantizer that GPTQ approximates. It treats layer-wise compression as ||WX - ŴX||² and applies OBS-style updates [recalled, not re-fetched] — [OBC arXiv](https://arxiv.org/abs/2208.11580)
- **AWQ (2023-06, MLSys 2024):** protecting about 1% of salient weight channels, chosen by activation magnitude, through per-channel scaling greatly reduces quantization error. It needs no backpropagation or reconstruction [recalled, not re-fetched] — [AWQ arXiv](https://arxiv.org/abs/2306.00978)
- **SmoothQuant (2022-11, ICML 2023):** an equivalent transformation that moves activation outliers into the weights, for W8A8 [recalled, not re-fetched] — [SmoothQuant arXiv](https://arxiv.org/abs/2211.10438)
- **OmniQuant (2023-08, ICLR 2024):** learnable weight clipping plus learnable equivalent transformations, fitted by block-wise output reconstruction [recalled, not re-fetched] — [OmniQuant arXiv](https://arxiv.org/html/2308.13137v3)
- **QuaRot (2024-03):** Hadamard rotations remove outliers and enable end-to-end 4-bit inference. Its FP16 Llama-2-7B Wikitext2 baseline is 5.47 and plain 4-bit RTN gives 8.37 — [QuaRot arXiv](https://arxiv.org/pdf/2404.00456). The QServe paper lists QuaRot W4A4 on Llama-2-7B at 6.10 — [QServe](https://arxiv.org/html/2405.04532v2)
- **SpinQuant (ICLR 2025):** learned rotations. W4A4KV4 Llama-2-7B is 2.9 points below full precision on zero-shot reasoning (average 64.0) — [SpinQuant ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf). A summary says it closes the gap to full precision by 30.2% (Llama-2-7B) and 34.1% (Llama-3-8B) relative to QuaRot — [HF papers listing](https://huggingface.co/papers?q=LLM+quantization)
- **Hessian accumulated across recurrence (most relevant precedent):** "Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness" (Nux Li, 2026-09, preprint, 27 pp.) studies Huginn-3.5B and 7 looped architectures, 9 checkpoints in all.
  - *Calibration blindness:* one-step GPTQ builds its Hessian from step-0 activations only. Input directions used later in the recurrence get little weight, and one-step GPTQ is worse than RTN on 5 of 9 checkpoints.
  - *Fix:* accumulating the GPTQ Hessian across recurrence steps beats one-step GPTQ and RTN on all 9 checkpoints and recovers bf16-level accuracy on Huginn.
  - *Feedback exposure:* per-channel INT4 mainly damages the non-residual loop-entry adapter. Quantizing the residual core hurts much less.
  - Source: [arXiv 2609.30820](https://arxiv.org/abs/2609.30820)
- **Loopy (2026-10-04)** selects shared low-bit representations (candidates built from channel scaling and orthogonal rotations) by their loss at the final target depth, using progressive full-depth evaluation. It finds that the best quantization configuration changes with recurrent depth. On Ouro-1.4B at W4A4 it cuts LAMBADA perplexity by 36.5% relative to SpinQuant. It is evaluated on Ouro, Huginn and Recurrent-Llama at W4A8 and W4A4 — [Loopy arXiv 2610.05265](https://arxiv.org/abs/2610.05265)
- **Sequential propagation:** "Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization" (2025-04) revisits layer-wise PTQ to account for error propagated from earlier quantized layers. Only the title and abstract-level description were seen — [arXiv 2504.09629](https://arxiv.org/html/2504.09629v1)
- **Joint objectives can fail:** "Cross-Layer Subspace Coupling for LLM Compression" (2026-05) couples adjacent layers' SVD subspaces, a framework that unifies SVD-LLM and Basis Sharing. At a fixed compression ratio d/D = 0.25 it lowers Frobenius reconstruction error by 46.4% (Pythia-70M) and 37.2% (Pythia-1.4B) relative to SVD-LLM.
  - Yet perplexity rises from 50.17 (SVD-LLM) to 159.14, and HellaSwag falls from 43.5% to 30.3%. Every interior configuration was 2.8x to 23x worse in perplexity.
  - The authors attribute this to the residual stream decoupling layers in the forward pass, so joint Frobenius optimization gives per-layer-suboptimal solutions. A third-party review says that explanation lacks isolating evidence.
  - Sources: [arXiv HTML](https://arxiv.org/html/2605.30836); [Moonlight review](https://www.themoonlight.io/en/review/cross-layer-subspace-coupling-for-llm-compression-a-unifying-framework-and-its-empirical-limits)

### Inferences
- **Shared-tensor objective, derived here and not taken from a paper.** Suppose layers i ∈ S share Ŵ, with original weights W_i and inputs X_i. Then:
  - Σ_i ||W_i X_i − Ŵ X_i||²_F = tr((Ŵ − W*) H_Σ (Ŵ − W*)^T) + C, where H_Σ = Σ_i H_i and W* = (Σ_i W_i H_i) H_Σ^-1.
  - C = Σ_i tr(W_i H_i W_i^T) − tr(W* H_Σ W*^T) is the irreducible sharing error.
  - Any GPTQ, LDLQ, QuIP#, QTIP or AWQ implementation can therefore quantize a shared tensor by passing (W*, H_Σ) instead of (W, H).
  - The looped case (all W_i identical, X_i = activations at loop t) reduces to W* = W with H_Σ = Σ_t X_t X_t^T. That is exactly the Hessian accumulation of arXiv 2609.30820.
- **Shared base plus per-layer delta, W_i ≈ B + Δ_i:**
  - The joint calibration objective Σ_i ||(W_i − B − Δ_i) X_i||² is block-separable given B, and is a single (W*, H_Σ) problem given the Δ_i.
  - Alternating minimization works, as in CALDERA's Q/LR alternation (Section 5):
    - Step (a): for fixed B, each Δ_i solves its own layer-wise problem with target W_i − B and Hessian H_i. Use GPTQ for a dense low-bit delta, or a QERA/SVD-LLM-style whitened SVD for a low-rank delta.
    - Step (b): for fixed Δ_i, B = (Σ_i (W_i − Δ_i) H_i) H_Σ^-1, then quantize B with H_Σ.
  - With unequal H_i, W* is not the plain average of the layers. Relaxed Recursive Transformers' "Average" initialization (Section 4) is the special case H_i = I.
- **Inputs that depend on the shared tensor:** in a looped or shared model, X_i itself depends on the shared tensor, which feeds earlier passes. A single closed-form pass is therefore only a proxy, and the Hessians should be re-collected after B is fixed. Loopy chooses its configurations by end-to-end deep loss, and the subspace-coupling paper shows a better proxy can still mean worse perplexity. Both support a final end-to-end check, or a light global fine-tune of scales, as in OmniQuant or QuIP#/QTIP fine-tuning.

### Gaps
- No published GPTQ, AWQ or QuIP#-style method was found that fits one shared tensor to several distinct pretrained layers with Σ H_i. The derivation above is mine and untested.
- Full tables for 2609.30820 (per-checkpoint accuracy and bit settings) could not be read; only abstract-level claims are available.

## 2. Vector quantization and codebooks (AQLM, QuIP#, QTIP, GPTVQ, VPTQ): bpw vs perplexity at 2–3 bits, and cross-layer codebook sharing

### Takeaway
At about 2 bpw, the best VQ and trellis methods keep Llama-2-70B Wikitext2 perplexity at 3.78–3.94 against an FP16 baseline of about 3.12 at ctx 4096, and Llama-2-7B at about 5.7–6.7 against 5.12. Scalar GPTQ or AWQ at 2 bits is several-fold worse or collapses. Lattice and trellis codebooks (QuIP# E8P, QTIP compute-based codes) are fixed and data-independent, so they are trivially shared by every layer. Learned codebooks (AQLM, GPTVQ, VPTQ) are per layer. Explicit cross-layer sharing of a learned codebook exists in a few 2025–2026 papers (an Arm CPU kernel paper, VQRound, LegoLM). These papers report that differences in scale between layers are the main obstacle.

### Cited Findings
- **AQLM model zoo, Wikitext2 perplexity (directly fetched README):**
  - Llama-2-7b: 1x16 = 5.92 (2.4 GB); 2x8 = 6.69 (2.2 GB); 8x8 = 6.61.
  - Llama-2-13b: 1x16 = 5.22; 2x8 = 5.63.
  - Llama-2-70b: 1x16 = 3.83 (18.8 GB); 2x8 = 4.21.
  - Mistral-7b 1x16 = 5.40; Mixtral-8x7b 1x16 = 3.35.
  - PV-tuned: Llama-2-7b 1x16g8 = 5.68; 2x8g8 = 5.90; 1x16g16 = 9.21 (1.7 GB); 1x8g8 = 7.85 (1.34 GB, the "1-bit" model).
  - PV-tuned: Llama-2-70b 1x16g8 = 3.78; Llama-3-8B 1x16g8 = 6.99; 1x16g16 = 9.43; Llama-3-70B 1x16g8 = 4.57; 1x16g16 = 8.67.
  - "num_codebooks" is defined as codebooks *per layer*, each with 2^nbits vectors. The README does not say codebooks are shared across layers.
  - Source: [AQLM GitHub](https://github.com/Vahe1994/AQLM)
- **AQLM in the VPTQ comparison table:** AQLM at 2.07 bits on Llama-2-70B gives Wikitext2 3.94 and C4 5.72, and QuIP# 2-bit gives 3.91 — [VPTQ paper (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.467.pdf), via search extract
- **QuIP# (2024-02, ICML 2024):**
  - The E8P codebook is 2-bit and 8-dimensional, built on the E8 lattice. It has 2^16 entries but needs lookups into only a 2^8-entry table.
  - It uses residual vector quantization for higher rates: 4 bits means rounding with the 2-bit E8P codebook twice. Configs are E8P12 (2-bit), E8P12RVQ3B and E8P12RVQ4B.
  - Llama-2-7B: 3-bit Wikitext2 / C4 = 5.41 / 7.04 against AQLM 5.46 / 7.10. 2-bit = 6.19 in one table extract and 6.66 in another, probably ctx 4096 vs 2048 (unresolved).
  - Sources: [QuIP# arXiv](https://arxiv.org/abs/2402.04396); [quip-sharp README](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/README.md); [llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/4327)
- **QTIP (2024-06, NeurIPS 2024):**
  - Uses incoherence processing (random Hadamard transform) so weights are approximately i.i.d. Gaussian, then trellis-coded quantization with "compute-based codes" that remove TCQ's slowness. Decoding speed at batch 1 on an RTX 6000 Ada (2-bit): QTIP 188 tok/s, QuIP# 186, AQLM 81.5 (Llama-2-7B) — [QTIP GitHub](https://github.com/Cornell-RelaxML/qtip)
  - Llama-2-70B at 2 bits, C4: QTIP 5.48 vs QuIP# 5.71 vs AQLM 5.62 — [QTIP paper](https://arxiv.org/pdf/2406.11235), via search extract
  - Llama-2-70B at 2 bits, Wikitext2: QTIP 3.78 vs QuIP# 3.91, and GLVQ-32D claims 3.36. FP16 Llama-2-7B is 5.12 Wikitext2 / 6.63 C4 at ctx 4096 — [GLVQ](https://arxiv.org/pdf/2510.20984), via search extract
  - Llama-2-7B at 4 bits: QTIP 5.17 vs QuIP# 5.19 vs AQLM 5.21. At 2 bits, a review gives QTIP 7.05 / 6.82 (1MAD / 3INST, apparently without fine-tuning), and a later reproduction gives 5.91 Wikitext2 / 7.76 C4. These conflict; the QTIP paper's own table was not readable here — [Liner review of QTIP](https://liner.com/review/qtip-quantization-with-trellises-and-incoherence-processing); [search extract](https://arxiv.org/pdf/2510.20984)
- **Scalar baselines at 2–3 bits (OmniQuant appendix extract; the model column was unlabeled in the snippet):**
  - W2A16 g128: GPTQ 27.71, AWQ about 1.9e5, OmniQuant 12.97.
  - W3A16 g128: GPTQ 7.85, AWQ 7.92, OmniQuant 7.75.
  - An earlier version's table shows W2A16 g128 OmniQuant 9.72, GPTQ 15.60, AWQ 2.6e5, and W3A16 g128 6.55 / 6.46 / 6.15.
  - Source: [OmniQuant arXiv v3](https://arxiv.org/html/2308.13137v3); [OmniQuant ICLR PDF](https://openreview.net/pdf?id=8Wuvhh0LYW)
- **TesseraQ** reports that at Llama-2-7B W2A16, "all existing methods except OmniQuant failed (>1e3 perplexity)", with OmniQuant at about 37.4 in that setting — [TesseraQ](https://arxiv.org/html/2410.19103v1)
- **GPTVQ (2024-02):** extends GPTQ to vector quantization, with codebooks per group of a layer. Its title claim is "the blessing of dimensionality" — [GPTVQ arXiv](https://arxiv.org/pdf/2402.15319). **VPTQ (2024-09, EMNLP 2024):** second-order-guided, channel-independent vector PTQ with residual VQ, aimed at about 2 bits — [VPTQ arXiv](https://arxiv.org/pdf/2409.17066)
- **Cross-layer codebook sharing, published evidence:**
  - An Arm CPU kernel paper (2024-12/2025-01) aims to use as few codebooks as possible *for all LLM layers* so they fit in the CPU register file. Its codebooks are shared across rows or columns within a layer, and accuracy suffers noticeably at 2-bit — [arXiv 2501.00032](https://arxiv.org/pdf/2501.00032)
  - VQRound (2026-02) learns a shared codebook for rounding variables. Its end-to-end variant updates codebook parameters jointly across all layers under a global objective, allowing cross-layer error compensation — [arXiv 2602.02151](https://arxiv.org/html/2602.02151v1)
  - XFP (2026-05) condenses per-channel Lloyd codebooks into a library of 32 codebooks shared *within* each layer, with each 128-weight group choosing one — [arXiv 2605.14844](https://arxiv.org/html/2605.14844v1)
  - QAM-W (2026-05) uses a *single* Lloyd-Max 2D codebook trained on the unit circular Gaussian, made universal by Hadamard rotation and activation-aware per-channel scaling — [arXiv 2605.26339](https://arxiv.org/abs/2605.26339)
  - GLVQ argues the opposite way: each weight group gets its own learned lattice codebook to match heterogeneous statistics — [GLVQ](https://arxiv.org/html/2510.20984v1)
- **LegoLM (2026-08, one author, preprint, data-free):** "structured weight sharing" with a global block library across the model.
  - It finds that transformer weight matrices have widely varying scales across layers, a key obstacle to cross-layer codebook sharing.
  - Fixes: scalar-block encoding, percentile-selective replacement that keeps outliers, and keeping the first and last blocks unshared.
  - Mistral-7B: +0.03% perplexity at 4.41x compression and −0.02% at 2.67x. Downstream accuracy is within noise at 5.12x (K = 64, p = 99%). It claims to beat "PTQ-8bit", a weak baseline.
  - Failure modes: distributional mismatch in vector blocks (very high perplexity) and outlier dominance in scalar blocks.
  - Sources: [LegoLM arXiv](https://arxiv.org/abs/2608.08652); [awesomepapers listing](https://awesomepapers.io/llm-papers/papers/2608.08652)

### Inferences
- **Fixed codebooks are already shared.** After a random Hadamard transform, each layer's weights look like i.i.d. Gaussian once scaled. A fixed lattice or trellis codebook (E8P, QTIP's 1MAD/3INST) is then already a "codebook shared across all layers", with only per-layer scales differing. This is the cleanest existing evidence that a shared codebook plus per-layer scale works at 2 bits.
- **Codebook storage is negligible.** At 2 bpw the cost is the per-weight code indices, not the codebooks. Sharing codebooks across layers therefore saves almost nothing in bpw. The win from looping has to come from sharing the *weights* (indices), not the codebook.
- **LegoLM bpw equivalents** (vs 16-bit): 4.41x ≈ 3.6 bpw, 5.12x ≈ 3.1 bpw, 2.67x ≈ 6.0 bpw. That puts it in INT3–INT4 territory, where GPTQ, AWQ and QuIP# are already near-lossless. It is not evidence that cross-layer sharing beats PTQ at 2 bits.
- **AQLM code bits** [inference from the scheme definition]: 1x16g8 = 16 bits per 8 weights = 2.0 code bpw; 1x16g16 and 1x8g8 ≈ 1.0 code bpw; 2x8g8 = 2.0 code bpw. Codebook and scale overhead and unquantized embeddings come on top.

### Gaps
- I could not read the QTIP, QuIP# or AQLM tables directly. Llama-2-7B 2-bit numbers conflict across extracts: QuIP# 6.19 vs 6.66; QTIP 5.86 (my recall) vs 5.91 vs 6.82. Use the primary tables before finalizing.
- No paper was found that learns a codebook or index map shared across layers of a *pretrained* LLM and reports Wikitext2/C4 at 2–3 bpw against QuIP#, AQLM or QTIP.

## 3. Residual / multi-stage quantization (the same thing as base + quantized delta)

### Takeaway
Residual VQ is already standard inside the best LLM quantizers: QuIP# builds 3- and 4-bit by applying the 2-bit E8P codebook to successive residuals, and VPTQ uses residual VQ. The CALDERA, LQER, QERA and LoftQ family (Section 5) is two-stage "quantized backbone + low-rank residual". Applied across layers, the same structure becomes "stage 1 = shared base (one per group of layers), stage 2 = per-layer residual".

### Cited Findings
- **QuIP# RVQ:** quantizing to 4 bits means rounding with the 2-bit E8P codebook twice. Configs E8P12RVQ3B and E8P12RVQ4B are the 3- and 4-bit models — [QuIP# arXiv](https://arxiv.org/pdf/2402.04396); [quip-sharp README](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/README.md)
- **QuIP#** claims 3-bit models can scale better than their 4-bit counterparts — [QuIP# alphaxiv](https://www.alphaxiv.org/abs/2402.04396)
- **VPTQ** uses residual vector quantization and channel-independent second-order optimization for extreme low-bit (about 2-bit) PTQ — [VPTQ](https://arxiv.org/pdf/2409.17066)
- **AQLM:** each weight group is the *sum* of M codewords from M codebooks. This is additive (multi-codebook) quantization, a joint relative of residual VQ — [AQLM GitHub](https://github.com/Vahe1994/AQLM)

### Inferences
- **"Shared base + per-layer residual" is two-stage residual quantization in which stage 1's codewords (whole layers or blocks) are reused across layers.** Standard RVQ results then suggest:
  - The stage-1 approximation should be fitted to minimize the error that remains *after* stage 2, as in joint or alternating RVQ, not fitted independently.
  - Stage 2's bit budget is what limits quality.
- **Recursion can stack.** A hierarchy (global base → group base → per-layer delta) is a three-stage RVQ, with bpw given by the accounting in Section 6.

### Gaps
- No LLM paper was found that applies residual VQ *across layers*, with a stage-1 codeword shared by several layers. All published RVQ for LLM weights is within a layer.

## 4. Delta compression (fine-tune deltas vs base), and whether inter-layer deltas behave the same

### Takeaway
Fine-tune deltas compress very well: about 1 bpw is near-lossless for chat fine-tunes with BitDelta, and mixed-precision SVD (Delta-CoMe) holds up on math and code where BitDelta degrades. Compression ratios run to 16x and, with dropout plus quantization, up to 128x. This works because fine-tune deltas are tiny in norm relative to W_base and have fast-decaying spectra. Inter-layer deltas (W_i − shared base) are a different object. Adjacent layers are more similar than distant ones (DOCS), and RRT says the residuals "can be approximated well with low ranks". But converted recursive models still need billions of tokens of uptraining, and no paper reports norm ratios or spectra of inter-layer deltas for modern LLMs.

### Cited Findings
- **BitDelta (2024-02, NeurIPS 2024):** stores the delta as sign(Δ) times a per-matrix scale, which is then distilled on calibration data. Deltas are more than 10x smaller.
  - Llama-2-7B-Chat MT-Bench: BitDelta 6.47 vs baseline 6.56.
  - A rank-128 SVD delta with the *same memory* reaches only 5.85, so the sign structure beats low rank at equal bits for these deltas.
  - Sources: [BitDelta arXiv HTML](https://arxiv.org/html/2402.10193v3); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/187d94b3c93343f0e925b5cf729eadd5-Paper-Conference.pdf); [HF page](https://huggingface.co/papers/2402.10193)
- **Delta-CoMe (2024-06, NeurIPS 2024):** mixed precision over singular vectors. Vectors with larger singular values get more bits, and the smallest are dropped (0-bit).
  - 7B average: Delta-CoMe 53.2 vs BitDelta 49.3 vs low-rank 47.8 vs uncompressed aligned model 53.5.
  - 13B: BitDelta 50.9, low-rank 51.0, aligned 54.5.
  - BitDelta is near-lossless on chat models but degrades significantly on math and code models.
  - Sources: [Delta-CoMe arXiv](https://arxiv.org/pdf/2406.08903); [GitHub](https://github.com/thunlp/Delta-CoMe)
- **PRINMIX (2025-06)** criticizes Delta-CoMe's allocation, which assumes larger singular values always matter more, as heuristic and suboptimal. It proposes SVD-based quantization-error minimization for deltas — [arXiv 2506.11087](https://arxiv.org/pdf/2506.11087)
- **DARE** randomly drops delta elements with probability p and rescales survivors by 1/(1−p).
  - At p = 0.99 a delta shrinks from 417.7 MB to 11.4 MB, but accuracy is poor in that comparison.
  - A comparison table lists DARE at 14.7x / 23.7x and BitDelta and Delta-CoMe at 16x.
  - DeltaDQ (group-wise dropout plus separate quantization) reaches 128x. At that ratio DeltaDQ (m = 8) scores 52.69 while the m = 1 variant scores 0.00.
  - Sources: [DeltaDQ](https://arxiv.org/html/2410.08666v1); [DARE the Extreme](https://arxiv.org/html/2410.09344); [UltraDelta](https://arxiv.org/html/2505.13563v3)
- **DeltaZip** compresses deltas up to 10x using structured pruning plus quantization, aimed at multi-tenant serving — [DeltaZip GitHub](https://github.com/eth-easl/deltazip). **ComPEFT:** magnitude pruning, then quantization of survivors to a single shared value — [search extract via DeltaDQ](https://arxiv.org/html/2410.08666v1)
- **D²-MoE (ICML 2025)** is the closest calibrated "shared base + delta" analogue inside one layer. It merges experts into a base using Fisher-information weighting, then compresses each expert's delta by SVD — [OpenReview](https://openreview.net/forum?id=ziezViPoN1&noteId=kjdxe9QSHw); [ICML poster](https://icml.cc/virtual/2025/poster/43461)
- **DeltaLLM (2025-01):** keeps shared base layers and stores other layers as base + low-rank delta, trained with progressive module replacement (only deltas trained).
  - DeltaPhi with 12% parameter reduction keeps about 90% of base performance.
  - MLP layers are more compressible than attention layers.
  - It claims to be orthogonal to quantization and pruning.
  - Sources: [DeltaLLM arXiv](https://arxiv.org/html/2501.18596v1); [Moonlight review](https://www.themoonlight.io/en/review/deltallm-compress-llms-with-low-rank-deltas-between-shared-weights)
- **Relaxed Recursive Transformers (2024-10, ICLR 2025):**
  - Tied layers are initialized from the *average* of the original layers. Per-layer LoRA is initialized by truncated SVD of the residuals between original layer weights and shared weights, which the paper says "can be approximated well with low ranks".
  - SVD init beats zero init by up to 6.5 points. Stepwise init sometimes degrades after relaxation because the LoRA rank is too low.
  - A recursive Gemma-1B built from Gemma-2B gains up to 13.5 points with uptraining. With 15B uptraining tokens it beats TinyLlama and Pythia.
  - Sources: [RRT arXiv](https://arxiv.org/pdf/2410.20672); [RRT HTML](https://arxiv.org/html/2410.20672v1); [replication repo](https://github.com/avramdj/rrt-lora)
- **DOCS (ICLR 2025):** adjacent layers show higher weight similarity, and similarity falls as layer-index distance grows. First and last layers are exceptions. Base and instruction-tuned variants remain highly similar — [DOCS arXiv](https://arxiv.org/abs/2501.16650); [ICLR slides](https://iclr.cc/media/iclr-2025/Slides/29315.pdf)
- **Basis Sharing (ICLR 2025):** a whitened global SVD over stacked weight matrices from several layers gives a shared basis plus per-layer coefficients. Sharing is worthwhile only for some weight types, with a layer-grouping criterion — [OpenReview](https://openreview.net/forum?id=gp32jvUquq); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf)

### Inferences
- **Why 1-bit deltas work for fine-tunes:** the absolute error of a sign-and-scale delta is about ||Δ||, which is small relative to ||W_base||.
  - For inter-layer deltas, take N layers with pairwise weight cosine similarity ρ (after any permutation or alignment). The optimal shared base (the mean) leaves relative residual energy ||W_i − W̄||² / ||W_i||² ≈ (1 − ρ)(1 − 1/N). This is my derivation for equal-norm, equicorrelated matrices.
  - Unless ρ is high (about ≥ 0.8), the "delta" is nearly as large as the weight itself. A 1–2 bit delta then gives about the same error as quantizing W_i directly to 1–2 bits, so sharing buys nothing.
  - This is a decision criterion to *measure first*: compute ρ and the Hessian-weighted residual C/Σ tr(W_i H_i W_i^T) from Section 1 for each layer group and weight type.
- **The evidence points toward low-rank or small inter-layer deltas only after training.** RRT needs 15B tokens, DeltaLLM trains its deltas, and Cross-Layer Subspace Coupling's training-free joint fit failed badly. Expect a pure closed-form base + 1-bit delta fit of a pretrained, untied model to lose more than INT2 VQ methods do. This is unverified.
- **MLP vs attention:** DeltaLLM finds MLP more shareable, while ALBERT (Section 5) found FFN sharing hurts most. These are opposite findings in different regimes, so sensitivity should be measured per weight type.

### Gaps
- **No published measurement of inter-layer delta norms, ranks or spectra (||W_i − W̄||/||W_i||, effective rank) for Llama-, Qwen- or Mistral-class models was found.** DOCS gives only a similarity index, not reconstruction-relevant norms. This is the most important missing number for the user's idea.
- DeltaLLM training-token count and full tables, and Basis Sharing perplexity numbers, were not retrievable here.

## 5. Low-rank + quantization decompositions (LQ-LoRA, LoftQ, CALDERA, LQER, QERA, ZeroQuant-V2 LoRC, GlowQ)

### Takeaway
"W ≈ Q + LR" is a mature, calibration-driven family. CALDERA (2-bit Q, 4-bit low-rank factors, alternating Hessian-aware updates) is the best result below 2.5 bpw among Q+LR methods. QERA gives the *closed-form* output-error-optimal low-rank correction for a given quantized backbone. GlowQ (ICLR 2026) shares one low-rank factor across modules with the same input, the first "shared correction" result. These are the exact sub-solvers a shared-base + per-layer-delta method needs.

### Cited Findings
- **CALDERA (2024-05, NeurIPS 2024):**
  - Approximates W ≈ Q + LR, with Q, L and R all quantized. The main setting is Q at 2 bits and L, R at 4 bits.
  - Solved by alternating minimization. Q uses the QuIP# quantizer, layer Hessians come from calibration data, and LoRA fine-tuning of part of the low-rank factors is optional.
  - Evaluated on Llama-2 7B/13B/70B and Llama-3-8B. It outperforms prior post-training compression below 2.5 bpw, and rank 256 with 4-bit factors is generally best.
  - Sources: [CALDERA GitHub](https://github.com/pilancilab/caldera); [arXiv](https://arxiv.org/pdf/2405.18886); [follow-up confirming 2-bit Q / 4-bit LR](https://arxiv.org/pdf/2506.02077); [Liner review](https://liner.com/review/compressing-large-language-models-using-low-rank-and-low-precision)
- **LQER (ICML 2024):** adds a high-precision low-rank term to the quantized weight. An activation-induced scale matrix shapes the error's singular values before SVD. L2QER-MXINT (W4A8) averages only a +0.15 perplexity increase across LLMs, against +0.20 for OmniQuant (W6A6) and +1.45 for AQAS (W4A8) — [LQER arXiv](https://arxiv.org/abs/2402.02446); [PMLR](https://proceedings.mlr.press/v235/zhang24j.html); [GitHub](https://github.com/ChengZhang-98/lqer)
- **QERA (ICLR 2025):** an analytical, closed-form solution for the low-rank error-reconstruction term that minimizes *output* error. LQER and LQ-LoRA heuristics are treated as approximations of it.
  - Wikitext2: QERA-exact gives the lowest perplexity, about 0.28 below LQER.
  - 4-bit Llama-3.1-70B: 2.97% higher accuracy than ZeroQuant-V2.
  - Sources: [QERA ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/21718991f6acf19a42376b5c7a8668c5-Paper-Conference.pdf); [ResearchGate](https://www.researchgate.net/publication/384770537_QERA_an_Analytical_Framework_for_Quantization_Error_Reconstruction)
- **ZeroQuant-V2 LoRC (2023-03):** low-rank compensation obtained from the SVD of the quantization error E = W − Ŵ [recalled, not re-fetched] — [arXiv 2303.08302](https://arxiv.org/abs/2303.08302)
- **LoftQ (2023-10, ICLR 2024):** alternates quantization and SVD to find Q, A, B minimizing ||W − Q − AB^T||_F, used to initialize LoRA [recalled, not re-fetched] — [arXiv 2310.08659](https://arxiv.org/abs/2310.08659). **LQ-LoRA (2023-11, ICLR 2024):** iterative low-rank plus quantized decomposition with Fisher weighting and ILP-based mixed configuration — [LQ-LoRA](https://arxiv.org/pdf/2311.12023)
- **GlowQ (2026-03, ICLR 2026):** for modules that share an input (q/k/v; gate/up), it learns *one shared right factor per group* with module-specific left factors. On MoE this cuts the low-rank correction's memory by about 63%. Reported latency figures differ between sources: TTFB −5.6% and throughput +9.6% on the poster, vs −23.4% and +37.4% for GlowQ-S in another listing — [ICLR poster](https://iclr.cc/virtual/2026/poster/10007786); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c9f95e9ec39fa5ad3d0a562b993b92aa-Paper-Conference.pdf)
- **Other recent Q+LR work, titles only seen:**
  - "GPTQ-intrinsic LoRA: a near-optimal algorithm for low-precision quantization with low-rank adaptation" (2026-06) — [arXiv 2606.01412](https://arxiv.org/pdf/2606.01412)
  - D-QRELO, training- and data-free delta compression by quantization plus residual low-rank (2026-04) — [arXiv 2604.16940](https://arxiv.org/pdf/2604.16940)

### Inferences
- For the per-layer delta step (Section 1, step (a)) with a low-rank delta, QERA-exact is the right closed form. Take the SVD of (W_i − B) R_xx,i^{1/2}, truncate to rank r, then map back with R_xx,i^{-1/2}. This is the same whitening idea as SVD-LLM and Basis Sharing.
- For a dense low-bit delta, run GPTQ, LDLQ or QTIP on target (W_i − B) with Hessian H_i.
- CALDERA's alternating loop is the template for alternating B (shared, quantized with H_Σ) and the deltas (per layer, with H_i).
- GlowQ shows that sharing a low-rank *factor* across modules costs little quality. Sharing the right factor across *layers* (same residual-stream input space) is the obvious untested extension.

### Gaps
- CALDERA's Table 1 numbers (bpw vs Wikitext2/C4 for each rank and model) could not be retrieved. LQ-LoRA and LoftQ numbers were also not retrieved.
- No paper was found that pairs a cross-layer shared quantized base with per-layer Q+LR deltas fitted by calibration.

## 6. Quantizing shared weights: error accumulation and sensitivity; effective bpw accounting

### Takeaway
Shared, reused weights are measurably *more* fragile under quantization, through several mechanisms:
- systematic bias that accumulates on each reuse;
- activation-distribution shift across loops;
- calibration that sees only the first pass;
- in ALBERT's case, redundancy already removed by sharing.

Weight-only INT8 is safe on looped LLMs. Weight-only INT4 RTN visibly degrades, especially when extrapolating loop depth. Loop-aware calibration (Hessian accumulated across passes, per-loop scales) largely repairs this. For bpw, sharing k blocks across N layers divides the base cost by N/k, so to beat 2-bit VQ the per-layer delta must cost well under about 1.5 bpw and still recover quality. No published result shows this is achievable post-training.

### Cited Findings
- **Ouro-1.4B weight-only fake quantization of the shared 24-layer core** (group-128 RTN; 168 Linears, 1.23B parameters; trained at 4 loops), evaluated at loop 8:
  - Wikitext2 perplexity: FP 10.68; INT8 10.69; FP8 10.92; INT4 12.48. Hidden-state cosine to FP: 0.998 / 0.990 / 0.927.
  - GSM8K 8-shot (200 problems) at T = 2 / 4 / 8 loops: FP 70.0 / 79.0 / 78.5; INT8 68.0 / 78.0 / 76.5; FP8 70.0 / 80.5 / 72.5; INT4 56.0 / 73.0 / 60.0.
  - INT4 error saturates by loop 2 at about 0.35–0.37 relative error, with plateau e* = ||q|| / sqrt(1 − gain²). Empirical gain at loop 8 is INT8 1.06, FP8 0.91, INT4 0.81, so the Jacobian is not loop-invariant. A shared J leaves about 30% directional residual.
  - Recommendation: give loop 1 (and possibly loop 2) its own compensation.
  - Source: [Loop_Transformer_Quantization GitHub](https://github.com/Nemo0412/Loop_Transformer_Quantization) (unreviewed repo)
- **"Quantizing Recursive Reasoning Models" (2026-07, preprint):** weight-tied recursive reasoners.
  - 8-bit quantization is a small perturbation that the recursion damps.
  - Per-tensor 4-bit adds a systematic bias that builds up on each reuse; exact Sudoku accuracy drops from 84.1% to 0.0%. Per-block activation scaling recovers near-full-precision accuracy.
  - Activation quantization dominates the damage, and weight-only INT4 hurts the tied model more than an untied one.
  - Source: [arXiv 2607.16237](https://arxiv.org/html/2607.16237) (via search extract)
- **LoopQ (2026-05):**
  - Diagnoses distribution shift across loops, state reuse across loop transitions, and recursive error accumulation.
  - Keeps one shared quantized backbone plus loop-dependent activation scales and cross-loop adapters. These touch under 5% of transform parameters and under 0.12% of total parameters.
  - At W4A4 it improves average downstream accuracy by 68.8% and lowers average perplexity by 87.7% relative to the strongest static PTQ baseline.
  - A later paper reports LoopQ at 65.88% GSM8K on Ouro-1.4B W4A4 vs FlatQuant 68.31%.
  - Sources: [LoopQ arXiv](https://arxiv.org/abs/2605.16343); [ResidualQuant 2610.10381](https://arxiv.org/html/2610.10381v1)
- **"Quantizing Looped Transformers" (2026-09):** feedback exposure (the loop-entry adapter is most sensitive) and calibration blindness. The Hessian-accumulation fix is detailed in Section 1 — [arXiv 2609.30820](https://arxiv.org/abs/2609.30820)
- **Loopy (2026-10):** the best shared quantization configuration changes with recurrent depth, so it selects at deployment depth. Ouro-1.4B W4A4 LAMBADA perplexity is −36.5% vs SpinQuant — [arXiv 2610.05265](https://arxiv.org/abs/2610.05265)
- **ALBERT vs BERT:** ALBERT is much less robust to quantization. With k-means quantization it keeps 93.4% of performance at 4-bit and 72.5% at 3-bit. The authors argue that sharing and quantization remove partly overlapping redundancy — [An Investigation on Different Underlying Quantization Schemes for PLMs](https://arxiv.org/pdf/2010.07109)
- **ALBERT's own ablation:** most of the drop from sharing comes from sharing FFN parameters, while sharing attention costs almost nothing at E = 128 — [ALBERT](https://arxiv.org/pdf/1909.11942)

### Inferences
- **Effective bpw accounting** (my derivation). Take N layers each with an m x n matrix of a given type, and k unique shared bases at b_s bits. Each layer stores either a rank-r delta with factors at b_r bits or a dense delta at b_d bits, plus per-layer scale overhead o:
  - bpw = (k/N)·b_s + r·(m+n)·b_r/(m·n) + (b_d if a dense delta) + o
- **Llama-2-7B shapes:** for square 4096 x 4096 attention matrices, r(m+n)/(mn) = r/2048. A rank-r delta therefore costs r/128 bpw with FP16 factors and r/512 bpw with 4-bit factors. For the MLP (11008 x 4096), r(m+n)/(mn) ≈ r/2986.
- **Worked examples** (N = 32 layers):

  | Setup | bpw |
  |---|---|
  | k = 8 (4x sharing), base INT4, no delta | 1.0 |
  | k = 8, base INT4, rank-64 FP16 delta | 1.0 + 0.5 = 1.5 |
  | k = 8, base INT4, rank-256 4-bit delta (CALDERA-like) | 1.0 + 0.5 = 1.5 |
  | k = 8, base INT4, dense 1-bit BitDelta-style delta | 1.0 + 1.0 = 2.0, plus scales |
  | k = 16 (2x sharing), base 2-bit QTIP-style, rank-64 4-bit delta | 1.0 + 0.125 ≈ 1.1 |
  | k = 4 (8x sharing), base FP16, rank-128 FP16 delta (RRT-like) | 2.0 + 1.0 = 3.0 |

- **Comparison targets** (Llama-2, ctx 4096, FP16 7B 5.12 / 70B ≈ 3.12 Wikitext2):

  | Target | 7B Wikitext2 | 70B Wikitext2 |
  |---|---|---|
  | 4-bit VQ/trellis | 5.17–5.21 | — |
  | 3-bit QuIP# | 5.41 | — |
  | 2-bit QuIP#/QTIP/AQLM | about 5.7–6.7 | 3.78–3.94 |
  | about 1-bit AQLM-PV (1x8g8) | 7.85 | — |

  A sharing scheme at about 1.5–2 bpw must beat about 6 perplexity on 7B and about 3.9 on 70B to be competitive. No published post-training sharing method approaches this. RRT needs uptraining, DeltaLLM loses about 10% at only 12% parameter reduction, and LegoLM's best is about 3.1 bpw-equivalent.
- **Error reuse:** a shared base is reused N/k times per forward pass and, in a looped deployment, its quantization *bias* is reused every loop. This argues for:
  - unbiased, incoherence-processed quantization (Hadamard plus VQ/trellis rather than per-tensor RTN);
  - Hessians summed over all uses (H_Σ);
  - small per-use scales (LoopQ, Loopy);
  - the most bits on non-residual entry points (feedback exposure).
- **Stacking:** sharing and quantization remove overlapping redundancy (ALBERT finding), so gains are sub-multiplicative. Budget for the shared base needing *more* bits than an untied model's layers, for example INT4–INT8 base rather than INT2. Spend the savings on the number of distinct layers removed.

### Gaps
- No published experiment quantizes a model converted post-training into shared base + deltas (RRT or DeltaLLM style) and reports bpw vs Wikitext2 against QuIP#, AQLM or QTIP. The comparison above is constructed, not measured.
- No published sensitivity study of tied embeddings / LM head under PTQ was found in this session.
- Exact LoopQ per-model tables and the TRM/HRM details of 2607.16237 were not readable; the figures are abstract-level.
