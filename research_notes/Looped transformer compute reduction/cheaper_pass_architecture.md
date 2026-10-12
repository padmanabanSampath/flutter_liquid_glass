# Making each pass of a looped (weight-shared) transformer cheaper: per-pass architecture

Scope: architecture changes to the shared block itself, so that each loop pass costs fewer FLOPs or bytes. Covered: MoE inside the loop, per-pass FFN or width reduction and activation sparsity, low-rank, structured and 2:4 layers, cheaper attention and cross-pass KV sharing, skipping sublayers inside a pass, cheap dense-to-sparse conversion, and efficient ViT blocks for per-frame vision. Not covered (other researchers have these): fewer passes or adaptive depth, fewer tokens, loop execution tricks, and hardware or precision.

Source-type tags used throughout:
- **[P] primary:** text I read directly in an official GitHub README or repo, or an official issue thread.
- **[S] search excerpt:** a paper's own text or tables as quoted by the web-search tool. The paper PDF itself could not be opened, because arxiv.org, openreview.net, huggingface.co, ACL Anthology and nvidia.com were blocked by the proxy.
- **[2nd] secondary:** a third-party review, blog or social post.

Dates are first-arXiv or venue dates. "Train?" says whether a method needs training. "FLOPs" means an analytical or counted saving. "Measured" means wall-clock.

Baseline cost model (my arithmetic, used in the Inferences below; standard formulas, not from a source): for one pre-LN block with model width d, MHA and a SwiGLU FFN of hidden size about 8/3·d, one token costs about 2·(4d² + 8d²) = **24d² linear FLOPs per pass**. Q/K/V/O account for 8d² (1/3) and the FFN for 16d² (2/3). Attention scores and the value mix add about 4·d·L FLOPs per token, where L is the attended context. A looped model with R passes costs R times this, so every per-pass saving multiplies by R. At batch-1 decode on phones, Jetson or WebGPU, latency is set mainly by the weight bytes read per pass, not by FLOPs.

---

## 1. MoE inside looped / universal transformers (MoEUT, SUT, MoR+MoE, 2025–2026 looped-MoE)

### Takeaway
Sparse MoE is now the main way to make looped models scale. Several 2026 papers show dense looped models lose to compute-matched untied baselines, while looped MoE models match or beat them. Each pass activates only k experts, so per-pass FLOPs stay low, and the router picks different experts on different passes, which restores the per-depth specialisation that weight tying removes. SUT and MoEUT show the same effect at small scale: equal quality at roughly 40–50% of the MACs of a dense model. All of these need training from scratch or continued pretraining. Edge latency is untested: no paper reports phone, Jetson or WebGPU numbers for a looped MoE.

### Cited Findings
**MoEUT (Csordás et al., arXiv 2405.16039, May 2024; NeurIPS 2024)**
- MoEUT puts σ-MoE FFN experts and SwitchHead MoE attention into a Universal Transformer that is shared in layer groups. G is the number of non-shared layers per group (default `group_size=2`). The FFN defaults are `ff_k=8` active experts of `ff_expert_size=128` each, and the attention default is `att_k=2` active attention experts. The entropy regularisers are γ=0.01 for the MLP and δ=0.001 for attention. [P] — [MoEUT README](https://github.com/RobertCsordas/moeut)
- The README says `att_n_experts=1` (SwitchHead off) runs faster, with "minimal" perplexity loss, and still beats the dense baseline. This was tested at 244M and 768M. It gives no numbers. The released configs are MoEUT_44M, 126M, 244M, 318M, 727M and 1B. [P] — [MoEUT README](https://github.com/RobertCsordas/moeut)
- Per the paper, σ-MoE FFNs "cut MAC requirements roughly in half" relative to the dense counterpart. The attention uses a quarter of the dense model's heads at double the head dimension, so the V/O projections match dense MACs while the K/Q and attention-matrix costs drop. [S] — [MoEUT OpenReview PDF](https://openreview.net/pdf?id=ZxVrkm7Bjl), [arXiv 2405.16039](https://arxiv.org/abs/2405.16039)
- At equal parameters (same d_model and layer count as dense), MoEUT "slightly outperforms" dense on C4, and the gap grows with scale. At equal compute it "outperforms other models by a large margin". Attention gets 10–15% of the non-embedding parameter budget. The non-shared control "performs significantly worse". G=2 was best in the 244M ablation. Tested up to about 1B (1040M) parameters. [S] — [arXiv 2405.16039](https://arxiv.org/pdf/2405.16039)
- Training code is in a separate repo. The kernel is the CVMM Triton kernel from σ-MoE. Triton fp16 is broken on Volta from PyTorch 2.2 onward. [P] — [MoEUT README](https://github.com/RobertCsordas/moeut)

**SwitchHead (MoE attention used in MoEUT; arXiv 2312.07987; NeurIPS 2024)**
- At 262M parameters on C4, SwitchHead matches dense perplexity with **44% of the compute and 27% of the memory**. [S] — [SwitchHead arXiv](https://arxiv.org/pdf/2312.07987), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/87be61bf9338389702712f5e9754a986-Paper-Conference.pdf)
- On WikiText-103 at 47M parameters, the 2-head SwitchHead reaches **12.27 ppl at 170.4M MACs**. The 10-head dense model reaches **12.31 ppl at 453.4M MACs** and uses 3.5M memory floats against SwitchHead's 0.8M. On C4 at 47M, the scores are 22.49 against 22.63. [S] — same sources

**SUT, Sparse Universal Transformer (Tan et al., arXiv 2310.07096, Oct 2023; EMNLP 2023)**
- SUT is one shared block with SMoE in both the FFN and the attention heads, a mutual-information routing loss, and stick-breaking halting. Halting cuts inference compute by about 50% on formal-language tasks with little loss. [S] — [arXiv 2310.07096](https://arxiv.org/abs/2310.07096v1)
- WMT'14 En-De, Table 1 [S] — [arXiv PDF](https://arxiv.org/pdf/2310.07096):

  | Model | Params | BLEU | MACs |
  |---|---|---|---|
  | SUT-base | 66M | 29.2 | 787M |
  | Transformer-base | 65M | 27.3 | 604M |
  | SUT-big | 110M | 29.4 | 787M |
  | Transformer-big | 210M | 29.3 | 2090M |

  The "half the computation and parameters" claim holds only against Transformer-big. SUT-base uses *more* MACs than Transformer-base.

**Looped-MoE scaling (2026)**
- *Sparse Layers are Critical to Scaling Looped Language Models* (Lee, Biloki, Hu, May; arXiv 2605.09165, May 2026). Looped-MoE scales better than the standard baseline, and dense looped models do not. The authors attribute this to **routing divergence between passes**. In a compute-optimal Looped-MoE at 1e18 FLOPs with k=2 and E=8, **25–53% of tokens get completely non-overlapping experts on pass 2 versus pass 1, and only 4–14% get identical experts** (layers 1–6 and 8; layer 7 is an outlier). Loop boundaries also make good early-exit points. [S] — [arXiv 2605.09165](https://arxiv.org/abs/2605.09165), [PDF](https://arxiv.org/pdf/2605.09165)
- *SMELT* (Wang et al., Tsinghua/ByteDance Seed; arXiv 2609.01343, Sep 2026). It loops the **middle half of the layers twice** in an MoE. Per-token FLOPs, non-embedding parameters **and KV cache** are all matched against the unlooped MoE. Twice beat 3 or 4 loops. The scaling-law exponent is 0.250 against 0.237 for the baseline. It saves **6.8–18.0% of training FLOPs** on the compute-optimal frontier, and **14.7% at 1e21 FLOPs**. It was tested up to 54B non-embedding parameters. Gains are largest on code, and the second visit reduces the attention sink. [S] — [arXiv 2609.01343](https://arxiv.org/abs/2609.01343), [HTML](https://arxiv.org/html/2609.01343v1)
- *How to Loop MoE: Flatten the Experts, Untie the Attention* ("Foil"; Wang et al., arXiv 2609.35751, Sep 2026). Expert parameters and per-token expert compute are held fixed. Foil halves the expert layers, doubles the experts per layer and doubles the passes, and gives **each pass its own attention** while experts and routers stay shared. At 100B tokens the most flattened variant ends **0.012 nat below baseline** at equal parameters and compute, with equal or better downstream scores. The distinct experts a token reaches grow **from 16 (no loop) to 36 (8 passes)**. The gains are in loss, not wall-clock, and come from one width and mostly one seed. Code is at github.com/SR-A-W/how-to-loop-moe. [S] — [arXiv 2609.35751](https://arxiv.org/abs/2609.35751); [2nd] — [Ken Ashe blog](https://kenashe.ai/blog/2026-09-29-foils-recipe-for-looped-moe-flatten-the-experts-untie-the-attention)
- *Tying the Loop: Tied Expert Layers in MoE LMs* (Jaggi et al., EPFL; arXiv 2606.16825, Jun 2026). Expert weights are shared across consecutive layers, while routers, attention and norms stay per-layer. Grouping g layers cuts expert parameters g-fold (g=1, 2, 4), and the tied learning rate is divided by √g. It was tested on OLMoE, Qwen3-MoE and DeepSeekMoE. [P] — [epfml/looped-moe README](https://github.com/epfml/looped-moe). The abstract reports **almost 2× memory reduction at "virtually no degradation"** in perplexity or downstream scores. [S] — [arXiv 2606.16825](https://arxiv.org/pdf/2606.16825). This saves memory only: per-token FLOPs are unchanged.
- *Loop the Loopies!* (Gao et al., IQuest Research; arXiv 2607.16051, Jul 2026). There are two looped MoE models, **20B total / 2B active** and **6B total / 0.6B active**, and each MoE layer loops twice before moving on ("layer-loop"). Layer-loop lags model-loop early in training and overtakes it later. Loopie beats compute-matched vanilla baselines on all four rungs of its scaling ladder. [S] — [arXiv 2607.16051](https://arxiv.org/abs/2607.16051). Per secondary sources, the 20B-A2B model beats a compute-matched vanilla 30B-A3B MoE and scores MMLU 81.28. The checkpoint and code links returned 404 at the time. [2nd] — [alphaXiv post](https://x.com/askalphaxiv/status/2079283518224691582), [ArXivIQ](https://arxiviq.substack.com/p/loop-the-loopies)
- *Looping Beyond Twice: A Scalable Recipe for Looped MoE* (arXiv 2610.01153, Oct 2026). One search snippet says "experiments across 100M–1.7B models show stable scaling to 9–12 loops". A second search did not confirm this. [S, low confidence] — [arXiv 2610.01153](https://arxiv.org/abs/2610.01153)
- Other 2025–2026 looped-MoE or recursive-expert titles, listed only and not read: MoRE (Mixture of Reused Experts, 2609.18176), LoopMoE (2606.04438), UniPool (2605.06665), Mixture of Universal Experts (2603.04971), ReXMoE (2510.17483), Chain-of-Experts (2506.18945). [P, titles only] — [looped-lm-survey list](https://github.com/ArminAzizi98/looped-lm-survey)

**Mixture-of-Recursions (MoR; Bae et al., arXiv 2507.10524, Jul 2025; NeurIPS 2025)**. This is token routing *over depth* rather than over experts.
- The models are 135M–1.7B. At 16.5e18 training FLOPs, MoR (expert-choice, 2 recursions) reaches **43.1% average few-shot accuracy against 42.3% for vanilla, with about 50% fewer unique parameters**. [S] — [arXiv 2507.10524](https://arxiv.org/pdf/2507.10524), [aimodels.fyi](https://www.aimodels.fyi/papers/arxiv/mixture-recursions-learning-dynamic-recursive-depths-adaptive)
- Inference throughput is "up to 2×" over vanilla at similar accuracy (README); v1 says 2.18× and a later summary says 2.06×. Pretrained 360M checkpoints are Llama-based and trained on FineWeb-Edu. [P] — [MoR README](https://github.com/raymin0223/mixture_of_recursions); [S] — [arXiv v1 HTML](https://arxiv.org/html/2507.10524v1)

**Mixture of LoRAs inside a shared FFN (arXiv 2512.12880, Dec 2025)**
- LoRA experts sit inside the shared FFN of a recursive transformer, with 8 experts and top-2 routing. The ModernALBERT models (50M–120M) reportedly beat larger fully parameterised baselines on GLUE, SQuAD-v2 and BEIR. The experts can be **merged into one adapter at inference**, which removes the routing cost. Merge-accuracy numbers were not retrieved. [S] — [arXiv 2512.12880](https://arxiv.org/abs/2512.12880v1)

**Context: dense looped LMs without MoE**
- Ouro (ByteDance; arXiv 2510.25741, Oct 2025) has 1.4B and 2.6B models trained on 7.7T tokens, with 4 recurrent steps by default. The authors claim they match SOTA LLMs of up to 12B. [S] — [arXiv 2510.25741](https://arxiv.org/pdf/2510.25741)
- Huginn-0125 (arXiv 2502.05171, Feb 2025) is a 3.5B recurrent-depth model ("nebel-raven-3.5b") trained on 4096 AMD GPUs and evaluated with `mean_recurrence=32`. [P] — [seal-rg/recurrent-pretraining README](https://github.com/seal-rg/recurrent-pretraining)

### Inferences
- **Compute versus parameter ratio.** A looped MoE with E experts, top-k, and R passes holds E experts' worth of parameters but spends about R·k experts' worth of FLOPs per token. The parameters that looping frees can therefore buy a larger E at fixed per-pass FLOPs. MoEUT (FFN MACs about 0.5× dense) and SwitchHead (44% compute) show roughly 2× per-pass FLOP cuts at equal quality at the ≤1B scale.
- **Batch-1 bandwidth on edge devices.** Per pass, a looped MoE reads attention, router and k experts' weights. Without MoE it reads the whole FFN. So MoE cuts *per-pass* weight bytes to about k/E of the FFN bytes, whether or not the experts change between passes. Routing divergence (25–53% fully different experts on pass 2) does hurt one thing loops could otherwise exploit: a small shared block that stays resident in on-chip cache across passes. With divergence, R passes touch up to min(E, R·k) distinct experts. For example, E=8, k=2, R=4 can touch all 8.
- **Memory.** MoE spends back the memory that looping saved. On phones and in browsers, where RAM, not FLOPs, often binds, Tying-the-Loop-style expert tying or Foil-style flattening is the parameter-neutral version: expert compute and parameters stay fixed while the loop count rises.
- **Untying attention.** Foil's finding that attention should be untied per pass while experts stay shared suggests a cheap split for an A→B→A→B student: share the expert pools and give each pass a small private attention block, or a LoRA on it.
- **Missing evidence.** None of the looped-MoE papers report wall-clock latency on edge devices. Their gains are training-FLOP or loss gains, and they need pretraining or long continued training.

### Gaps
- Exact MoEUT C4 perplexity tables and its wall-clock numbers. The paper could not be opened, and the README gives no numbers.
- Loopie and Looping-Beyond-Twice benchmark tables. These come from search excerpts and secondary sources only.
- I found no measured latency of any looped MoE on a phone, a Jetson or WebGPU. I also found no study of MoE expert-gather kernels on WebGPU or mobile NPUs.
- I found no paper on MoE *distilled from a dense teacher into a looped student*. The closest are MoR, Relaxed Recursive Transformers (dense), and MoL with distillation-based initialisation.

---

## 2. Per-pass width / FFN reduction and contextual activation sparsity (Deja Vu, CATS, TEAL, ReLU-fication / ProSparse, TurboSparse); stability across passes

### Takeaway
Activation sparsity gives the biggest batch-1 decode wins and is the most edge-proven family. Training-free options (TEAL, CATS, R-Sparse, WINA) reach 40–50% sparsity with small quality loss and **1.15–1.8× measured decode speedups** on GPUs. ReLU-fication with continued training (ProSparse, TurboSparse) reaches about 90% FFN sparsity, and with offloading (PowerInfer-2) it runs a 47B MoE at **11.68 tok/s on a OnePlus 12 phone**. I found **no measurement of whether activation sparsity patterns stay stable across loop passes**. The closest evidence, MoE routing divergence of 25–53% fully different experts between passes, suggests that sparsity masks and thresholds should be per pass.

### Cited Findings
- **TEAL** (Liu et al., arXiv 2408.14690, Aug 2024; ICLR 2025 Spotlight). **Training-free**, magnitude-based activation sparsity on hidden states throughout the model, at 40–50% model-wide. Measured **single-batch decoding speedups of up to 1.53× at 40% and 1.8× at 50%**, on Llama-2, Llama-3 and Mistral at 7B–70B, with "minimal degradation". It needs per-model threshold calibration from activation histograms. The paper claims compatibility with weight quantization, but the repo's kernels are FP16-only. [P] — [TEAL README](https://github.com/FasterDecoding/TEAL)
- **CATS** (arXiv 2404.08763; COLM 2024). 50% activation sparsity on Mistral-7B and Llama2-7B, within 1–2% of base on downstream tasks. A custom kernel gives about **15% faster inference**, and the README doesn't say whether that is MLP-only or end-to-end. It works training-free, and fine-tuning recovers more. [P] — [CATS README](https://github.com/ScalingIntelligence/CATS)
- **Deja Vu** (ICML 2023). On OPT-175B, about **80% sparsity on attention heads and 95% on MLP neurons, about 85% overall**, using small learned per-layer predictors trained on C4 activations. Latency drops **over 2× versus FasterTransformer and over 6× versus Hugging Face**. Train? Only the predictors; the base model is untouched. [P] — [DejaVu README](https://github.com/FMInference/DejaVu); [S] — [arXiv 2310.17157](https://arxiv.org/pdf/2310.17157)
- **R-Sparse** (ICLR 2025). **Training-free**, rank-aware activation sparsity that uses input-channel sparsity plus SVD components and needs no predictor. It targets "small-batch on-device" inference and SiLU/GELU models, whose small negative values otherwise demand "extensive continual pre-training". [S] — [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30686). Per a secondary review, it matches the full model at 50% model-level sparsity, with a 43% end-to-end efficiency gain from a custom kernel (not confirmed in the primary source). [2nd] — [Moonlight review](https://www.themoonlight.io/en/review/r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference)
- **WINA** (arXiv 2505.19427; ICLR 2026). **Training-free**. It scores neurons by hidden-state magnitude times the column ℓ2-norm of the weight, and claims less degradation than R-Sparse and TEAL at equal sparsity. Code is at github.com/microsoft/wina. [S] — [alphaXiv](https://www.alphaxiv.org/abs/2505.19427.md)
- **ProSparse** (arXiv 2402.13516, Feb 2024). ReLU swap plus a sparsity penalty that ramps up in sine-curve stages, then FATReLU, which applies a positive threshold. It reaches **89.32% sparsity on Llama2-7B and 88.80% on 13B**, with performance comparable to the Swish originals; ReluLLaMA had 66.98% and 71.56%. It reports **up to 4.52× inference speedup** with PowerInfer and custom GPU operators. **Needs continued training.** [S] — [arXiv 2402.13516](https://arxiv.org/html/2402.13516v2), [operators repo](https://github.com/Raincleared-Song/sparse_gpu_operator)
- The PowerInfer README FAQ says a ReLU 70B model fine-tuned on only **5B tokens** shows a "noticeable downgrade" in metrics. ReLU-fication needs a real token budget. [P] — [PowerInfer README](https://github.com/SJTU-IPADS/PowerInfer)
- **TurboSparse / dReLU** (arXiv 2406.05955, Jun 2024). Uses **<150B tokens (<1% of typical pretraining)** to reach **about 90% average FFN sparsity** in Mistral, with improved capability. Mistral and Mixtral then activate **2.5B and 4.3B** parameters per token. [S] — [arXiv 2406.05955](https://arxiv.org/html/2406.05955v1). The README says this cost "$0.1M". [P] — [PowerInfer README](https://github.com/SJTU-IPADS/PowerInfer)
- **PowerInfer-2** (arXiv 2406.06282, Jun 2024), smartphone results. **TurboSparse-Mixtral-47B runs at 11.68 tok/s** on a OnePlus 12 (24 GB) with 19 GB available, and at **2.13 tok/s with only 7 GB**. It is **24.6× (up to 27.8×) faster than llama.cpp** and 3.84× faster than LLMFlash on the OnePlus 12. The earlier version said 22.2×. [S] — [arXiv 2406.06282v2](https://arxiv.org/html/2406.06282v2); [P] — [PowerInfer README](https://github.com/SJTU-IPADS/PowerInfer)
- **PowerInfer** (Dec 2023), desktop GPU results. Measured on an RTX 4090: 13.20 tok/s average and 29.08 peak, up to 11.69× over llama.cpp. On an RTX 2080 Ti with INT4: up to 8× on Falcon-40B. [P] — [PowerInfer README](https://github.com/SJTU-IPADS/PowerInfer)
- **Stability across loop passes:** I found no paper that measures activation-sparsity masks per loop iteration. The closest evidence is the Looped-MoE routing divergence above (25–53% of tokens fully switch experts between passes, 4–14% keep identical experts). [S] — [arXiv 2605.09165](https://arxiv.org/pdf/2605.09165). An analogue from diffusion models finds the set of active columns grows monotonically over denoising steps. [S] — [arXiv 2606.00567](https://arxiv.org/pdf/2606.00567)
- **Per-pass width (looped FFN wide-and-deep reuse):** VersatileFFN (arXiv 2512.14531, Dec 2025) adapts "wide-and-deep reuse" of FFN parameters; I have the title only. [P, title] — [looped-lm-survey](https://github.com/ArminAzizi98/looped-lm-survey)

### Inferences
- **Per-pass FLOP accounting.** The FFN is about 2/3 of a block's linear FLOPs. At 50% FFN activation sparsity (TEAL or CATS level), each pass's linear FLOPs drop by about 1/3. At 90% FFN sparsity (ProSparse or TurboSparse level), they drop by about 60%. At batch 1 these are mainly *bytes-read* savings. The measured 1.5–1.8× (TEAL) is in line with that.
- **Per-pass thresholds.** In a looped model the same weights see a different input distribution on each pass. TEAL-style magnitude thresholds and Deja Vu-style predictors should therefore be calibrated per (block, pass). That is cheap: a table of thresholds or a tiny predictor indexed by pass. The routing-divergence result implies one shared mask would be wrong for a large fraction of tokens.
- **Partial-width passes.** Using a narrower FFN on some passes is a static version of the same idea. With MoE-style experts it amounts to choosing k per pass (for example k=2 on early passes, k=1 on later ones). I found no paper that tests pass-dependent k.
- **Kernel availability.** The phone (PowerInfer-2) and RTX numbers depend on custom sparse kernels and CPU/GPU split. On WebGPU, and on Jetson TensorRT, unstructured activation-sparse GEMV kernels are not standard, so speedups would require custom kernels. This is my judgement; I found no source.

### Gaps
- No direct measurement of activation sparsity stability across loop iterations in Huginn, Ouro, MoR or MoEUT.
- No TEAL or CATS numbers on Jetson, mobile or WebGPU. The speedups are on datacenter or desktop GPUs, and the specific GPU for TEAL is not named in the README.
- I did not look up dReLU or ProSparse results for small (<1B) models of the size used in vision or edge students.

---

## 3. Low-rank or structured layers in the shared block (low-rank FFN, Monarch/butterfly, 2:4 sparsity on Ampere/Orin)

### Takeaway
Monarch and Block Tensor-Train layers match dense quality at about 2× training speed, or 3.8× less compute for ViT-S/32, but I found no inference numbers for them on edge devices. 2:4 semi-structured sparsity is the only structured option with hardware support on Jetson Orin, whose Ampere sparse tensor cores give 275 sparse versus 138 dense INT8 TOPS. Measured end-to-end gains elsewhere are only **about 1.1–1.4×**. A recent GitHub issue shows **0% speedup** when the build bypasses sparse tactics. One-shot 2:4 pruning also costs a lot of quality on 7B models, so 2:4 needs sparsity-aware fine-tuning or distillation.

### Cited Findings
- **Monarch** (Dao et al., ICML 2022). Training speedups at equal quality: **GPT-2-Small on WT103, 20.7 versus 20.6 ppl at 1.8× speedup; GPT-2-Medium, 20.3 versus 20.9 at 2.0×; ViT-B/16 on ImageNet, 78.9 versus 78.5 top-1 at 2.0×**. Monarch replaces the dense attention-projection and FFN weights. The authors say plain low-rank and elementwise sparsity gave worse efficiency-quality tradeoffs in end-to-end training. Train? Yes, these are training-time results. [S] — [arXiv 2204.00595](https://arxiv.org/pdf/2204.00595), [ICML slides](https://ICML.cc/media/icml-2022/Slides/17900_R7TNeV4.pdf)
- **Compute Better Spent / BTT** (ICML 2024). Block Tensor-Train, which contains Monarch, **matches dense ViT-S/32 on ImageNet-1k with 3.8× less compute** and beats dense for small GPT-2. It needs weight normalisation to avoid divergence. [S] — [PMLR v235](https://proceedings.mlr.press/v235/qiu24f.html), [arXiv 2406.06248](https://arxiv.org/html/2406.06248v1)
- **Orin hardware.** AGX Orin's headline INT8 figure assumes 2:4 sparsity: 275 sparse TOPS against **138 dense TOPS**. 2:4 lets tensor cores skip zeros, which "doubles" the math throughput. [S] — [NVIDIA Jetson AGX Orin technical brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)
- **Measured 2:4 speedups (not on Orin):**
  - TensorRT INT8 with sparsity: about 1.4× over dense-INT8 for both PTQ and QAT. [S] — [NVIDIA blog](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/)
  - PyTorch on SAM inference: 10% end-to-end, on A100, for ViT-B/H backbones. [S] — [PyTorch blog](https://pytorch.org/blog/accelerating-generative-ai/)
  - PyTorch on ViT-L MLP linears in training: 1.3× forward plus backward. [S] — [PyTorch blog](https://pytorch.org/blog/accelerating-neural-network-training/)
  - BERT inference: 1.3×, at 86.48 F1 sparse versus 86.92 dense. cuSPARSELt matmul runs up to 1.6× against a theoretical 2×. [S] — [PyTorch tutorial](https://pytorch.org/tutorials/advanced/semi_structured_sparse.html)
- **Pitfall** (GitHub issue, Jul–Sep 2026). A 2:4 Qwen3-4B (ModelOpt 0.45, SparseGPT plus sparsity-aware fine-tuning) on an L4 with TensorRT-LLM 1.3.0rc18 was **identical to dense: about 485 tok/s plateau and 33.8 ms ITL**. The cause was `--gemm_plugin auto`, which routes GEMMs to cuBLASLt and bypasses sparse tactics. Rebuilding with `--gemm_plugin disable --weight_sparsity` was "measurably faster than dense", with no numbers given. The issue is still open, with no NVIDIA explanation of which shapes or which phase (prefill or decode) benefit. [P] — [NVIDIA/Model-Optimizer #1974](https://github.com/NVIDIA/Model-Optimizer/issues/1974)
- **Orin specifically:** "No direct measurement of a 2:4 speedup on Jetson Orin with TensorRT" was found. One Orin paper only *projects* that 2:4 on the DLA could roughly halve batch-16 latency. [S] — [search summary incl. arXiv 2608.11770](https://arxiv.org/pdf/2608.11770)
- **Quality of one-shot 2:4 (no retraining)**, WikiText ppl on Llama-2 7B / 13B / 70B: dense 5.12 / 4.57 / 3.12; **2:4 SparseGPT 10.17 / 8.32 / 5.40; 2:4 Wanda 11.02 / 8.27 / 5.16**. 4:8 is milder: Wanda gives 7.97 / 6.55 / 4.47. [P] — [Wanda README](https://github.com/locuslab/wanda)

### Inferences
- **Error compounding.** A looped student reuses the same pruned matrix R times. One-shot 2:4 error, which roughly doubles perplexity at 7B, may compound across passes, so 2:4 should be applied *during* stage-A teacher-matching distillation, not after it. This is untested for loops.
- **Fit for small edge models.** Expected per-pass gains on Orin are about 1.1–1.4× end-to-end for the GEMM-heavy parts. Batch-1 LLM decode is GEMV and bandwidth-bound; there 2:4 cuts weight bytes only when the kernel reads the compressed format. The saving is about 44% in fp16 (2 of 4 values kept, plus 2-bit indices) and about 37% in INT8. Prefill and per-frame vision run large GEMMs, where sparse tensor cores help most.
- **Low-rank FFN.** Low-rank is the simplest cut, d·r·2 instead of d·d_ff·3 parameters, and runs anywhere, including WebGPU. Monarch reports it underperforms. Pass-specific LoRA deltas on a shared full-rank block (Relaxed Recursive, Zamba2) are a different use: they *add* a little compute to recover per-pass specialisation, rather than saving compute.

### Gaps
- No measured 2:4 numbers on Jetson Orin, Orin Nano or Thor for transformers. I also found no 2:4 support on phone NPUs or WebGPU.
- No inference-time (rather than training-time) Monarch or BTT speedups, and none on edge devices.
- No study of 2:4 or Monarch inside weight-shared or looped blocks.

---

## 4. Cheaper attention in loops: local/sliding-window passes, linear-attention/SSM hybrids, cross-pass KV sharing (MoR, CLA, YOCO)

### Takeaway
Three approaches carry over well to loops:
1. **KV sharing across passes or layers.** CLA halves KV with under 1% perplexity change. MoR's recursive sharing cuts KV memory to 1/N_r at a small accuracy cost that is larger under expert-choice routing. Sharing also drops K/V projection FLOPs on the reusing passes.
2. **Local or sliding-window attention on most passes.** Gemma 3 uses 5 local layers to 1 global with a 1024-token window and sees little perplexity effect. Universal YOCO (Apr 2026) loops only cheap sliding-window layers and computes the global KV cache once.
3. **SSM backbones with a few shared attention blocks.** Zamba and Zamba2 use this design, and looped Mamba has been tested (Jul 2026).

Quality costs are small in every reported case, but nearly all of the evidence is from training from scratch.

### Cited Findings
**Cross-pass / cross-layer KV sharing**
- **MoR caching modes** (arXiv 2507.10524) [S] — [arXiv HTML v1](https://arxiv.org/html/2507.10524v1), [PDF](https://arxiv.org/pdf/2507.10524):
  - *Recursion-wise caching*: KV memory and cache I/O are about (N_r+1)/(2N_r) of vanilla. This is the more accurate mode.
  - *Recursive KV sharing*: the first recursion's KV is reused at all later depths, so KV memory is about 1/N_r, but **cache I/O stays at the full baseline level**. It "slightly reduces performance" compared with independent caching, causes "quite large performance degradation" under expert-choice routing, and benefits token-choice routing.
  - A secondary summary gives token-choice against expert-choice accuracy of 40.0% versus 42.6% at 3 recursions. [2nd] — [Turing Post](https://www.turingpost.com/p/mixtureofrecursions)
- **CLA** (Brandon et al., arXiv 2405.12981, May 2024; NeurIPS 2024). Adjacent layers share KV projections. **CLA2 on top of MQA halves the KV cache again, with at worst <1% perplexity change, and sometimes improves it**, at 1B and 3B trained from scratch. CLA3 and beyond give slightly worse tradeoffs than CLA2. [S] — [arXiv 2405.12981](https://arxiv.org/abs/2405.12981v1), [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9e23d020c18e4c40d81c6a0fc7a46f68-Abstract.html)
- **YOCO** (Sun et al., arXiv 2405.05254, May 2024; NeurIPS 2024). An efficient-attention self-decoder produces one global KV cache, which the cross-decoder layers reuse. Prefill can exit after the self-decoder, so **at least half the prefill compute is skipped**. At 512K context, prefill drops from about 180 s to under 6 s. The headline figure lists 6.4×, 30.3× and 9.6× for memory, throughput and prefill latency, but the mapping of numbers to metrics was unclear in the excerpt. [S] — [arXiv 2405.05254](https://arxiv.org/pdf/2405.05254)
  - Quality: YOCO-3B (1T tokens) averages 0.634 on 8 harness tasks, against 0.619 for OpenLLaMA-3B-v2. [P] — [YOCO README](https://github.com/microsoft/unilm/tree/master/YOCO)
- **Universal YOCO / YOCO-U** (Sun, Dong et al., arXiv 2604.01220, Apr 2026). Recursion runs **only over the shallow efficient-attention layers** (the "Universal Self-Decoder"). The global KV cache is built once and stays constant however many loops run; only the local window caches grow. Sliding-window attention is the default, and RetNet, Mamba and gated DeltaNet "perform about the same" in this hybrid. The authors claim it stays competitive on general and long-context benchmarks. [S] — [arXiv 2604.01220](https://arxiv.org/abs/2604.01220), [HTML](https://arxiv.org/html/2604.01220v1); [2nd] — [Liner review](https://liner.com/review/universal-yoco-for-efficient-depth-scaling)

**Local / sliding-window attention on most layers or passes**
- **Gemma 3** (Mar 2025) uses **5 local : 1 global** layers with a **1024-token** window, against Gemma 2's 1:1 with 4096. Its ablation found "little effect on perplexity". [S] — [Gemma 3 report](https://arxiv.org/pdf/2503.19786); [2nd] — [Vizuara](https://vizuara.substack.com/p/technical-deep-dive-into-gemma3-next). A secondary KV-overhead figure ("60% to <15%") cites a 1:3 ratio that contradicts the 5:1 design, so treat it as unreliable.

**SSM or linear-attention hybrids with shared or looped blocks**
- **Zamba / Zamba2** (Zyphra; arXiv 2411.15242, Nov 2024). A Mamba (Zamba2: Mamba2) backbone with **shared attention+MLP blocks applied repeatedly**. Zamba1 has one shared block, and Zamba2 alternates two shared blocks, with **non-shared per-depth LoRAs** on the MLP and/or attention. The authors say this means fewer attention layers at inference. Zamba2-1.2B uses a single shared block, "justified by a higher FLOP count at a given parameter budget at small scale". RoPE was added to the shared blocks, and the original embeddings are concatenated into the shared-block input. [S] — [Zamba2 report](https://arxiv.org/html/2411.15242v1), [Zamba2-1.2B card](https://huggingface.co/Zyphra/Zamba2-1.2B)
- **Looped Mamba and looped hybrid Mamba-Transformer** (Yu, Kojima, Matsuo, Iwasawa; arXiv 2607.10110, Jul 2026). These loop a shared Mamba-2 or hybrid block. They beat parameter-matched non-looped baselines on synthetic tasks and stay competitive downstream with far fewer distinct parameters, but **"deeper non-looped models retain an advantage in validation perplexity under strict isoFLOPs"**. They also adapt Ouro's exit gate. [S] — [alphaXiv 2607.10110](https://www.alphaxiv.org/abs/2607.10110v1), [arXiv PDF](https://arxiv.org/pdf/2607.10110)
- **Sliced Recursive Transformer** (SReT, vision, 2021). "Sliced group self-attention across recursive layers" cuts cost by **10–30% without sacrificing performance**. [S] — [arXiv 2111.05297](https://arxiv.org/abs/2111.05297)
- **Untied attention per pass:** Foil found per-pass attention, with shared experts, gives more balanced routing and lower loss (see §1). [S] — [arXiv 2609.35751](https://arxiv.org/abs/2609.35751)

### Inferences
- **K/V projection FLOPs saved by cross-pass sharing** (my arithmetic). With MHA, K and V projections are 4d² of the 24d² block FLOPs, so a pass that reuses another pass's KV saves about **1/6 of its linear FLOPs**, plus KV-cache writes. With GQA at 1/4 KV heads the saving is only about 1/24. So KV sharing is mainly a memory and bandwidth win, not a FLOP win. Recursion-wise caching saves more I/O than recursive sharing, because sharing keeps full KV I/O.
- **Attention cost at long context.** Score and value FLOPs are 4·d·L per token per pass. At L=4096 and d=2048 that is about 34M against about 100M linear FLOPs, so attention is about 25% of a pass. A 1024-token window on all but one pass caps this at about 8M per local pass. For an A→B→A→B→A→B student, an attractive pattern is to make most passes local and keep one global pass (or one global KV computed once, as in YOCO-U).
- **Short-sequence vision.** Per-frame vision at, say, 196–1024 tokens is not attention-score-dominated. Linear attention helps mainly at high resolution (§7).

### Gaps
- MoR's exact perplexity and accuracy deltas for recursive KV sharing against recursion-wise caching (no table retrieved).
- No retrofitting result for KV sharing or sliding windows added *after* training to a distilled looped student, or the token budget it would need. All the evidence above trains from scratch.
- Universal YOCO's benchmark numbers and FLOP breakdown.

---

## 5. Mixture-of-Depths-like skipping of sublayers inside a pass ("attention drop")

### Takeaway
Attention sublayers are more redundant than MLPs in deep layers. Training-free attention drop on Mistral-7B gives **1.22× speedup with about 0.4–0.7 point losses (4 of 32 attention layers dropped), and 1.45× with 1.5–2.4 point losses (8 dropped)**. Mixture-of-Depths, which needs training, routes tokens around whole blocks with a static top-k and gives up to **50% faster sampling** at equal quality. Both carry over naturally to loops, for example skipping attention on some passes, but I found no loop-specific measurements.

### Cited Findings
- **LLM-Drop** ("What Matters in Transformers? Not All Attention Is Needed", arXiv 2406.15786, Jun 2024; now TMLR 2026 as "Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping"). Mistral-7B-v0.1 table [P] — [LLM-Drop README](https://github.com/CASE-Lab-UMD/LLM-Drop):

  | Setting | MMLU | GSM8K | ARC-c | HellaSwag | Speedup | KV cache |
  |---|---|---|---|---|---|---|
  | Dense | 64.2 | 37.8 | 60.1 | 83.3 | 1.00× | – |
  | Attention drop, 4 layers | 63.8 | 37.1 | 59.6 | 82.9 | **1.22×** | −12.5% |
  | Attention drop, 8 layers | 62.5 | 35.4 | 58.2 | 81.7 | **1.45×** | −25% |
  | MLP drop, 4 layers | 63.1 | 36.2 | 58.9 | 82.4 | 1.28× | – |
  | Block drop, 4 layers | 62.7 | 35.0 | 58.4 | 81.9 | 1.32× | −12.5% |
  | Attention drop, 4 layers, + AWQ-4bit | – | – | – | – | **2.14×** | −12.5% |

  - The README claims "up to 2.1× speedup, >40% KV reduction, retains >95–98%". The table itself only shows −25% KV.
  - Mechanism, per the README: deep-layer attention "collapses into static routing patterns", while MLPs keep doing knowledge retrieval. Dropping a whole block can cause representational collapse.
  - Train? The README describes importance scoring plus dropping, with no fine-tuning step mentioned, so it appears training-free.
- **Mixture-of-Depths** (Raposo et al., DeepMind, arXiv 2404.02258, Apr 2024). A per-block top-k router chooses which tokens enter self-attention and the MLP. Because k is fixed in advance, the computation graph is static. MoD matches baseline quality at equal training FLOPs and wall-clock, while using "a fraction of the FLOPs per forward pass", and is "upwards of 50% faster to step during post-training sampling". **Needs training.** [S] — [arXiv 2404.02258](https://arxiv.org/abs/2404.02258v1)
- **Sparse Growing Transformer** (arXiv 2603.23998, Mar 2026). Adds recurrence (looping) only to a small subset of high-entropy *attention heads* during training, as "progressive attention looping". [S] — [arXiv 2603.23998](https://arxiv.org/html/2603.23998v2)
- **SMELT's mechanism finding:** the second loop visit reduces the attention sink and redirects attention mass toward content tokens. The second visit's attention is doing different work, which argues *against* blindly dropping attention on later passes. [S] — [arXiv 2609.01343](https://arxiv.org/html/2609.01343v1)

### Inferences
- **Pass-level attention drop.** In an A→B→A→B→A→B student, an obvious cheap variant runs attention only on some passes (for example every other pass), running the FFN, or the FFN experts, on every pass. Per the cost model, each skipped attention saves about 1/3 of the linear FLOPs plus the score cost and one KV write. SMELT's and Foil's findings that later-pass attention is useful mean this should be tested per pass, not assumed.
- **MoD inside a loop.** MoD's static top-k (fixed capacity) is friendlier to TensorRT, Core ML and WebGPU than data-dependent halting, because tensor shapes are known ahead of time.

### Gaps
- MoD's exact capacity setting (commonly cited as 12.5%) and its exact FLOP reduction were not confirmed in the retrieved text.
- No study of attention-drop or MoD *per pass* in a weight-shared model.
- LLM-Drop's README table is the only table found, it is for Mistral-7B only, and the hardware behind the speedups is not stated.

---

## 6. Cheaply converting a dense looped student into a sparse/MoE one (MoEfication, LLaMA-MoE, sparse upcycling, ReLU-fication) and the training budget

### Takeaway
There are three cheap conversion routes, ordered by training cost:
1. **Training-free:** MoEfication-style expert splitting with a router, or TEAL/R-Sparse/WINA activation sparsity. These keep about 95% of performance at 10–30% FFN usage, or reach about 40–50% sparsity.
2. **Tens to hundreds of billions of tokens:** ReLU-fication to about 90% sparsity costs <150B tokens for TurboSparse, and 5B tokens is too little at 70B.
3. **Sparse upcycling:** beats dense at about 50% of the original dense pretraining cost, but later scaling-law work shows it loses to from-scratch MoE at large token budgets.

Relaxed Recursive Transformers show that converting dense to looped needs about 15–60B uptraining tokens at 1B scale, which is a useful anchor for a further looped-to-MoE conversion. I found no paper that does looped-to-looped-MoE upcycling.

### Cited Findings
- **MoEfication** (Zhang et al., arXiv 2110.01786; ACL Findings 2022). Partitions pretrained FFN neurons into experts, either by balanced K-Means parameter clustering or by METIS on a co-activation graph, then adds a similarity or MLP router. It uses **10–30% of FFN parameters and keeps >95% of original performance**, which is a **2× FLOP speedup with 25% of FFN parameters**. That is a FLOP figure, not wall-clock. Train? The routers need light training; the experts come from the existing weights. [P] — [MoEfication README](https://github.com/thunlp/MoEfication); [S] — [abstract](https://sotaverified.org/papers/moefication-conditional-computation-of)
- **LLaMA-MoE** (2023–2024). Partitions LLaMA-2-7B FFNs into experts, adds a top-K gate, then does continued pretraining on SlimPajama with Sheared-LLaMA data weights. [P] — [LLaMA-MoE README](https://github.com/pjlab-sys4nlp/llama-moe)

  | Model | Active / experts | Active params | Avg of 11 tasks |
  |---|---|---|---|
  | LLaMA-MoE-3.0B | 2/16 | 3.0B | 55.5 |
  | LLaMA-MoE-3.5B | 4/16 | 3.5B | 57.7 |
  | LLaMA-MoE-3.5B | 2/8 | 3.5B | 57.6 |
  | Sheared-LLaMA-2.7B | dense | 2.7B | 56.4 |
  | Open-LLaMA-3B-v2 | dense | 3B | 55.6 |

  The continued-pretraining token count was not in the README text I retrieved.
- **Sparse upcycling** (Komatsuzaki et al., ICLR 2023). Experts are copied from a dense checkpoint, including optimizer state, and only the router starts random. On T5 and ViT, upcycled models beat their dense counterparts on SuperGLUE and ImageNet **using about 50% of the initial dense pretraining sunk cost**, and beat MoEs trained from scratch on 100% of that budget. [S] — [ICLR 2023 (mlanthology)](https://mlanthology.org/iclr/2023/komatsuzaki2023iclr-sparse)
  - Caveat: Liew et al. (ICML 2025) find that above a token-budget curve, from-scratch MoE is more efficient. Muennighoff et al. (2024) report upcycling can slow convergence. [S] — [Scaling Laws for Upcycling MoE](https://arxiv.org/html/2502.03009v2)
- **ReLU-fication budgets.** ProSparse needs continued training, at an unretrieved token count. TurboSparse uses <150B tokens for about 90% FFN sparsity. ReLU-70B on 5B tokens shows a noticeable downgrade. [S] — [TurboSparse](https://arxiv.org/html/2406.05955v1); [P] — [PowerInfer README](https://github.com/SJTU-IPADS/PowerInfer)
- **Dense to looped conversion anchor: Relaxed Recursive Transformers** (Bae et al., arXiv 2410.20672; ICLR 2025). Gemma 2B was converted to a 1B recursive model.
  - After about 15B uptraining tokens it beats TinyLlama-1.1B and Pythia-1B; this figure is from a secondary thesis citing the paper.
  - With knowledge distillation and 60B tokens it is reported to approach full-size Gemma, which was pretrained on 3T tokens. [S] — [arXiv HTML v2](https://arxiv.org/html/2410.20672v2), [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html), [early-exit thesis](https://arxiv.org/pdf/2509.05915)
  - Average initialisation of the looped layers worked best with per-depth LoRA. The claimed 2–3× throughput of continuous depth-wise batching is analytical, not measured. [S] — same
- **Ouro-2.6B** was "upcycled" from Ouro-1.4B by repeating layers after the first training phase. This is depth upcycling of a looped model. [S] — [Ouro search summary](https://huggingface.co/ByteDance/Ouro-1.4B)
- **Mixture of LoRAs** for recursive models uses distillation-based initialisation and merges its LoRA experts into one adapter at inference. [S] — [arXiv 2512.12880](https://arxiv.org/html/2512.12880v2)

### Inferences
- **Cheapest route for an A/B looped student.**
  1. Distil the dense shared blocks as now (stage A).
  2. MoEfy each shared FFN by clustering neurons into, for example, 8–16 experts. Use **per-pass routers**, because routing diverges between passes, and keep the experts shared.
  3. Short continued distillation against the teacher's internals, so the stage-A losses are reused.

  Budget anchors from the sources: MoEfication router training is light, LLaMA-MoE needs continued pretraining, and ReLU-fication and upcycling take 10⁹–10¹¹ tokens. The scale should be smaller for edge-sized students, but I found no number for <1B looped models.
- **Activation-sparsity alternative.** If the student's FFN uses ReLU or ReLU², or is ReLU-fied during stage A, MoE becomes unnecessary for decode: Deja Vu/PowerInfer-style neuron sparsity gives the bandwidth win without router or load-balance machinery. This is simpler on WebGPU and phones.
- **Rank of per-pass LoRA.** Relaxed Recursive found low-rank LoRA could not capture the differences between fully distinct layers (step-wise init), which suggests per-pass adapters for a shared block should be initialised from averaged weights.

### Gaps
- No published looped (weight-shared) dense to looped-MoE upcycling or MoEfication result, and so no measured distillation budget for it.
- LLaMA-MoE and ProSparse token counts were not confirmed from primary text.
- No conversion results at the <500M scale typical of edge students.

---

## 7. Vision/video: efficient ViT block designs on edge (EfficientViT, MobileViT, FastViT, linear attention) and whether they work with weight sharing

### Takeaway
Efficient ViT blocks with ReLU linear attention, such as EfficientViT, give measured Jetson Orin latencies of **1.5–5.6 ms at batch 1** for ImageNet classifiers at 79–84% top-1, and **7–82 ms** for segmentation up to 1024×2048. FastViT runs at **0.8–4.6 ms on an iPhone 12 Pro**. These blocks were designed and measured *without* weight sharing. The evidence that ViTs tolerate looping is mixed:
- MiniViT compresses DeiT-B 9.7× (86M to 9M parameters) by sharing one block with small per-layer transforms plus distillation, but naive cross-layer sharing in ViTs is reported to be unstable.
- SReT's sliced group attention cuts recursion cost by 10–30%.
- 2026 papers (ViTRM, RD-ViT, MoR-Swin, Soft-MoR ViT, LoopMoEVR, LoopTrack) are pushing looped vision models, with few edge numbers.

### Cited Findings
- **EfficientViT** (Cai et al., MIT Han Lab; ICCV 2023). Multi-scale **ReLU linear attention** gives a global receptive field with hardware-friendly operations. It avoids softmax attention, which is quadratic in resolution, and large-kernel convolutions. Cityscapes latency is **up to 8.8× lower than SegFormer** (ICCV) and up to 13.9× in arXiv v6, measured on Jetson AGX Orin and A100 with TensorRT fp16. [S] — [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf), [arXiv v6](https://arxiv.org/html/2205.14756v6)
- **EfficientViT classification**, measured with TensorRT fp16 including data transfer, at batch 1 [P] — [efficientvit_cls README](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md):

  | Model | Resolution | Params | MACs | Top-1 | Jetson Nano | Jetson Orin |
  |---|---|---|---|---|---|---|
  | B1 | 224 | 9.1M | 0.52G | 79.39 | 24.8 ms | **1.48 ms** |
  | B2 | 224 | 24M | 1.6G | 82.10 | 50.6 ms | **2.63 ms** |
  | B3 | 224 | 49M | 4.0G | 83.47 | 101 ms | **4.36 ms** |
  | B3 | 288 | 49M | 6.5G | 84.15 | 141 ms | **5.63 ms** |

  For comparison, the large models on A100: L1 at 224 has 53M params, 5.3G MACs, 84.48 top-1 and 6207 img/s. EfficientNetV2-S at 384 has 22M params, 8.4G MACs, 83.9 top-1 and 2869 img/s.
- **EfficientViT segmentation** at batch 1 [P] — [efficientvit_seg README](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_seg/README.md):

  | Model | Dataset, resolution | Params | MACs | mIoU | Jetson Nano | Jetson Orin |
  |---|---|---|---|---|---|---|
  | B0 | Cityscapes 1024×2048 | 0.7M | 4.4G | 75.65 | 275 ms | 9.9 ms |
  | B1 | Cityscapes 1024×2048 | 4.8M | 25G | 80.55 | – | 24.3 ms |
  | B2 | Cityscapes 1024×2048 | 15M | 74G | 82.07 | – | 46.5 ms |
  | B3 | Cityscapes 1024×2048 | 40M | 179G | 83.02 | – | 81.8 ms |
  | L1 | Cityscapes 1024×2048 | 40M | 282G | 82.72 | – | 45.9 ms |
  | L1 | ADE20K 512×512 | 40M | 36G | 49.19 | – | 7.2 ms |
- EfficientViT-SAM measures **48.9× TensorRT speedup on A100 over SAM-ViT-H** at matched zero-shot quality. EfficientViT is also the backbone of Grounding DINO 1.5 Edge. [P] — [efficientvit README](https://github.com/mit-han-lab/efficientvit)
- **FastViT** (Apple, ICCV 2023), measured on an **iPhone 12 Pro** with Core ML via ModelBench [P] — [ml-fastvit README](https://github.com/apple/ml-fastvit):

  | Model | Top-1 | Distilled top-1 | Latency |
  |---|---|---|---|
  | T8 | 76.2 | 77.2 | 0.8 ms |
  | T12 | 79.3 | 80.3 | 1.2 ms |
  | SA12 | 80.9 | 81.9 | 1.6 ms |
  | SA24 | 82.7 | 83.4 | 2.6 ms |
  | SA36 | 83.6 | 84.2 | 3.5 ms |
  | MA36 | 83.9 | 84.6 | 4.6 ms |

  FastViT uses structural reparameterisation: the training-time branches fold into single convolutions for inference.
- **MiniViT** (Zhang et al., CVPR 2022). "Weight multiplexing": one block's weights are shared across consecutive layers, with small per-layer transforms plus weight distillation from a large ViT. It **compresses DeiT-B 9.7×, from 86M to 9M parameters, "without seriously compromising performance"**, and **Swin-B by 48% with +1.0% top-1**. [S] — [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html). Per a review, naive cross-layer weight sharing in ViTs "leads to training instability and significant performance degradation", unlike in NLP. [2nd] — [Liner review](https://liner.com/review/minivit-compressing-vision-transformers-with-weight-multiplexing)
- **SReT** (2021). Sliced recursion allows more than 100 shared layers at about 13–15M parameters. Sliced group self-attention across recursive layers **cuts cost by 10–30% without sacrificing performance**. [S] — [arXiv 2111.05297](https://arxiv.org/abs/2111.05297)
- **ViTRM** (arXiv 2603.19503, Mar 2026). A small weight-sharing recursive encoder matches ResNet and ViT baselines with **6–84× fewer parameters**. More recursion steps do not necessarily raise accuracy. [S] — [alphaXiv 2603.19503](https://www.alphaxiv.org/abs/2603.19503)
- **2025–2026 looped vision titles**, listed but not read in full: MoR–Swin and MoR-ST (MoR for efficient ViTs, 2026), Soft Mixture-of-Recursions for recursive ViTs (2607.00774), RD-ViT recurrent-depth segmentation (2605.03999), Investigating Single-Block Recurrence in ViTs (2605.10661), MergeOver post-training token merging for recursive ViTs (2608.13141), LoopTrack parameter-efficient tracking (2609.33306), LoopMoEVR loop-based MoE for UHD video restoration (2610.05109), and Loop-Mamba (2608.02346). [P, titles] — [looped-lm-survey](https://github.com/ArminAzizi98/looped-lm-survey)
- **LoopViT** (arXiv 2602.02156, Feb 2026) is a weight-tied recurrent core with dynamic exit for visual ARC reasoning, not edge classification. [S] — [arXiv 2602.02156](https://arxiv.org/html/2602.02156v1)

### Inferences
- **Weight sharing changes the economics of these blocks.** EfficientViT B-series and FastViT are hierarchical: their stages have different widths and resolutions, so they cannot loop one block across stages. Looping fits *within* a stage, for example repeating the stage-3 or stage-4 block. SReT and MiniViT are isotropic ViTs, where looping is natural.
- **Attention cost at high resolution.** A shared block at high resolution is dominated by attention; ReLU linear attention changes the per-pass cost from O(N²d) to O(Nd²), so it is the right attention for a looped per-frame model at high resolution. For about 196 tokens, softmax attention is not the bottleneck.
- **Video.** For per-frame video, cross-pass KV sharing (§4) has no direct analogue. Looped stages could reuse the *previous frame's* final-pass state as the initial state, which is a temporal-recurrence idea outside my slice. I found no source.
- **Hardware fit.** Jetson Orin numbers above are TensorRT fp16 dense. 2:4 (§3) and reparameterisation (FastViT) are the structural tricks with real edge toolchain support. MoE in vision on the edge is unproven: I found no Orin or phone latency for a vision MoE.

### Gaps
- MobileViT numbers were not retrieved.
- I found no paper that measures EfficientViT- or FastViT-style blocks *with* weight sharing on Jetson or phones.
- No weight-shared ViT with ReLU linear attention was found.
- No edge latency figures for any 2026 looped ViT (ViTRM, RD-ViT, MoR-Swin).
- I did not verify MiniViT's exact DeiT-B accuracy delta.
