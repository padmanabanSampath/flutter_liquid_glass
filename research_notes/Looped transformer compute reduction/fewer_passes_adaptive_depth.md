# Fewer passes: adaptive depth for looped (recurrent-depth, weight-shared) transformers

Source tags used below. **[P]** = primary: the authors' own GitHub README or released code, read in full. **[S]** = search excerpt: a web-search summary of the paper's own text. The paper itself (arXiv/OpenReview/HF/CVF) could not be reached from this environment (DNS blocked), so treat these numbers as needing a check against the PDF. **[2nd]** = secondary: a third-party review, or the huskydoge "Awesome-Loop-Models" catalog TL;DRs and daily briefings (GitHub, generated 2026-10-12), which paraphrase arXiv abstracts. Dates are arXiv first-posting dates unless a venue is given. "Trained-in" means the method needs training; "zero-shot" means it works on a frozen model.

## 1. Halting / adaptive computation (ACT, Universal Transformer halting, PonderNet) and how they behave in practice

### Takeaway
Classic learned halting works on toy and algorithmic tasks, but in practice it is fragile. ACT is acknowledged by its own author to be "quite sensitive" to the ponder-cost weight. PonderNet fixes some of that with a probabilistic halting distribution plus a KL-to-geometric prior, and uses about 6× fewer steps than UT+ACT on bAbI. Practitioners have largely dropped ACT from UTs, and several 2026 diagnostics find that jointly-learned halting gates give poorer stop signals than fixed-prior depth supervision.

### Cited Findings
- **ACT (Graves 2016, arXiv 1603.08983):** a deterministic, differentiable halting unit that learns a variable number of state transitions per input [S/2nd] — [Awesome-Loop-Models catalog](https://github.com/huskydoge/Awesome-Loop-Models); [ACT paper](https://arxiv.org/abs/1603.08983)
- Graves' own stated weakness: "the numerical weight assigned to the time cost must be hand-chosen, and the behaviour of the network is quite sensitive to its value." He also reports performance "dramatically improved" by ACT on synthetic parity/addition tasks [S] — [ACT paper](https://arxiv.org/abs/1603.08983)
- A 2021 survey calls ACT's "sensitivity to the ponder cost, which controls the trade-off between speed and accuracy" a key shortcoming [S] — [Practical survey on faster and lighter Transformers](https://arxiv.org/abs/2103.14636)
- **Universal Transformer (ICLR 2019):** a shared self-attention + transition block applied repeatedly over depth, "optionally with per-position adaptive halting" (ACT) [2nd] — [Awesome-Loop-Models catalog](https://github.com/huskydoge/Awesome-Loop-Models); [UT paper](https://arxiv.org/abs/1807.03819)
- A 2026 paper notes that ACT in UTs "has shown mixed empirical results" and that Csordás et al. (2021) implemented UTs without ACT, part of a broader trend of dropping it. It suggests some failures stem from initialization/optimization pathologies [S] — [Universal Transformers Need Memory (arXiv 2604.21999, Apr 2026)](https://arxiv.org/abs/2604.21999)
- A 2025 analysis of HRM says the original implementation trains an ACT halting module, but evaluation ignores it and always runs the maximum number of steps [S] — [HRM: Perspectives and Misconceptions (arXiv 2510.00355)](https://arxiv.org/abs/2510.00355)
- **PonderNet (Banino et al. 2021, ICML AutoML workshop):** probabilistic halting over a recurrent step function, with the KL to a geometric prior acting as the compute-cost control. The authors describe ACT as "notably unstable and sensitive to the choice of a hyper-parameter that trades-off accuracy and computation" [S] — [PonderNet paper](https://arxiv.org/abs/2107.05407)
- PonderNet on bAbI (20 tasks): 1,658 total steps vs 10,161 for UT (+ACT), with average error 0.15 vs 0.29; both solve all 20 tasks. That is about 6.1× fewer steps at lower error [S] — [PonderNet paper](https://arxiv.org/abs/2107.05407)
- A secondary summary reports that ACT failed on out-of-distribution parity extrapolation while PonderNet stayed near-perfect, and that PonderNet is robust to its prior whereas ACT is sensitive to its time penalty. This is unverified against the paper [S/2nd] — [PonderNet search excerpt](https://api.emergentmind.com/topics/pondernet)
- **Sparse UT (EMNLP 2023):** combines a UT shared block with sparse MoE plus *stick-breaking* halting to reduce compute while keeping compositional generalization [2nd] — [Awesome-Loop-Models catalog](https://github.com/huskydoge/Awesome-Loop-Models) (arXiv 2310.07096)
- **2026 diagnostics of learned halting:**
  - *Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts* (Jul 2026) finds that "fixed-prior depth supervision shapes more useful stopping signals than jointly learned halting gates," on synthetic tasks and on Ouro-1.4B/2.6B checkpoints [2nd] — [arXiv 2607.20519](https://arxiv.org/abs/2607.20519) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
  - *Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping* (Jun 2026) treats loop count as a stochastic training-time variable and studies RL-Halting. Learned stochastic stopping "often improves the accuracy-stability tradeoff, though it can also stabilize a suboptimal computation" (binary addition, Dyck-1, Unique Set, Copy) [2nd] — [arXiv 2606.29983](https://arxiv.org/abs/2606.29983) via [catalog briefings](https://github.com/huskydoge/Awesome-Loop-Models)
  - *Understanding Dynamic Compute Allocation in Recurrent Transformers* (ICML 2026, Feb) finds adaptive compute "can align with token complexity while failing to extrapolate to longer al[gorithmic inputs]" [2nd] — [arXiv 2602.08864](https://arxiv.org/abs/2602.08864) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- **A-ViT (CVPR 2022)** reuses the ACT formulation per token, with ACT ponder loss (`ponder_scale_token`), a fixed gate scale and bias for the halting neuron, and a KL "distributional prior" regularizer (`distr_prior_alpha`) to make it trainable. These are the knobs that had to be added to tame ACT [P] — [NVlabs/A-ViT README](https://github.com/NVlabs/A-ViT)

### Inferences
- For the user's distillation pipeline, plain ACT (a learned halting score with a hand-tuned ponder cost) is the riskiest option: it has a sensitive scalar to tune and is known to be unstable. If a learned halting distribution is wanted, the PonderNet- or Ouro-style form (per-step exit distribution, loss weighted by exit probability, a KL/entropy regularizer to a prior) is the better-behaved descendant. 2026 evidence (2607.20519) suggests training the stop head *after* or *separately from* the backbone, against a fixed depth prior or oracle labels, rather than end-to-end jointly.
- With only 3 loops (A→B→A→B→A→B), a "halting distribution" has only 3 (or 6, at block granularity) support points. Simple per-loop exit classifiers or confidence thresholds are nearly as expressive as ACT/PonderNet there, and much easier to tune.

### Gaps
- No primary-source numbers for ACT's ponder-cost sweep or for the UT paper's ACT ablation could be retrieved (arXiv blocked).
- No published measurement was found of ACT/PonderNet *wall-clock* savings on edge hardware.

## 2. Looped-model-specific exits: MoR routers, Ouro/LoopLM exit gates, Huginn zero-shot KL exit, Relaxed Recursive Transformers, 2025–2026 follow-ups

### Takeaway
Looped LMs now have three proven families of exit:
- **Trained token-level routers.** MoR (NeurIPS 2025) and T-LoopFormer (Sep 2026) match or beat vanilla models with 2× throughput or equal latency. TaH skips 94% of second iterations.
- **Trained exit gates.** In Ouro, about 2.5 average loops of 4 gives about 66% MMLU vs 67.35% at fixed 4 (about 37% fewer loops for about 1.4 points). Ragged KV (Oct 2026) gets up to 30% fewer FLOPs at full-depth quality.
- **Zero-shot convergence exits on models trained with random depth.** Huginn's KL/entropy criteria give about 38% fewer loops at equal quality on a 135M model, and up to 38% lower latency with a second-order criterion.

The key engineering finding of 2025–2026 is that savings only materialize if the KV cache and batching are redesigned. Ouro's released code still runs all 4 loops for every token.

### Cited Findings
**Mixture-of-Recursions (MoR; Bae et al., Jul 2025, NeurIPS 2025) — trained-in**
- Motivation (authors' README): early exit à la CALM has two bottlenecks:
  - a **missing-KV-cache problem**, where exited tokens lack deeper-layer KV that later tokens need and "approximating them often hurts performance";
  - **inefficient batching**, where exited tokens idle while others finish.

  FREE fixed the first with parallel decoding, and Recursive Transformers fixed the second with depth-wise batching. MoR unifies them with end-to-end-trained routers plus **recursion-wise KV caching** (only tokens active at a recursion depth store KV there) [P] — [MoR README](https://github.com/raymin0223/mixture_of_recursions)
- Claim: "up to 2× greater inference throughput compared to standard transformers at similar accuracy," with lower training FLOPs and memory [P] — [MoR README](https://github.com/raymin0223/mixture_of_recursions)
- MoR-4 reaches up to **2.06×** throughput (continuous depth-wise batching + early exit vs a vanilla baseline with continuous sequence batching, at max batch). One summary says 2.18× instead [S] — [MoR paper](https://arxiv.org/abs/2507.10524) (search excerpt)
- **Expert-choice vs token-choice:**
  - Expert-choice: each recursion depth picks its top-k tokens, giving a fixed compute budget. This breaks causality in autoregressive decoding, which MoR fixes with an auxiliary router used at inference and an auxiliary loss on the main router.
  - Token-choice: each token picks its own depth, which keeps causality but causes load imbalance.
  - At 3 recursions, token-choice reached **40.0%** average few-shot accuracy vs **42.6%** for expert-choice [S] — [MoR paper](https://arxiv.org/abs/2507.10524)
- At equal training compute (16.5e18 FLOPs): MoR expert-choice with Nr=2 has 167M params, **43.1%** few-shot accuracy and val NLL 2.7511, vs vanilla **42.3%**, using "nearly 50% fewer parameters". MoR expert-choice with Nr=3 has 118M params and 42.6% [S] — [MoR paper](https://arxiv.org/abs/2507.10524)
- KV variants: recursive KV *sharing* (reusing first-recursion KV) saves memory but "slightly reduces performance" compared with recursion-wise caching. The drop is "quite large" for expert-choice under a fixed-token setting, so token-choice may benefit more from sharing [S] — [MoR paper](https://arxiv.org/abs/2507.10524)
- Implementation: routers do top-k selection, gather tokens, run the shared block once, `scatter_add` back, and repeat per recursion. Token-choice needs padding for variable lengths [P] — [MoR README](https://github.com/raymin0223/mixture_of_recursions)
- Follow-ups:
  - **MoR-MLLM** (Sep 2026) applies MoR routers per visual/text token in a Phi-2 backbone and cuts inference TFLOPs **86.6%** vs LLaVA-1.5-7B while "staying competitive" on 9 VL benchmarks. This compares against a different, larger model [2nd] — [arXiv 2610.08830](https://arxiv.org/abs/2610.08830) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
  - **bioMoR** (Aug 2026) applies MoR to omics [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models)

**T-LoopFormer (Sep 2026, under review ICLR 2027) — trained-in**
- Token-level router with recursion-wise KV caches, so exited tokens are not recomputed. At the "24×" setting: perplexity **20.13/19.14** and 10-task average **44.20**, vs **20.17/19.22** and **43.8** for an untied 24-layer baseline, at **0.197 vs 0.202 ms/token** [2nd] — [arXiv 2609.15160](https://arxiv.org/abs/2609.15160) via [catalog briefing](https://github.com/huskydoge/Awesome-Loop-Models); code [P] [YuMingQian1234/T-LoopFormer](https://github.com/YuMingQian1234/T-LoopFormer)

**Think-at-Hard (TaH, Nov 2025, ICML) and TaH2 (Sep 2026) — trained-in, selective extra iterations**
- TaH: a lightweight decider plus "duo-causal attention" runs extra latent iterations only on hard tokens. It **skips 94% of second iterations** and beats fixed two-iteration baselines by **8–11%** and single-iteration Qwen3 by **4–5%** [P] — [thu-nics/TaH README](https://github.com/thu-nics/TaH)
- TaH training is two-stage:
  1. Fixed per-token iteration labels from a reference model's mismatch (oracle "hard" tokens), with LoRA on the extra iteration.
  2. The backbone is frozen and an MLP iteration decider is trained [P] — [thu-nics/TaH README](https://github.com/thu-nics/TaH)
- TaH2 jointly trains backbone, input updater and decider. Labels are generated online as "whether another iteration improves next-token cross-entropy" ("lookahead depth supervision"). It improves the AIME accuracy-compute slope by **53%** over the non-looped baseline and peak accuracy by about **3.4 pts** at matched test-time compute. Checkpoints: TaH2-1.7B max-2, with a threshold of 0.5 at serving [P] — [thu-nics/TaH README](https://github.com/thu-nics/TaH)
- The TaH2 paper notes that Ouro's backbone is trained only at full depth, causing a train–inference mismatch under early exit [S] — [arXiv 2609.35748](https://arxiv.org/abs/2609.35748)

**Ouro / LoopLM (ByteDance Seed et al., Oct 2025) — trained-in exit gate**
- 1.4B and 2.6B LoopLMs trained on 7.7T tokens, with an "entropy-regularized objective for learned depth allocation" [S/2nd] — [Ouro paper](https://arxiv.org/abs/2510.25741); [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- **Stage I** loss: the sum over steps of exit-probability × per-step LM loss, minus β × entropy of the exit distribution (uniform prior over exit steps). **Stage II** fine-tunes only the exit gates with a performance-improvement signal. Inference uses a deterministic "Q-exit": exit at the first step where the CDF of exit probabilities crosses threshold q [S/2nd] — [Ouro paper](https://arxiv.org/abs/2510.25741); [ouro-llm.github.io](https://ouro-llm.github.io/) (search excerpt); reimplementation [P] [rkstgr/LoopLM](https://github.com/rkstgr/LoopLM) ("all T steps contribute to training via a weighted sum, where weights come from the learned exit distribution")
- Ouro-1.4B MMLU by fixed loop count (project page): **51.6** (1 loop), **67.6** (2), **73.6** (3), **74.6** (4, trained depth). It drifts down to about **72.2** at 8 loops [S] — [ouro-llm.github.io](https://ouro-llm.github.io/) (search excerpt)
- Early-exit curve (project page text): a trained gate gives about **66% MMLU at 2.5 average loops** vs **67.35%** at fixed 4 loops. The trained gate beats an untrained gate, a hidden-state-difference threshold and fixed depth at every budget. **Conflict:** the 67.35 fixed-4 figure disagrees with 74.6 in the per-depth table, presumably a different eval setup [S] — [ouro-llm.github.io](https://ouro-llm.github.io/) (search excerpt)
- **The released Ouro code runs all four loops for every token**, because a later token reaching loop d needs loop-d K/V from every earlier token. Early exit therefore saves nothing as shipped [2nd/S] — [Enabling Dynamic Computation in Looped LMs (arXiv 2610.09013, Oct 2026)](https://arxiv.org/abs/2610.09013) via [catalog briefing](https://github.com/huskydoge/Awesome-Loop-Models)
- **Ragged KV** (2610.09013): later tokens read each earlier token's *exit-depth* KV entries instead of forcing all loops.
  - Out of the box it cuts **FLOPs and KV memory by up to 30%** at full-depth quality ("maintaining or slightly improving performance"), and ragged-aware training narrows the remaining gap.
  - It also shows Ouro's *per-token* entropy prior pushes all tokens to the same exit depth. Applying the KL to the *batch-average* exit distribution gives confident, difficulty-dependent exits (tested on 160M looped models pretrained on 20B FineWeb-Edu tokens) [2nd/S] — [arXiv 2610.09013](https://arxiv.org/abs/2610.09013) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models) and search excerpt
- *Closing the Loop* (Sep 2026) is a from-scratch recipe that cuts Ouro's 7.7T tokens to 310B using mid-training, LR warmup and **stronger exit-gate regularization**. A minimal recipe converts Qwen3-1.7B-Base into a looped model with one learned input-mixing scalar and a **smoothed exit loss** [2nd] — [arXiv 2610.00673](https://arxiv.org/abs/2610.00673) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- *Towards Looped Models Done Right Part II* (IFM, Oct 2026):
  - A depth prior learned from prediction feedback, plus an entropy term and orthogonal input injection.
  - "At 1.6B parameters the learned prior with a **3× smaller KV cache** matches the downstream average of fixed-depth training with the full cache."
  - Terminal KV sharing, and distilled prefill up to **1.79× faster** [2nd] — [catalog entry](https://github.com/huskydoge/Awesome-Loop-Models); code/paper links [P] [ifm-ai/xllm-loop](https://github.com/ifm-ai/xllm-loop) (arXiv 2610.06833)

**Huginn-0125 / recurrent depth (Geiping et al., Feb 2025, NeurIPS 2025) — zero-shot exits on a randomly-unrolled model**
- 3.5B params (prelude, recurrent core, coda), trained on 800B tokens with a *randomly sampled* recurrence count. Benchmarks are run with `mean_recurrence=32` [P/2nd] — [seal-rg/recurrent-pretraining README](https://github.com/seal-rg/recurrent-pretraining); [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- Zero-shot per-token exit: stop iterating when the KL between successive steps' output distributions is < **5×10⁻⁴**. On MMLU, the step-count histograms converge faster on high-school math than on logical fallacies or moral scenarios [S] — [Huginn paper](https://arxiv.org/abs/2502.05171)
- Released code implements six exit criteria with "auto" thresholds, and the model card notes the model "is not trained for" per-token adaptive compute [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py):

  | Criterion | Default threshold |
  |---|---|
  | `entropy-diff` | 1e-3 |
  | `latent-diff` (relative state change) | 0.03 |
  | `kl` | 1e-3 |
  | `minp-kl` | 1e-5 |
  | `argmax-stability` | 5 steps |
  | `cosine` | 1e-3 |
- MT-Bench: standard decoding about **5.63** vs about **5.56** with early exits. With a KV-cache budget of 4 (cache sharing), the score is **5.86**, i.e. no loss. The extracted text was garbled, so verify against the paper's appendix table [S] — [Huginn paper](https://arxiv.org/abs/2502.05171) (search excerpt)
- Code docstring warning: adaptive compute "should only ever be benchmarked with batch_size=1", because batched KV entries contain the deepest sequence's steps and "will be picked up by later compute steps, making early exits look better than they are" [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)

**Zero-shot exit follow-ups (2025–2026)**
- *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers* (Jul 2026), on a 135M model trained on FineWeb-Edu with random depth:
  - Mean successive-output KL falls from 3.9×10⁻¹ at loop 2 to 8.5×10⁻⁶ by loop 16. The median token settles by loop 6; a 10% tail keeps updating through the training-mean depth of 8.
  - A training-free convergence-exit (freeze a token once successive outputs stop changing) matches uniform depth-8 quality at **4.94 average loops (−38%)**.
  - A linear router trained on the same convergence labels "needs nearly full depth and saves nothing".
  - Caveat: the 38% is a FLOP proxy from a teacher-forced harness, not a realized speedup [S/2nd] — [arXiv 2607.14427](https://arxiv.org/abs/2607.14427); [Pith review](https://pith.science/paper/2607.14427)
- *Two-Scale Latent Dynamics for Recurrent-Depth Transformers* (Sep 2025, NeurIPS 2025 listing): loop updates become smaller and increasingly orthogonal. A **second-order** (step-size acceleration) exit criterion beats Huginn's KL exit and the first-order variant on performance, stability and time. It reduces latency **up to 38%** while preserving quality (alphaXiv summary, not confirmed in paper text) [S/2nd] — [arXiv 2509.23314](https://arxiv.org/abs/2509.23314); [alphaXiv overview](https://alphaxiv.org/overview/2509.23314v2). Huginn's released code has a related second-order criterion: `freeze_strategy="latent-acceleration-1"` (the norm of the change in successive state deltas, normalized by state norm, default `exit_t=0.03`) in its diffusion-style parallel sampler `generate_diffusion_style` [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)
- *Prediction Dynamics in Depth-Recurrent LMs* (Sep 2026): stable-prediction-direction analyses reduce the mean earliest qualifying recurrent depth by **22.5–34.4%** [2nd] — [arXiv 2609.21383](https://arxiv.org/abs/2609.21383) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- **Counter-evidence:** *Think Shallow, Solve Deep* (Aug 2026) reports that Huginn-3.5B's per-token latent states *do not settle*, and that freezing the answer does not track task difficulty. It classifies recurrent operators as settling, marginal or drifting and derives a depth-safety condition [S/2nd] — [arXiv 2608.18222](https://arxiv.org/abs/2608.18222). This conflicts with the convergence picture in 2607.14427, which used a different, small model.
- *Fixed-Point Reasoners (FPRM)* (Jun 2026) uses fixed-point convergence as an end-to-end halting signal (Sudoku, Maze, state tracking, ARC-AGI) [2nd] — [arXiv 2606.18206](https://arxiv.org/abs/2606.18206); code [P] [fprm](https://github.com/nilskiKonjIzDunava/fprm)
- *LoopViT* (Feb 2026) uses a parameter-free entropy-based "predictive crystallization" exit. Loop-ViT-Small (3.8M) reaches **60.1%** on ARC-AGI-1 vs 54.5% for an 18M VARC. The exit "significantly improv[es] the accuracy-FLOPs Pareto frontier" (no number in README) [P] — [WenjieShu/LoopViT](https://github.com/WenjieShu/LoopViT)

**Relaxed Recursive Transformers (Bae et al., Oct 2024, ICLR 2025)**
- Pretrained LLMs are converted into recursive models with depth-specific LoRA. Continuous **depth-wise batching** plus early exit gives "2–3×" throughput in a *theoretical / oracle-exit* analysis: a token exits at the earliest loop whose prediction matches the final loop's. Up to **2.66×** generation speedup vs vanilla counterparts in that simulation. The authors state that practical exit algorithms are out of scope [S/2nd] — [RRT paper](https://arxiv.org/abs/2410.20672); [catalog](https://github.com/huskydoge/Awesome-Loop-Models) ("simulated oracle-exit ... not a demonstrated deployment speedup")

**Serving and system follow-ups that make exits pay off**
- **Continuous Depth Batching (CDB)** (Aug 2026) schedules loop iterations, boundary stages (prelude/coda) and one-step-ahead exit decisions separately. On Ouro-1.4B and Huginn-3.5B it realizes **up to 99% of the theoretical adaptive-depth speed-up**, with **1.5–1.9×** offline throughput and **45–90%** lower normalized latency under dynamic load [2nd] — [arXiv 2608.09444](https://arxiv.org/abs/2608.09444) via [catalog briefing](https://github.com/huskydoge/Awesome-Loop-Models). The README example on Huginn uses `max_recurrent_steps=16, exit_threshold=0.28, kv_policy="single"` (a shared KV with an early-exit lookahead gate), greedy decoding only [P] — [kschwethelm/continuous-depth-batching](https://github.com/kschwethelm/continuous-depth-batching)
- **FlashLoop** (Sep 2026, training-free) uses "token-sparse updates" that reuse converged token states across loops (an implicit per-token exit), plus loop-aware sparse attention and KV-residual quantization. Results: up to **1.64×** end-to-end speedup and **6×** KV reduction. On Ouro-2.6B R4 at 8K prefill: 214.4 → **121.8 TFLOPs** (−43%) and 16.9 → 6.9 GiB [P] — [Superone77/FlashLoop](https://github.com/Superone77/FlashLoop)
- Self-speculative uses of shallow loops (they save latency, not passes):
  - **WaveFront Decoding:** 2.42× (Ouro), 3.54× (Huginn), 4.81× with KV sharing.
  - **DAS-Wave:** 4.00–6.96×. Giving shallow drafts full-depth *values* raises draft–target agreement from 47.2% to 66.9% on Huginn at draft depth 1.
  - **Diffusion-forcing parallel sampler:** up to 5× on 3.5B looped LMs without retraining.

  [2nd] — [arXiv 2609.23033](https://arxiv.org/abs/2609.23033), [arXiv 2609.34538](https://arxiv.org/abs/2609.34538), [arXiv 2510.14961](https://arxiv.org/abs/2510.14961) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
- Other 2026 adaptive-loop LMs:
  - **AdaPonderLM** (Mar 2026): token-wise halting gates plus KV reuse under a fixed compute budget.
  - **RecurTrace** (Sep 2026): oracle-distilled adaptive halting on a weight-tied Qwen3 block, 0.6B–8B.
  - **LoopUS** (May 2026): converts a pretrained LLM into encoder / looped block / decoder with a confidence head for early exit.
  - **Looped LMs for tool calling** (Aug 2026): "adaptive stopping uses fewer iterations".

  No savings numbers found for these [2nd/P] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models); [Thrillcrazyer/LoopUS](https://github.com/Thrillcrazyer/LoopUS)

### Inferences
- For a looped *student distilled from a full teacher*, the most transferable recipe is a combination of three pieces:
  1. Deep supervision at every loop boundary: KD loss of each loop's readout against teacher logits, weighted by an exit distribution or a fixed prior. This is the Ouro Stage-I/LoopLM idea and makes intermediate loops usable.
  2. A cheap exit head trained afterwards with lookahead labels. TaH2 uses "does one more loop reduce CE"; in distillation, "does one more loop reduce KL to teacher" is the natural analogue.
  3. Ragged-KV or MoR recursion-wise KV, so exited tokens never need deeper KV.
- Token-choice routing (or a threshold exit) is the right choice for autoregressive on-device decode. Expert-choice's fixed top-k needs future tokens, and MoR needed an auxiliary router to fix this.
- Evidence suggests 25–40% fewer loops at roughly iso-quality on LMs trained to tolerate variable depth:
  - 38% at equal quality (135M, zero-shot convergence exit);
  - up to 30% at full-depth quality (Ragged KV on Ouro);
  - 22.5–34.4% (prediction-dynamics analysis);
  - about 37% fewer loops for −1.35 MMLU points (Ouro gate).

  With only 3 loops the granularity is coarse. Expect an average of about 2.1–2.4 loops (20–30% saved) unless exits are also allowed between the A and B blocks (6 half-steps).
- Shipping an early-exit head is not enough. If KV for skipped loops is still computed (as in Ouro's released code), the FLOP saving is zero.

### Gaps
- The exact Ouro threshold sweep (q vs average loops vs accuracy on several tasks) and the Stage-II loss details were not retrievable from the primary PDF. The MMLU numbers come from project-page search excerpts and contain an internal inconsistency (74.6 vs 67.35 at fixed 4 loops).
- MoR's full KV-sharing table (throughput and accuracy per KV policy) was not retrieved.
- No paper found measures adaptive-loop exits on phones, browsers or Jetson.

## 3. Early exit in standard transformers (CALM, LayerSkip, DeeBERT/PABEE, SkipDecode, FREE) and the KV-cache problem for later tokens

### Takeaway
Standard-transformer early exit reports 1.5–3× (CALM up to 3×; LayerSkip 1.8–2.2×; SkipDecode 2–5×; PABEE/DeeBERT about 1.5–2× on classification). Every autoregressive method has to solve the KV problem: a token that skips layers or loops has no deep K/V for later tokens. The solutions map directly onto loops:
- state copying (CALM): cheap but degrades quality;
- computing missing KV in parallel later (FREE / self-speculation);
- monotone exit schedules (SkipDecode);
- reading the earlier token's exit-depth KV (Ragged KV, Huginn's cache "lookup strategies");
- recursion-wise KV (MoR);
- a single shared KV (Huginn KV sharing, MoR recursive sharing, "Shared Memory" looped LMs).

Weight-sharing makes the KV problem *easier* than in standard transformers, because all loops produce K/V in the same space.

### Cited Findings
- **CALM** (Schuster et al., NeurIPS 2022): T5 1.1 with 8 encoder and 8 decoder layers, an exit head on every layer sharing the output embedding. It names three challenges:
  1. which confidence measure to use;
  2. tying sequence-level quality constraints to per-token exits;
  3. "attending back to missing hidden representations due to early exits in previous tokens".

  It reports a potential speedup of **up to 3×** while keeping performance high [S] — [CALM paper](https://arxiv.org/abs/2207.07061); [Google Research blog](https://blog.research.google/2022/12/accelerating-text-generation-with.html)
- A 2024 follow-up compared softmax, classifier and state-based exits on T5-v1.1 small/base. Softmax at threshold 0.85 exits at about 53.8% of layers on small, with the best ROUGE-L agreement among the early-exit variants. It handles skipped layers by filling the KV cache of the remaining layers before the iteration terminates, gets up to **1.25×** over vLLM, and concludes hidden-state similarity may be the best exit signal for pure efficiency [S] — [arXiv 2407.20272](https://arxiv.org/abs/2407.20272)
- **FREE** (Bae et al., EMNLP 2023) cites "performance degradation caused by a state copying mechanism or numerous exit paths" and threshold sensitivity in prior work. It uses a shallow-deep module plus *synchronized parallel decoding*: exited tokens are stacked and their deep KV is computed in parallel with the next token's deep pass. Thresholds come from a Beta-mixture adaptive estimator [S] — [FREE paper](https://arxiv.org/abs/2310.05424); also cited in [MoR README](https://github.com/raymin0223/mixture_of_recursions) [P]
- **SkipDecode** (Del Corro et al. 2023):
  - One exit point per batch per sequence position, so batching works.
  - Exit points **decrease monotonically** along the sequence, so earlier tokens always have the KV that later (shallower) tokens need and nothing is recomputed. Later tokens are "easier due to more context".
  - It *skips lower/middle layers* rather than exiting, so upper layers always run.
  - **2–5×** speedups with "negligible regression" on OPT-1.3B/6.7B [S] — [SkipDecode paper](https://arxiv.org/abs/2307.02628)
  - Critique: it "lacks stable and concrete control over the speedup ratio" and leaves an imbalanced subnetwork [S] — [Unified Layer Skipping, arXiv 2404.06954](https://arxiv.org/abs/2404.06954)
- **LayerSkip** (Elhoushi et al., ACL 2024): training adds *layer dropout* (low rates in early layers, higher in later) plus an **early-exit loss with one shared exit head**. Inference uses self-speculative decoding: exit early to draft, then verify with the remaining layers. Speedups: **2.16×** on CNN/DM summarization, **1.82×** on coding, **2.0×** on TOPv2 [S] — [LayerSkip paper](https://arxiv.org/abs/2404.16710)
- LayerSkip README notes [P] — [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip):
  - Speedup is observed only with LayerSkip-trained checkpoints (Llama2 7B/13B/70B, CodeLlama 7B/34B, Llama3 8B, Llama3.2 1B) and self-speculation.
  - Classification and multiple-choice tasks get no speedup.
  - The training recipe is in torchtune (Dec 2024) and TRL (Mar 2025).
- **PABEE** (Zhou et al., NeurIPS 2020) attaches an internal classifier per layer and exits when predictions are unchanged for *patience* consecutive layers. Code includes ALBERT fine-tuning scripts [P] — [JetRunner/PABEE](https://github.com/JetRunner/PABEE)
  - An independent re-implementation on ALBERT-base: PABEE at **2.0×** layer-speedup scores 81.6 macro-average vs 81.9 for full ALBERT-base and 81.7 for DeeALBERT at 2.0× [S] — [Early Exiting with Ensemble Internal Classifiers, arXiv 2105.13792](https://arxiv.org/abs/2105.13792)
  - DeeBERT reports *wall-clock* while PABEE reports *layer-count* speedup, so they are not directly comparable [S] — [arXiv 2310.19152](https://arxiv.org/abs/2310.19152)
- **DeeBERT** (Xin et al., ACL 2020): entropy-threshold early exit with "highway" classifiers. Training is two-stage: fine-tune the backbone, then the highway exits [P] — [castorini/DeeBERT](https://github.com/castorini/DeeBERT)
- **Missing KV in looped models: concrete mechanisms**
  - Huginn's `HuginnDynamicCache` stores KV per (recurrent-step, position), "held uncoalesced because certain recurrent steps may be missing for some sequence ids if using per-token adaptive compute". A `lookup_strategy` decides what to read. Options include "latest-m4" (latest available step) and "compress-s4" (KV sharing modulo 4 steps); the self-speculative path defaults to `"latest-m4-compress-s4"` [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)
  - With KV-cache sharing at budget 4, MT-Bench shows no loss (5.86) [S] — [Huginn paper](https://arxiv.org/abs/2502.05171). A later paper reports Huginn's cache can drop to 4 or 1 shared slots with similar accuracy, and that storing only the most recent recurrence's KV per token keeps cache memory at the level of a fixed-depth transformer [S] — [Parallel samplers, arXiv 2510.14961](https://arxiv.org/abs/2510.14961)
  - Ragged KV (later tokens read each earlier token's exit-depth KV) on Ouro gives up to 30% fewer FLOPs and less KV at full-depth quality [2nd] — [arXiv 2610.09013](https://arxiv.org/abs/2610.09013)
  - "Shared Memory in Looped Transformers" (Oct 2026) pretrains looped LMs where **only the first recursion writes KV** and later recursions read it, with a short local window. At 150M–1B, a gated-DeltaNet hybrid at 5 recursions lowers FineWeb-Edu perplexity by 1.12–1.82 vs a same-size transformer with **76–79% less context memory**. This makes per-token early exit KV-free by construction [2nd] — [arXiv 2610.02383](https://arxiv.org/abs/2610.02383) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
  - "Shallow Queries, Mature Values": drafts reading full-depth *values* of the verified prefix helps (agreement 47.2% → 66.9%); replacing *keys* alone does not [2nd] — [arXiv 2609.34538](https://arxiv.org/abs/2609.34538)

### Inferences
- **ALBERT is a fully weight-shared ("looped") BERT, so PABEE/DeeALBERT on ALBERT-base (about 2× at −0.3 macro points) is effectively early exit on a looped encoder.** This is good evidence that encoder-style looped models (per-frame vision) tolerate about 2× depth cuts for classification-like outputs. (That ALBERT shares parameters across all layers is general background, not from the retrieved sources.)
- The cleanest fit for a token-by-token looped student on device is: train with a first-recursion-only KV or KV-sharing policy (Huginn/MoR sharing, 2610.02383), so exiting never leaves holes. If per-loop KV is kept, use Ragged KV (read the exit-depth KV) and train with it so the mismatch is learned. Avoid CALM-style state copying, which FREE identifies as a source of degradation.
- LayerSkip's recipe translates almost one-to-one: "layer dropout" becomes random loop truncation during training, and the "shared exit head" is natural because a looped model already reuses one LM head after every loop.

### Gaps
- CALM's exact speedup per task and its state-propagation ablation, plus SkipDecode's per-task speedup tables, were not retrieved (arXiv blocked).
- The original PABEE (1.57×-type) and DeeBERT (time-saving) headline numbers were not verified. Only a third-party ALBERT re-implementation number was found.

## 4. Warm-starting the loop (previous token/frame latent) and fixed-point acceleration

### Takeaway
Warm starts are the cheapest pass-saver for *video/streaming* models:
- **StreamDEQ:** initializing each frame's fixed-point solve from the previous frame's representation gives **>3× faster** inference at on-par accuracy.
- **DEQ-Flow:** fixed-point reuse is reported to accelerate inference by about **1.6×**.
- **Huginn token-level warm start ("continuous CoT"):** saves only about **1–2 of ~32** steps on average.
- **Jacobian regularization:** cuts solver iterations by 2–3× (17→6 NFE on CIFAR-10; 27→14 on ImageNet).
- **Anderson/Broyden:** helps true DEQs but costs extra memory and compute per step, and has no reported use on small fixed-count looped transformers.

### Cited Findings
- **Huginn warm start / continuous CoT** (zero-shot): starting each new token's recurrence from the previous token's final latent instead of a random state reduces the average number of steps to convergence by **about 1–2** [S] — [Huginn paper §6.3](https://arxiv.org/abs/2502.05171) (search excerpt). Code exposes it as `continuous_compute=True` ("warm-start state / continuous CoT"), combinable with any exit `criterion` [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)
- Huginn's default state initialization is random (`initialize_state(..., scale=init_scale)`), and the model was trained that way. Warm start is therefore off-distribution and zero-shot [P] — [raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)
- **StreamDEQ** (Ertenli, Akbas, Cinbis, ECCV 2022) seeds each video frame's MDEQ solve with the previous frame's representation, exploiting slow change in video. It recovers "near-optimal representations in a few frames' time" and stays up to date. On **video semantic segmentation and video object detection** it matches standard MDEQ accuracy while running **more than 3× faster** [S] — [StreamDEQ arXiv 2204.13492](https://arxiv.org/abs/2204.13492); [ECCV 2022 page](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2054_ECCV_2022_paper.php)
- **DEQ-Flow** (Bai, Geng, Savani, Kolter, CVPR 2022) recasts RAFT-style recurrent flow refinement as a fixed-point layer. "DEQ flows converge faster, require less memory, are often more accurate." With the same model size, DEQ-Flow-H-1x vs RAFT-H-1x scores Sintel clean 1.27 vs 1.36 and KITTI F1-all 12.95 vs 16.16 [P] — [locuslab/deq-flow README](https://github.com/locuslab/deq-flow). The paper's fixed-point reuse across frames reportedly accelerates inference "by a factor of about 1.6×" [S] — [DEQ-Flow paper](https://arxiv.org/abs/2204.08442) (one search excerpt; a second search could not re-locate the figure, so treat as unverified)
- **LDEQ / "Recurrence without Recurrence"** (Micaelli et al., NVIDIA, CVPR 2023):
  - A landmark DEQ gets 3.92 NME on WFLW with O(1) training memory in the number of recurrent modules.
  - RwR reformulates inference as constrained optimization, emulating recurrence across video frames to fix flicker without temporal training data. On WFLW-V hard: −10% NME and −13% flicker (NMF) vs the best prior.
  - This is a *stability* use of the previous frame, not a reported pass saving [S] — [LDEQ arXiv 2304.00600](https://arxiv.org/abs/2304.00600)
- **Fixed-point acceleration (DEQ literature):**
  - Jacobian regularization lets multiscale DEQs converge in about **6** solver iterations vs **17** unregularized on CIFAR-10, and **27 → 14** on ImageNet (about 2× acceleration), at minimal added cost [S] — [Bai et al. ICML 2021, arXiv 2106.14342](https://arxiv.org/abs/2106.14342)
  - A physics-informed DEQ study: Broyden improved accuracy but made epochs **>6× longer** than Anderson; plain fixed-point iteration was about 3× faster than Anderson but less reliable [S] — [arXiv 2406.03472](https://arxiv.org/abs/2406.03472)
- **Solver/state reuse in 2026 looped generators:**
  - *Fixed-Point Masked Generative Modeling* (May 2026) uses consistency training plus *solver-state reuse* across denoising steps to adapt depth [2nd] — [arXiv 2605.31215](https://arxiv.org/abs/2605.31215)
  - *VIAR* (ICML 2026) replaces the middle VAR stack with one implicit equilibrium layer and exposes a per-scale iteration knob, "reducing high-resolution redundancy, latency, and KV cache usage while preserving quality" [P] — [mobiushy/VIAR](https://github.com/mobiushy/VIAR)
- **RD-VLA** (Feb 2026), a weight-tied recurrent action head: KL-divergence stopping (threshold 0.001, max 32 iterations) or cosine stopping (0.999). LIBERO average **92.5% adaptive vs 93.0% fixed-12**. The README does not report the average iteration count [P] — [rd-vla/rd-vla](https://github.com/rd-vla/rd-vla)
- **Counterpoint for LMs:** Huginn's latent states may *not settle* per token, so convergence-based warm-start reasoning may not hold for large LMs [S] — [arXiv 2608.18222](https://arxiv.org/abs/2608.18222). *A Tilted Bowl Is Not a Slippery Slope* (Sep 2026) finds that rounding error accumulates only in loops that never settle. A controller that stops when the halting head fires, then runs a few loops in 8-bit, raises 20 compressed Sudoku-Extreme models from 61.1% to 75.8% "at under a third of the weight traffic of fixed-depth inference" [2nd] — [arXiv 2609.39277](https://arxiv.org/abs/2609.39277) via [catalog](https://github.com/huskydoge/Awesome-Loop-Models)

### Inferences
- For the user's per-frame vision/video loops, a **warm start from the previous frame's final loop state plus a convergence exit** is the highest-leverage zero-retraining idea. StreamDEQ (>3×) and DEQ-Flow (about 1.6×) show the effect on genuinely iterative per-frame models.
- The catch is that a looped *transformer* distilled to run exactly 3 fixed passes from a fixed initial state (the input embedding) is not a fixed-point solver. Feeding it the previous frame's latent changes its input distribution. To make warm start safe:
  - train with randomly mixed initial states (fresh embedding vs the previous frame's loop-k state, plus noise);
  - add input re-injection each loop (Huginn/Ouro style), so the current frame's evidence always enters;
  - use a small fixed-point or "terminal" objective so loops settle (cf. 2608.18222's terminal fixed-point objective).
- For token-by-token LMs, expect warm start to save only a few percent of loops (about 1–2 of 32 in Huginn). With a 3× loop, that is unlikely to save a whole pass.
- Anderson/Broyden acceleration needs a history of states (memory per step) and only pays off when many (10+) iterations are needed. For a 3-loop edge model it is unlikely to save a full pass and adds bandwidth. Jacobian/spectral regularization during training (DEQ; STARS for looped LMs, see §5) is the cheaper way to make loops converge faster.

### Gaps
- No primary numbers were found for passes saved by warm-starting a looped *transformer* (rather than a DEQ or RAFT-style refiner) on video.
- The DEQ-Flow "1.6×" fixed-point-reuse figure could not be confirmed from the paper.
- RAFT's own warm-start ablation on Sintel was not retrieved.

## 5. Loop-count distillation, robustness to running a different number of loops than trained, and depth conditioning

### Takeaway
Looped models are *not* automatically robust to truncation. Ouro (trained at 4 loops, with deep supervision) loses about 1 MMLU point at 3 loops and about 7 at 2. Models trained at a single fixed depth, such as RL policies or Ouro's backbone, are reliable only near that depth.

Making fewer loops match more loops is a solved training problem, with several 2025–2026 recipes:
- random loop counts during training (Huginn, LoopICL, stochastic stopping);
- intra-loop self-distillation, where the full-depth output teaches intermediate loops (ELT);
- shortcut-consistency with time/step-size conditioning (LoopFormer, ICLR 2026);
- adjacent-depth distillation (FlexLoop);
- FiLM conditioning on normalized depth (LARM).

### Cited Findings
- **Truncation and extrapolation behaviour (no special training):** Ouro-1.4B MMLU at 1/2/3/4 loops is **51.6 / 67.6 / 73.6 / 74.6** (trained at 4), then degrades slowly to about **72.2** at 8 loops [S] — [ouro-llm.github.io](https://ouro-llm.github.io/)
- *What Makes Recurrence Effective in Looped LMs?* (Sep 2026): extra loops beyond the training horizon can improve reasoning while *degrading knowledge*. "Channel-wise history-state injection with timestep conditioning is more robust than initial-state injection across inference budgets" [2nd] — [arXiv 2609.36636](https://arxiv.org/abs/2609.36636)
- *STARS* (May 2026, ICML) diagnoses collapse of looped LMs at larger recurrence depths and combines **Jacobian spectral-radius regularization with random loop sampling** [2nd] — [arXiv 2605.26733](https://arxiv.org/abs/2605.26733)
- *Beyond Depth Truncation* (Sep 2026) argues naive inference-time truncation conflates block-application, distinct-depth and readout-calibration effects. Its Depth Control Protocol finds the controlled distinct-depth contribution is −2.8% in its main experiment [2nd] — [arXiv 2609.19934](https://arxiv.org/abs/2609.19934)
- **Random-depth training (train-short/test-long, and vice versa):**
  - Huginn trains with a sampled recurrence count, which is what makes zero-shot depth choice and exits work [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
  - IFM's `DepthControlledHuginn` trains "at a fixed, sampled (Poisson-lognormal) or learned depth with truncated backpropagation" [P] — [ifm-ai/xllm-loop](https://github.com/ifm-ai/xllm-loop)
  - LoopICL (tabular, Sep 2026) is trained with varying loop counts plus a learned exit gate. At equal FLOPs it is competitive with TabICLv2 using about 90% fewer parameters, with a user-selectable cost/accuracy trade-off [2nd] — [arXiv 2609.36108](https://arxiv.org/abs/2609.36108)
- **LoopFormer** (Jeddi, Ciccone, Taati, ICLR 2026, Feb 2026):
  - Trained on variable-length loop trajectories with a **shortcut-consistency** objective, conditioning each loop on its current time and step size ("shortcut modulation"). Shorter runs stay useful and longer runs keep refining.
  - A single 1B checkpoint serves user-specified budgets, with perplexity and zero-shot reasoning improving with budget.
  - "Naive early-exit baselines stagnate across steps" and "adaptive early-exit looped models exhibit representational collapse".

  No exact numbers retrieved [S] — [LoopFormer arXiv 2602.11451](https://arxiv.org/abs/2602.11451); [ICLR 2026 slides](https://iclr.cc/media/iclr-2026/Slides/10009450.pdf); code (NanoGPT fork, RMSNorm) [P] [armenjeddi/loopformer](https://github.com/armenjeddi/loopformer)
- **ELT — Elastic Looped Transformers for visual generation** (Apr 2026; listed for image and video synthesis):
  - **Intra-Loop Self Distillation (ILSD)** uses the full-loop configuration as teacher for a randomly sampled intermediate-loop exit (student) within the same training step.
  - This gives "any-time inference" from one training run, with high-quality images from 4× fewer parameters (review).
  - No FID-vs-loops numbers retrieved [S/2nd] — [ELT arXiv 2604.09168](https://arxiv.org/abs/2604.09168); [Moonlight review](https://www.themoonlight.io/review/elt-elastic-looped-transformers-for-visual-generation)
- **FlexLoop** (Sep 2026): pretrained looped RL policies "specialize to their trained recurrent depth". Post-training with **adjacent-depth policy distillation** makes shallower depths reliable and enables per-state stopping via recurrent-depth consistency (30 long-horizon environments) [2nd] — [arXiv 2609.34488](https://arxiv.org/abs/2609.34488)
- **RecursiveVLM** (Feb 2026) uses a "monotonic recursion loss" so extra recursion only refines, enabling on-demand compute [2nd] — [arXiv 2602.09080](https://arxiv.org/abs/2602.09080)
- **LiFT — Loop Flow Transformers** (Oct 2026): a depth-conditioned shared core in a DiT, supervising every loop with a point on the straight path from the initial estimate to the flow-matching target [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models) (arXiv 2610.05538)
- **Depth conditioning (FiLM on normalized depth):** LARM (Idiap, Jun 2026, ASR):
  - FiLM on d̄(k) = (k−1)/(K−1), plus a periodic "supervision-clock" embedding W_c[(k−1) mod c].
  - Delayed soft-posterior feedback reinjected into the state.
  - CTC loss only at sparse checkpoints {c, 2c, …, K}.

  A 7.7M LARM (d=384, K=12) beats a 28.9M 16-block unshared encoder on LibriSpeech-100h (greedy test-clean 11.34 vs 14.43 WER). Intermediate (unsupervised) loops can be **non-monotonic** (100h), while checkpoint loops improve monotonically [P] — [idiap/LARM README](https://github.com/idiap/LARM)
- Relaxed Recursive Transformers use **depth-specific LoRA deltas** on a shared block, a form of depth conditioning [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models); [RRT](https://arxiv.org/abs/2410.20672)
- ProgResViT (adaptive ViT that reuses blocks across rounds) uses "Progress-Conditioned Soft Gating", conditioning on round, block and resolution, and ships KD-trained presets (teacher distillation into the multi-round model) [P] — [ds-kiel/ProgResViT](https://github.com/ds-kiel/ProgResViT)

### Inferences
- A concrete recipe for the user's distillation pipeline:
  1. Sample the number of loops per batch (e.g. 1–4 for a nominal 3×, or a per-block cut at 6 half-steps).
  2. Apply teacher KD at the readout after every sampled loop, weighted toward the final loop.
  3. Add ILSD-style self-distillation of intermediate loops toward the student's own full-depth output, which is cheaper than querying the teacher at every depth.
  4. Condition each block on normalized depth via FiLM/LoRA-per-depth, so the same weights know "how much budget is left".
  5. Optionally add a spectral or "terminal fixed-point" regularizer, so loops beyond 3 do not drift (useful for warm-start video).

  This turns loop count into a runtime knob and is a prerequisite for any per-token or per-frame exit.
- If the model will always run exactly 3 loops at deployment, depth conditioning is harmless. If loops are skipped, the FiLM input must be the *actual* depth, not the planned budget. LoopFormer conditions on both time and step size, which suggests feeding both.
- Ouro's 3-of-4-loop result (about −1 MMLU point with no budget-specific training) hints that a well deep-supervised student can lose one loop at around 1% quality cost even *without* adaptive exits. That is a static 25–33% saving to test first.

### Gaps
- No numeric LoopFormer, ELT or FlexLoop results (perplexity/FID/return vs loops) were retrievable.
- No published study was found on distilling a *full non-looped teacher* into a looped student with variable-loop KD specifically. The recipe above is an inference combining the cited mechanisms.

## 6. Vision / video per-frame models: adaptive depth per frame or per region, and temporal reuse

### Takeaway
For images, token- or round-adaptive depth saves about **30–40% FLOPs at about 0.3% top-1 loss**:
- A-ViT: +62%/+38% throughput on DeiT-T/S;
- Edge-RecViT: −30.5% FLOPs, which explicitly combines recursion (full weight-sharing) with per-token depth;
- ProgResViT: up to −31.1% compute.

For video, exploiting temporal redundancy gives more (2–4×):
- Eventful Transformers: recompute only changed tokens;
- StreamDEQ: previous-frame warm start;
- FrameExit: 1.3–2.1× fewer GFLOPs by processing fewer frames.

### Cited Findings
- **A-ViT** (Yin et al., NVIDIA, CVPR 2022): ACT-style per-token halting. One neuron of each block's last dense layer is reused as the halting score, so there are no new parameters.
  - Reported throughput: **+62% for DeiT-Tiny and +38% for DeiT-Small with only 0.3% accuracy drop** [S] — [A-ViT paper](https://arxiv.org/abs/2112.07658)
  - Released weights: A-ViT-T 71.4% top-1 at 0.8 GFLOPs (5M params); A-ViT-S 78.8% at 3.6 GFLOPs (22M) [P] — [NVlabs/A-ViT](https://github.com/NVlabs/A-ViT)
  - A review cites a **39% FLOPs reduction** with average token depth 7.23 layers (of 12) [2nd] — [liner review](https://liner.com/review/avit-adaptive-tokens-for-efficient-vision-transformer)
  - Training starts from pretrained DeiT for 100 epochs on 4× V100 [P] — [NVlabs/A-ViT](https://github.com/NVlabs/A-ViT)
- **Edge-RecViT** (CVPR 2026) combines a *fully shared recursive hidden layer* with an edge-aware per-token ranker that assigns depth. Structurally informative edge tokens recurse deeper, while low-information tokens exit early. On ImageNet-1K it is within **0.3% top-1** of DeiT with **−30.5% FLOPs** (35.1 → 24.39 GFLOPs as written). At Base level, parameters drop from 86M to 23.21M with higher accuracy than ViT-Base. The router is a MoR-style `token_choice_router.py` [P] — [RicardoLee510520/Edge-RecViT](https://github.com/RicardoLee510520/Edge-RecViT)
- **ProgResViT** (Sep 2026) reuses the same ViT blocks across confidence-routed rounds (low-res/narrow first, then higher-res/wider), stopping when confident. Results:
  - up to **31.1%** lower average ImageNet compute [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models)
  - KD variant: 84.9% top-1 at 16.152 GMACs average (160→384 schedule); 83.8% at 6.267 GMACs (192→240) [P] — [ds-kiel/ProgResViT](https://github.com/ds-kiel/ProgResViT)
- **Video temporal reuse:**
  - **Eventful Transformers** (Dutson, Li, Gupta, ICCV 2023) recompute only tokens that changed significantly since the last frame, convert existing ViTs "often without retraining", and allow a runtime-adjustable budget. **2–4×** compute savings with minor accuracy loss on ImageNet VID and EPIC-Kitchens 100. GPU timings (RTX 3090): ViTDet@1024 86.6 → 38.1 ms; ViViT@320 950 → 545 ms [S] — [Eventful Transformers arXiv 2308.13494](https://arxiv.org/abs/2308.13494)
  - **FrameExit** (Qualcomm, CVPR 2021) uses gated conditional early exit over frames, so easy videos use fewer frames: **1.3×** (ActivityNet1.3) and **2.1×** (mini-Kinetics) fewer GFLOPs at better-than-SOTA accuracy [S] — [FrameExit arXiv 2104.13400](https://arxiv.org/abs/2104.13400)
  - **StreamDEQ:** >3× faster video segmentation/detection via previous-frame warm start (see §4) [S] — [arXiv 2204.13492](https://arxiv.org/abs/2204.13492)
- **Looped vision/robotics with adaptive stopping (2026):**
  - **DeepJEPA** (Sep 2026) uses a learned continue head per imagined transition. It matches or beats the best fixed-depth planner while averaging **1.00–1.26 updates per transition**, spending extra updates at contact events [2nd] — [arXiv 2610.00368](https://arxiv.org/abs/2610.00368)
  - **CoRELoop** (audio deepfake, Sep 2026): adaptive halting averages **1.18 encoder passes** with EER 3.73% vs 3.74% for the fixed variant [2nd] — [arXiv 2609.19818](https://arxiv.org/abs/2609.19818) via [catalog briefing](https://github.com/huskydoge/Awesome-Loop-Models)
  - **RD-VLA:** adaptive KL stopping gives 92.5 vs 93.0 for fixed-12 on LIBERO [P] — [rd-vla](https://github.com/rd-vla/rd-vla)
  - **LoopViT** uses an entropy exit [P] — [LoopViT](https://github.com/WenjieShu/LoopViT)
  - **Looped World Models** (Jun 2026) scale latent depth per prediction step [2nd] — [arXiv 2606.18208](https://arxiv.org/abs/2606.18208)
- A catalogued 2026 entry, *GTR: Gated Token Recurrence for Efficient Dense Prediction* (arXiv 2609.26590), appears in the loop-model daily briefing candidates, but no details were retrieved [2nd] — [catalog](https://github.com/huskydoge/Awesome-Loop-Models)

### Inferences
- For per-frame looped vision models, there are three axes, ranked by expected payoff on edge hardware:
  1. **Temporal reuse.** Warm-start the loop from the previous frame and/or skip recomputing unchanged tokens; 2–4× in the literature.
  2. **Per-frame exit.** Run 1, 2 or 3 loops based on a frame-level confidence or state-change signal. This is shape-static, so it is friendly to NPUs, WebGPU and TensorRT.
  3. **Per-token/region exit** (A-ViT, Edge-RecViT, MoR-ViT): about 30–40% FLOPs.

  Per-token exits create dynamic token counts, which mobile NPUs, browser runtimes and TensorRT engines handle poorly. Gather/scatter overhead can eat the gain unless implemented as masked, fixed-capacity top-k (expert-choice per frame is fine for vision because there is no causality constraint).
- Expert-choice routing (fixed top-k tokens per loop) is attractive for vision specifically. It gives a static compute budget per frame, avoids MoR's causality issue (irrelevant for non-autoregressive frames), and maps to fixed shapes.

### Gaps
- No primary numbers were found for per-frame adaptive *loop counts* in a weight-shared video transformer (as opposed to frame-count exits, token pruning or DEQs).
- No edge (phone/Jetson/browser) latency measurements for A-ViT-style per-token halting were found.

## 7. Realistic FLOP savings at <1% quality loss, and the wall-clock caveat on edge

### Takeaway
Realistic, evidence-backed budgets at about <1% quality loss:
- **LLM decode:** about 20–40% fewer passes. Examples: 38% at equal quality with a convergence exit; up to 30% with Ragged KV on Ouro at full-depth quality; 22.5–34.4% earliest-depth reduction; about 1-of-4 loops dropped statically in Ouro for about −1 MMLU point. More (around 2×) is possible with trained token routers (MoR, T-LoopFormer, TaH's 94% skip) when the model is co-designed for it.
- **Images:** about 30–40% FLOPs at about 0.3% top-1.
- **Video:** 2–4× when temporal redundancy is exploited.

On wall-clock: savings appear only if skipped passes truly skip *both* compute and memory traffic, which means no KV back-fill and no batch stragglers. Several 2026 papers show the theoretical FLOP saving goes unrealized without KV and scheduling changes. Batch-1 edge decode is bandwidth-bound, so the real metric is bytes of weights plus KV read per token.

### Cited Findings
- FLOP-proxy vs realized speedup:
  - The 38% saving in 2607.14427 is "a FLOP proxy from a teacher-forced harness, not a realized speedup" [2nd] — [Pith review](https://pith.science/paper/2607.14427)
  - RRT's 2–3× is an oracle-exit simulation [S/2nd] — [RRT](https://arxiv.org/abs/2410.20672)
  - Ouro's released code runs all 4 loops, so early exit saves nothing until Ragged KV [2nd] — [arXiv 2610.09013](https://arxiv.org/abs/2610.09013)
- Batching: early-exited tokens "end up waiting for others in the same batch" [P] — [MoR README](https://github.com/raymin0223/mixture_of_recursions). CDB recovers up to 99% of the theoretical adaptive-depth speedup (1.5–1.9× throughput) by scheduling loop iterations separately [2nd] — [arXiv 2608.09444](https://arxiv.org/abs/2608.09444)
- Measurement artefact: batched adaptive-compute benchmarks overstate early-exit quality, because deeper KV computed for other sequences leaks into the cache. Benchmark adaptive compute at batch_size=1 [P] — [Huginn raven_modeling_minimal.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)
- Metric mismatch in early-exit literature: DeeBERT reports wall-clock, PABEE reports layer-count speedup [S] — [arXiv 2310.19152](https://arxiv.org/abs/2310.19152). CALM's follow-up framework gets only **1.25×** over vLLM in practice [S] — [arXiv 2407.20272](https://arxiv.org/abs/2407.20272)
- Real measured end-to-end gains of exit-like mechanisms on looped LMs:
  - FlashLoop: up to **1.64×** end-to-end; Ouro-2.6B 8K prefill 214.4 → 121.8 TFLOPs [P] — [FlashLoop](https://github.com/Superone77/FlashLoop)
  - T-LoopFormer: 0.197 vs 0.202 ms/token vs an untied baseline at better perplexity [2nd] — [arXiv 2609.15160](https://arxiv.org/abs/2609.15160)
  - Eventful Transformers on video: 86.6 → 38.1 ms on GPU [S] — [arXiv 2308.13494](https://arxiv.org/abs/2308.13494)
- Weight traffic as the edge metric: stopping when a halting head fires, then running a few 8-bit loops, recovers 61.1% → 75.8% accuracy on compressed Sudoku models "at under a third of the weight traffic of fixed-depth inference" [2nd] — [arXiv 2609.39277](https://arxiv.org/abs/2609.39277)
- KV size dominates long-context decode in looped LMs:
  - Hybrid Latent Attention on Ouro shrinks the cache 10.7× and speeds decoding 2.5× at 1K and up to 7.4× at 16K context, keeping over 97% accuracy [2nd] — [arXiv 2610.07940](https://arxiv.org/abs/2610.07940)
  - FlashLoop gets 6× KV reduction [P] — [FlashLoop](https://github.com/Superone77/FlashLoop)

  These are the other researchers' slices, but they matter because skipping a pass also skips that pass's KV reads.

### Inferences
- **Bandwidth arithmetic for batch-1 edge decode (inference, not measured):**
  - Per token, a looped model reads the unique blocks' weights once *per pass*, unless they stay resident in on-chip SRAM/L2/SLC, plus each pass's KV.
  - If unique weights (e.g. two blocks A, B) are larger than cache, which is typical on phones and Jetson, each skipped pass saves about 1/N of weight traffic. The wall-clock saving then roughly tracks the pass saving, *as long as the exit does not force KV back-fill*.
  - If the unique weights fit in cache (small looped models), repeat passes are cheap in bandwidth and decode is closer to compute-bound. Pass skipping then saves compute but less latency than the FLOP ratio suggests.
  - In both cases, the per-pass KV read grows with context, so KV-sharing or first-recursion-only KV policies amplify the benefit of exits.
- **Per-token exit at batch 1 on device is the easy case.** There are no stragglers, and the only requirement is a KV policy that tolerates holes (shared KV, Ragged KV or recursion-wise KV). The exit check itself (an LM-head softmax for KL/entropy after each loop) costs a full vocab projection per loop. That can be a large fraction of a small model's per-pass compute: with a 50k-vocab head on a 1B model, consider a cheap state-change criterion (`latent-diff`, cosine) or a tiny trained gate instead of logit KL.
- **Browser/WebGPU and mobile NPUs favour static graphs.** Per-frame or per-sequence loop counts (1/2/3) are implementable as 3 pre-compiled graphs or a loop with a scalar exit. Per-token ragged depth in a batch, or per-region token subsets, need gather/scatter and dynamic shapes, which may erase the gain on those runtimes. The fixed-capacity expert-choice variant (MoR) is the static-shape-friendly option.
- **Expected ranges for the user at <1% loss, assuming the student is trained with per-loop deep supervision and random loop counts:**
  - token LLM: average 2.1–2.4 passes out of 3 (20–30% saved) from exits alone; a well-trained router may approach 2×-style savings only if the model was co-designed (MoR/T-LoopFormer);
  - per-frame vision: 30–40% saved via token/region depth, or 1.5–3× via previous-frame warm start plus a frame-level exit on slowly changing video.

  These are inferences from the cited ranges, not measured results for 3-loop students.

### Gaps
- No source measured early-exit wall-clock on phones, in browsers (WebGPU/WASM) or on Jetson for any looped model. All realized speedups found are on datacenter GPUs (H100/H200/RTX 3090).
- No source quantified the cost of the per-loop exit check (LM-head projection) as a fraction of per-pass FLOPs for small looped LMs.
- The interaction between pass skipping and weight-cache residency on specific edge SoCs is not covered by the retrieved literature. It is presumably covered by the hardware/precision researcher.
