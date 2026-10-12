# Loop-specific execution tricks for looped (weight-shared) transformers

Scope note for the report writer. These notes cover execution tricks that only work because the looped blocks reuse the same weights. arxiv.org, alphaxiv, huggingface.co, openreview and nvidia.com were blocked by the proxy in this session (confirmed with `__agentproxy/status`), so paper numbers come from two kinds of source: search-engine excerpts of the arXiv pages, or primary GitHub pages (READMEs, PRs, issues) fetched directly.

Source tags used below:
- **[P-repo]**: primary GitHub page, read directly.
- **[SE]**: search-engine excerpt of the primary paper or doc. The full PDF was not read.
- **[2nd]**: secondary summary (blog, wiki, aggregator).

Wherever a source says so, each number is marked **measured** (wall-clock on hardware) or **simulated/theoretical**. "Today" is 2026-10-12.

Reference geometries used throughout:
- **Ouro:** 1.4B and 2.6B, R=4 loops of the whole stack. 2.6B has 48 layers × 4 = 192 logical layers.
- **Huginn-0125 / Raven:** 3.5B, recurrent core run ~32 times.

---

## Q1. Depth-wise batching (continuous depth-wise / depth batching; Relaxed Recursive Transformers, Mixture-of-Recursions, 2026 serving engines)

### Takeaway
Measured gains exist only for multi-request serving on datacenter GPUs, and they depend on early exit.
- **Continuous Depth Batching (Aug 2026):** 1.5–1.9× offline throughput on Ouro-1.4B and Huginn-3.5B, about 99% of the theoretical adaptive-depth maximum.
- **MoR (Jul 2025):** up to 2.06× at maximum batch size on 360M models.
- **Relaxed Recursive Transformers (2024/ICLR 2025):** 2–3× is only a simulated/oracle estimate.

None of these targets a single-user edge device. There, the only sources of "other depth" work to batch are speculative tokens (Q2), parallel samplers (Q3), or independent video frames. That last option is untested in the literature.

### Cited Findings
**Relaxed Recursive Transformers (Bae et al., Google DeepMind; arXiv 2410.20672, Oct 2024; ICLR 2025)**
- Introduces continuous depth-wise batching. Because every loop iteration uses the same block, tokens or requests at different loop depths can go into one batch. Combined with early exit, the authors claim "up to 2-3×" throughput over a vanilla Transformer of the same architecture. This is explicitly theoretical/hypothetical (Section 3.8, details in an appendix). [SE] — [arXiv 2410.20672](https://arxiv.org/pdf/2410.20672); [ICLR 2025 slides](https://www.iclr.cc/media/iclr-2025/Slides/29334.pdf)
- The specific figures are **simulated with an oracle exit**: a token exits at the earliest loop whose prediction matches the final loop's prediction.
  - Recursive models: "up to 2.66× speedup in generation" vs vanilla counterparts.
  - Recursive Gemma: "nearly 4×" throughput vs vanilla Pythia.
  - The combination is claimed to beat continuous sequence-wise batching (vLLM/Orca-style).
  - Throughput trades off against LoRA rank, i.e. against how relaxed the sharing is.
  - [SE] — [arXiv 2410.20672](https://arxiv.org/pdf/2410.20672); [themoonlight review, 2nd](https://www.themoonlight.io/en/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layer-wise-lora)
- Setup of the throughput analysis, as reported by a related early-exit thesis that describes the same kind of analysis: per-token timings measured on an A100 with a 16 GB memory constraint, prefix 64, decode length 256, counting only time inside Transformer blocks. It is unconfirmed that this is identical to the RRT appendix. [SE] — [arXiv 2509.05915](https://arxiv.org/pdf/2509.05915)

**Mixture-of-Recursions (Bae et al.; arXiv 2507.10524, Jul 2025)**
- Throughput setup:
  - 360M-scale models with recursion depths 2/3/4, trained at 16.5e18 FLOPs.
  - Two batch settings: fixed batch 32, and "B = Max", i.e. 32 scaled by the ratio of vanilla to MoR maximum batch sizes.
  - Output lengths ~N(256), no input prefix.
  - [SE] — [arXiv 2507.10524](https://arxiv.org/html/2507.10524)
- Results:
  - MoR-4 reaches **up to 2.06× throughput at B = Max**.
  - Every MoR variant beats a vanilla baseline that already uses continuous sequence-wise batching, at both batch settings.
  - Deeper recursion means more early exits, a smaller KV cache, and higher throughput, at a "slight performance degradation". [SE] — [arXiv 2507.10524](https://arxiv.org/pdf/2507.10524)
  - The GPU type and the per-variant numbers at batch 32 were not visible in any excerpt.
- The official repo claims "up to 2× greater inference throughput" but contains no throughput scripts, batch sizes or inference hardware. Training hardware was 4× H100/A100. Its KV handling is simplified: "computed all sequences against a shared KV cache for simplicity", with masking. [P-repo] — [raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)
- VentureBeat repeats the 2.06× figure and notes exact real-deployment numbers are hard to give. [2nd] — [VentureBeat](https://venturebeat.com/ai/mixture-of-recursions-delivers-2x-faster-inference-heres-how-to-implement-it)

**Continuous Depth Batching, CDB (Schwethelm, Rückert, Kaissis; arXiv 2608.09444, Aug 2026)**
- Schedules at the granularity of individual loop iterations, so requests at different recurrence depths share one batch, with refill.
- On Ouro 1.4B and Huginn 3.5B it reaches **~99% of the theoretical maximum speedup from adaptive depth**:
  - **1.5–1.9× higher offline throughput**.
  - **45–90% lower normalized latency under dynamic load**.
- Fully looped architectures suit it best, because large non-looped layers add scheduling overhead. [SE] — [arXiv 2608.09444](https://arxiv.org/abs/2608.09444)
- Code status:
  - It is a research implementation, not a serving engine, and supports greedy decoding only.
  - Example config: early-exit gate `exit_threshold=0.28`, `max_recurrent_steps=16`, `kv_policy="single"` (shared KV).
  - Supports Ouro 1.4B/2.6B and Huginn-0125.
  - A serving engine, "loop-sglang", is in development. [P-repo] — [ZH-Haoran/continuous-depth-batching](https://github.com/ZH-Haoran/continuous-depth-batching)

**vllm-rlt (2026)**
- A standalone engine for recurrent-loop models built around "loop-level continuous batching": requests at different recurrence depths share a batch. Features:
  - Refill, chunked prefill, Ouro's early-exit gate with per-request loop bounds.
  - Depth-aware paged KV cache (LAST-EXITED and SHARED layouts).
  - CUDA Graph capture of the decode recurrent core.
  - Prefill/decode disaggregation.
- Only Ouro-1.4B is supported. [P-repo] — [yanbao1217/vllm-rlt](https://github.com/yanbao1217/vllm-rlt); same project under [ThinkFlowLab/vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt/issues/43)
- The README cites single-GPU benchmarks (Ouro-1.4B BF16, B300, 1,024-token inputs, concurrency 1/32/128) but gives no absolute values in the text. Its 4-GPU PD-disaggregation results (2P2D at 4 req/s: 99.0% of baseline throughput, −13.3% TTFT, −53.5% ITL) are not loop-specific. [P-repo] — [yanbao1217/vllm-rlt](https://github.com/yanbao1217/vllm-rlt)

### Inferences
- The structural reason depth-wise batching works is that a token at loop 3 and a token at loop 1 multiply against the same matrices. Their GEMVs can therefore be stacked into one GEMM that reads the weights once. In an unshared model, those tokens would need different layers' weights.
- In memory-bound decode, the extra rows are nearly free until the GEMM becomes compute-bound. That is why all reported gains come from filling batches that early exit has left ragged.
- All the gains above come from early exit plus multi-request load. For a single user at batch 1 with no early exit, depth-wise batching has nothing to batch. The single-user analogues are:
  1. Speculative tokens at different depths. This is exactly what WaveFront/LoopSpec do; see Q2.
  2. Parallel sampling (diffusion forcing, Q3).
  3. Independent video frames: frame f at loop r alongside frame f+1 at loop r−1. This is exact for per-frame models, because frames do not depend on each other's outputs. It trades higher per-frame latency (R calls) for throughput, and it helps only when per-frame compute is not already GEMM-saturated.
  4. Batching across prompt tokens adds nothing, because prefill already processes all prompt tokens in one call per pass.
- The 2–3× RRT number should not be cited as measured. The best measured figures are CDB's 1.5–1.9× and MoR's up to 2.06×, both on datacenter GPUs.

### Gaps
- RRT appendix specifics (batch sizes, hardware, per-model numbers) could not be read; arXiv was blocked.
- The GPU used for MoR throughput and its batch-32 per-variant numbers were not visible.
- No study found applies depth-wise batching on phones, browsers or Jetson, or across video frames.
- CDB's GPU type and batch sizes were not visible in excerpts.

---

## Q2. Self-speculative decoding with looped models (draft = fewer loops, verify = full loops)

### Takeaway
2025–2026 produced a cluster of training-free, lossless self-speculation methods built specifically for looped models.
- Speedups over autoregressive decoding: **2.4–3.4× on Ouro** (R=4) and **3.5–7× on Huginn/Raven** (R≈32), mostly measured at batch 1 on a single datacenter or workstation GPU.
- The loop-specific advantage over LayerSkip/Kangaroo-style self-speculation on unshared models:
  - The draft costs no extra parameters.
  - Draft and verification run the same block, so they can be fused into one batched call at different depths (WaveFront, LoopSpec, DAS).
- A plain synchronous draft-then-verify is capped at about D/d (e.g. 2× for d=2, D=4). The pipelined schedules exist to break that cap.

### Cited Findings
**Loop-specific methods (2026)**
- **WaveFront Decoding, WFD (arXiv 2609.23033, Sep 2026).**
  - Training-free. Intermediate recurrence outputs are the drafts. Because weights are shared, "token states at different positions and recurrence depths" are processed in one batched recurrent-block call (a wavefront).
  - Speedups over AR, across 6 Spec-Bench categories:
    - **2.42× on Ouro-2.6B**.
    - **3.54× on Huginn-3.5B**.
    - **4.81× on Huginn with cross-recurrence KV sharing**, which "reduces wavefront KV traffic".
  - Beats draft-then-verify (DtV) in both cases. GSM8K/MATH-500 accuracy is comparable to AR.
  - Draft depth T_d=1 for Ouro (of 4) and T_d=4 for Huginn (of 32).
  - Hardware: a single RTX A6000 at batch size 1. A limitation appears "at long context lengths".
  - [SE] — [arXiv 2609.23033](https://arxiv.org/html/2609.23033)
  - The paper's DtV baseline: draft a sequence autoregressively at depth T_d<T, then resume the retained states in a batch from T_d to T. [SE] — [arXiv 2609.23033](https://arxiv.org/html/2609.23033)
- **LoopSpec (Cho, Cui, Kim, Han, Han; arXiv 2609.17184, Sep 2026).**
  - Training-free and lossless under greedy and sampling.
  - Drafts the next token from an early recurrent state and immediately starts computing the token after it, conditioned on that draft. The speculative continuation and the original computation use the same blocks, so they are processed together in one batch even at different recurrent depths.
  - An optional second proposal comes from a deeper depth.
  - Results:
    - "**Up to 6.83×**" inference speedup across looped models on reasoning and coding.
    - Up to 1.7× over DFlash, a trained drafter baseline.
    - [SE] — [arXiv 2609.17184](https://arxiv.org/html/2609.17184)
    - Up to 3.36× for the Ouro family (R=4). [SE] — [arXiv 2609.17184](https://arxiv.org/pdf/2609.17184)
    - Ouro-1.4B at about 2.6–3.2× across several benchmarks (search-engine summary). [SE] — [arXiv 2609.17184](https://arxiv.org/pdf/2609.17184)
  - Code status:
    - Built on SGLang (`sglang_recurrent`); NVIDIA GPU only. A Docker image is provided (CUDA 13.3).
    - Models: Ouro-1.4B/2.6B (+Thinking) and Raven-Llama-3.2, Raven-OLMo-2, Raven-TinyLlama.
    - Proposal depths: Ouro first=1, second=2; Raven first=2, second=8. The first depth must divide the total depth.
    - The README does not attribute the 6.83× figure to a model and reports no acceptance rates or batch sizes. [P-repo] — [kaist-flexml-lab/loopspec](https://github.com/kaist-flexml-lab/loopspec), [yichuan-w/loopspec](https://github.com/yichuan-w/loopspec)
- **Depth-Asynchronous Self-Speculation, DAS ("Shallow Queries, Mature Values"; Li, Su et al.; arXiv 2609.34538, Sep 2026).**
  - Observation: queries and keys reach their final-depth form earlier than values.
  - The "Mature-V" primitive lets shallow-depth draft queries read full-depth prefix values from cache with no extra passes.
  - DAS-Wave adds carried parallel refinement, progressive block growth and an independent full-depth verifier.
  - Reports **4.00–6.96× mean throughput speedup** over full-depth AR across four recurrent checkpoints (math/code). [SE] — [arXiv 2609.34538](https://arxiv.org/html/2609.34538v1)

**Runtime implementation: vllm-rlt PR #44 (merged 2026-09-23)**
- Synchronous self-speculation for Ouro:
  - Draft depth d=2, target D=4. Verification reuses the first d loops' hidden states and KV and runs only loops d+1..D for K candidates plus a bonus position.
  - Greedy, or rejection sampling (min(1,p/q)).
  - Rejected KV is rolled back.
  - Config: `SpeculativeConfig(num_speculative_tokens, draft_loops, target_loops)`. [P-repo] — [ThinkFlowLab/vllm-rlt PR #44](https://github.com/ThinkFlowLab/vllm-rlt/pull/44)
- **Measured** setup: B300, Ouro-1.4B BF16, FA4, eager, greedy, 128 in / 64 out, median of 3 runs. These are earlier prototype runs, not rerun on the final PR head.

  | K | Ideal (cost model) | Concurrency 1: tok/s (speedup) | Concurrency 8: tok/s (speedup) |
  |---|---|---|---|
  | native | 1.00× | 29.08 (1.00×) | 225.85 (1.00×) |
  | 1 | 1.33× | 38.78 (1.33×) | 299.32 (1.32×) |
  | 2 | 1.50× | 43.78 (1.51×) | 319.19 (1.41×) |
  | 4 | 1.67× | 48.33 (1.66×) | 356.20 (1.58×) |
  | 8 | 1.80× | 52.46 (1.80×) | 306.72 (1.36×) |

  - The idealized model, 2(K+1)/(K+2), assumes all candidates are accepted.
  - At concurrency 8, K=8 regresses below K=2 and K=4, cause unprofiled.
  - No acceptance rates were reported. [P-repo] — [PR #44](https://github.com/ThinkFlowLab/vllm-rlt/pull/44)
- Follow-ups in RFC #43 (Sep 23 – Oct 4, 2026). [P-repo] — [ThinkFlowLab/vllm-rlt issue #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43)
  - Async speculation merged (#66, fixed-depth greedy).
  - CUDA-Graph capture of sync speculation merged (#48).
  - Host syncs per round cut from 73–110 to 5 (#51): +2.5% at B=1, +5.3% at B=8.
  - Eager speculation on B200 MIG is launch-bound: ~1,400 kernels per pass, GPU only 13–16% busy.
  - A proposal-improvement prototype (#59): +5.8/+5.3/−1.1% (d1) and +4.1/+4.2/+6.5% (d2) at concurrency 1/4/8.
  - Loop-level pipelining (LoopSpec-style) is claimed, but there is no PR yet.

**Related, not speculation**
- **LoopCD (arXiv 2610.02185, Oct 2026):** loop-wise contrastive decoding. Huginn-0125 at 16 of 32 iterations with LoopCD outperforms its own full-depth unguided baseline, i.e. an accuracy-at-reduced-depth result. [SE] — [arXiv 2610.02185](https://arxiv.org/html/2610.02185v1)
- **Name collision:** "DLoop: Looped Speculative Decoding" (NAVER, arXiv 2610.07659, Oct 6, 2026) is *not* about looped models. It loops the drafter (EAGLE-3, DFlash, etc.) while confident and reports 5–41% wall-clock speedup. [SE] — [arXiv 2610.07659](https://arxiv.org/abs/2610.07659)

**Baselines on unshared models (for comparison)**
- **LayerSkip** (Elhoushi et al., ACL 2024, arXiv 2404.16710): early-exit draft, remaining layers verify, with draft and verify sharing compute/activations. Peak speedups (needs a training recipe):
  - 2.16× on CNN/DM.
  - 1.82× on coding.
  - 2.0× on TOPv2.
  - [SE] — [Meta AI](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding)
  - The HF-native variant "does not share the KV cache or execution between the draft and verification stages". Integrated into HF transformers in Nov 2024. [P-repo] — [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- **Draft & Verify** (arXiv 2309.08168): layer-skipping draft, training-free, max 1.73× in the abstract excerpt. [SE] — [arXiv 2309.08168](https://arxiv.org/html/2309.08168v2). The 1.99× figure in the task brief was not found.
- **Kangaroo** (Huawei; arXiv 2404.18911, NeurIPS 2024): shallow sub-network plus a 67M adapter, with a second early exit that stops drafting on low confidence. Up to 1.68× on Spec-Bench with 88.7% fewer extra parameters than Medusa-1 (591M). [SE] — [arXiv 2404.18911](https://arxiv.org/pdf/2404.18911)

### Inferences
- **Why looped models beat LayerSkip-style self-speculation:**
  - In LayerSkip, the draft (layers 1..E) and the verifier (layers E+1..L) are different weights, so drafting token t+1 and verifying token t cannot share a weight read.
  - In a looped model, every depth uses the same block, so WaveFront/LoopSpec put draft and verify positions into one GEMM.
  - That is the loop-specific gain on top of ordinary speculation, where verifying k tokens costs about one weight read in memory-bound decode.
- **The D/d ceiling:**
  - The vllm-rlt cost model 2(K+1)/(K+2) tends to 2 = D/d as K grows, because sequential drafting still costs d loops per token.
  - Ouro (D=4, d=1) caps a naive DtV near 4×; Huginn (D=32, d=4) near 8×.
  - The reported 2.4–3.4× (Ouro) and 3.5–7× (Huginn/Raven) sit under those caps. Pipelining exists to get closer to them.
  - For the user's 3-pass models (A→B→A→B→A→B, D=3), draft at 1 pass gives a ceiling near 3×. Draft at 2 passes caps a synchronous scheme at 1.5×.
- **Acceptance rate:** The vllm-rlt concurrency-1 measurements track the all-accepted cost model almost exactly (1.33/1.51/1.66/1.80× vs 1.33/1.50/1.67/1.80×). That suggests very high greedy acceptance at 2 of 4 loops on that short benchmark. It also suggests the gains in that eager, launch-bound setup came partly from amortizing per-pass kernel launches over more tokens (29 tok/s for a 1.4B model on a B300 is far below the bandwidth limit). This is an inference; acceptance was not reported.
- **Edge fit:** At batch 1 on phones, Jetson or WebGPU, decode is memory-bound and/or dispatch-bound (Q5/Q6). Verifying K+1 positions in one pass reuses the same weight bytes and the same dispatch count, so the relative benefit should be at least as large as on the GPUs above. No edge measurements exist, though.
- **Long context:** Cross-depth KV reads grow with draft depth and context. WFD reports a long-context limitation, and its KV sharing variant helps. Expect smaller gains at long context on bandwidth-starved edge devices.

### Gaps
- No acceptance-rate tables for WFD, LoopSpec or DAS were visible.
- No speedup for any method at batch > 1 except vllm-rlt concurrency 8.
- No measurements on phones, Jetson or browsers.
- No method has been tested on vision/video looped models.
- The model behind LoopSpec's 6.83× is unattributed; it is likely Raven at 32 recurrences, but unconfirmed.
- Ouro and Huginn papers' own self-speculation sections could not be read (arXiv blocked). The Huginn repo README does not mention it ([seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)).

---

## Q3. Parallelizing loop iterations (cross-loop parallelism, staggered/diagonal scheduling, Jacobi / diffusion-forcing samplers)

### Takeaway
There are two families.
1. **Train-time architectural changes** make loops for different tokens independent within one forward pass.
   - **Parallel Loop Transformer**, PLT (ByteDance Seed, Oct 2025): 2-loop quality at ~1.02× vanilla decode latency instead of ~2×.
   - **StagFormer** (Jan 2025): ~33% potential decode saving, simulated.
2. **Training-free diagonal samplers** run (token, depth) pairs on a wavefront.
   - **Diffusion-forcing sampler** for Huginn (Oct 2025): up to ~5×.
   - **WaveFront/LoopSpec/DAS** (Q2): 2.4–7×.

All exploit the fact that each loop runs the same weights, so a diagonal of (token, loop) pairs is one batched call.

### Cited Findings
**Parallel Loop Transformer, PLT (ByteDance Seed; arXiv 2510.24824, posted 2025-10-28)**
- **Cross-Loop Parallelism (CLP)** computes different loops for different tokens in the same forward pass, breaking the sequential loop dependency.
- **Efficient Representation Enhancement** shares the first loop's KV cache with all later loops and adds gated sliding-window attention (gated-SWA) for local context.
- [SE] — [arXiv 2510.24824](https://arxiv.org/html/2510.24824v1); [papers.cool](https://papers.cool/arxiv/2510.24824)
- Table 3, on an in-house Seed-MoE model with 680M activated / 13B total parameters:

  | Batch size | Vanilla | Sequential loop-2 | PLT-2 |
  |---|---|---|---|
  | 4: decode latency per token | 4.8 ms | 9.4 ms | **4.9 ms (1.02×)** |
  | 4: KV cache | 280M | 560M | 284M |
  | 4: average score | 34.7 | 39.7 | 39.7 |
  | 64: decode latency per token | 10.9 ms | n/a | 11.3 ms (1.04×) |

  - The near-vanilla latency depends on decode being memory-bound.
  - [SE] — [arXiv 2510.24824 PDF](https://arxiv.org/pdf/2510.24824); [alphaXiv](https://www.alphaxiv.org/abs/2510.24824)
  - Summary: "+5.0 points at ~+2% latency, +1.4% KV". [2nd] — [Emergent Mind](https://www.emergentmind.com/topics/parallel-loop-transformer)
- **LoopCoder-v2: "Only Loop Once"** (arXiv 2606.18023, Jun 2026) applies PLT to code models.
  - Two loops give most of the gain. SWE-bench Verified rises from 43.0 to 64.4 with two loops (2nd-hand).
  - PLT's shared first-loop KV keeps memory fixed regardless of loop count.
  - Model size/token counts conflict across summaries (7B/18T vs 70B/180T).
  - [SE/2nd] — [arXiv 2606.18023](https://arxiv.org/html/2606.18023); [hyper.ai](https://hyper.ai/ko/papers/2606.18023); [besthub](https://www.besthub.dev/articles/why-parallel-loop-transformers-peak-at-two-iterations-insights-from-loopcoder-v2-31cc33a4fd8b)

**StagFormer (arXiv 2501.15665, Jan 2025)**
- Staggers the stack in time: a later section at position t depends only on the earlier section's outputs at positions <t (shifted one step). The sections can then run in parallel during decode.
- "Potential 33% speedup in decoding while being quality neutral **in our simulations**".
- Has a **weight-shared-sections variant** "more practical in settings with limited memory", plus bounded-window cross-section attention and >2 sections. [SE] — [arXiv 2501.15665](https://arxiv.org/pdf/2501.15665)

**Diffusion-forcing / parallel samplers for recurrent depth (arXiv 2510.14961, Oct 2025; NeurIPS 2025 Efficient Reasoning workshop; ELLIS Tübingen / MPI-IS / CMU)**
- Treats intermediate recurrent states as drafts and decodes a new token every forward pass while refining earlier tokens' latents in parallel. This is a diagonal pattern across sequence × depth.
- No retraining needed. Does not reduce FLOPs; it uses GPU parallelism better.
- Results:
  - **Up to 5×** on Huginn-0125 (3.5B) at ~1% accuracy cost on GSM8K, MATH500, HumanEval and MBPP.
  - GSM8K throughput rose **36.1 → 182.2 tok/s** (measured, HF transformers with dynamic KV caching, median tok/s).
  - Reported to beat tuned speculative-decoding baselines on the same model.
  - [SE] — [arXiv 2510.14961](https://arxiv.org/pdf/2510.14961); [alphaXiv overview](https://www.alphaxiv.org/overview/2510.14961v1)
- Lacuna reports median 4.3–4.8× across benchmarks. [2nd] — [Lacuna](https://lacuna.tiptreesystems.com/work/efficient-parallel-samplers-for-recurrent-depth-models-and-their-connection-to/wrk_123b5a1e0b851e23edeb7bd13f589b93)
- The hardware and batch size were not visible.

**Other inference-system papers on looped LMs (index only; mostly not read)**
- A 2026 survey index lists 30 "inference systems" papers for looped LMs, including WFD, DAS, LoopSpec, CDB, PLT, StagFormer and the parallel samplers. It also lists:
  - FlashLoop.
  - "Attention Routing Stabilizes Early: Working-Set Inference for Recurrent LMs" (2609.27373).
  - Looped Latent Attention (cross-loop KV compression, 2607.15456).
  - Memory-Efficient Looped Transformer (2605.07721).
  - "Shared Memory in Looped Transformers" (2610.02383).
  - SPEED (2310.12072, speculative pipelined decoding).
  - [P-repo] — [ArminAzizi98/looped-lm-survey](https://github.com/ArminAzizi98/looped-lm-survey)
- **FlashLoop** (Yang & Liu; arXiv 2609.29812, Sep 2026), training-free:
  - State changes between loops concentrate on few tokens, and adjacent-loop KV residuals quantize well.
  - Combines token-sparse updates, sparse attention and int4 KV-residual quantization.
  - Results: up to **1.64× end-to-end** and **6× KV memory reduction**, "lossless".
  - Motivating number: a 32K-token prefill takes 27 s on Ouro-2.6B (4 loops) vs 3 s on LLaMA-3.1-8B.
  - [SE] — [arXiv 2609.29812](https://arxiv.org/html/2609.29812v1); code at [Superone77/FlashLoop](https://github.com/Superone77/FlashLoop)
  - It overlaps the "fewer tokens per pass" slice.

### Inferences
- **Exact vs approximate parallelism:**
  - In an exact looped LM, token t+1's loop-1 input is token t's *final* output, so diagonal (token, loop) execution needs either speculation (WFD/LoopSpec/diffusion forcing, which verify to stay lossless or near-lossless) or an architecture trained for it (PLT, StagFormer).
  - For the user's per-frame vision/video models, frames are independent inputs (unless the model carries state across frames). A diagonal across (frame, loop) is therefore exact with no training. The cost is that a frame's result arrives after R wavefront steps.
- **Prefill regime:** PLT's ~1.0× claim is for memory-bound decode. In prefill or per-frame vision (compute-bound), CLP still performs R× the FLOPs, so it should not cut latency there. PLT-2's latency rising to 1.04× at batch 64 hints at this.
- **What fits an existing model:** For a model already trained as A→B→A→B→A→B, only the training-free samplers apply. PLT/StagFormer would require retraining with the cross-loop KV sharing and shifted dependencies.

### Gaps
- No "Jacobi/fixed-point decoding of recurrences" paper beyond the diffusion-forcing sampler was found. The sampler's link to Jacobi iteration is implicit.
- PLT's hardware type was not visible.
- StagFormer's measured (non-simulated) latency was not visible.
- No paper evaluates cross-loop parallelism on edge hardware or vision models.

---

## Q4. Weight residency (keeping the shared block on-chip across passes)

### Takeaway
The only measured on-device evidence is MobileLLM's (ICML 2024). On an iPhone 13, running each block twice back-to-back ("immediate block-wise sharing") cost **+2.6% execution time** (16.0 ms vs ~15.6 ms). A non-shared model of the same depth took **29.0 ms**. The authors attribute this to the shared block staying in on-chip SRAM/cache.

On NVIDIA, Ampere-and-later persisting-L2 controls (`cudaAccessPolicyWindow` / `cudaLimitPersistingL2CacheSize`) exist. However, no measured result for looped-model weights, or for any LLM weights on Jetson Orin, was found. Orin's GPU L2 (~4 MB, third-party figure) is small.

### Cited Findings
**MobileLLM (Liu et al., Meta; arXiv 2402.14905; ICML 2024)**
- "Immediate block-wise" sharing repeats each block twice in a row. Where memory movement is the latency bottleneck, adjacent-block sharing avoids weight transfer: the block's weights stay in SRAM/cache, so DRAM↔SRAM movement is skipped. The paper cites ~20 MB of usable SRAM (elsewhere 8–32 MB, inconsistent across sources). [SE] — [arXiv 2402.14905](https://arxiv.org/html/2402.14905v2)
- **Measured** on iPhone 13 (iOS 17.2.1), ExecuTorch with the Metal Performance Shaders backend, FP16:

  | Model | Load | Init | Execute |
  |---|---|---|---|
  | MobileLLM-LS-125M (shared, 2× depth) | 43.6 ms | 1388.2 ms | 16.0 ms |
  | 60-layer non-shared model | 68.6 ms | 3347.7 ms | 29.0 ms |

  - The paper states the sharing overhead as **2.2% (load) and 2.6% (execute)** over the unshared base model.
  - [SE] — [arXiv 2402.14905 v2](https://arxiv.org/html/2402.14905v2); [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.14905)
- A 2025 follow-up (SHARP, arXiv 2502.07832) notes that on mobile devices, loading weights into fast caches is a major part of latency overhead. [SE] — [SHARP](https://arxiv.org/pdf/2502.07832)

**NVIDIA L2 persistence**
- From the CUDA Programming Guide ("L2 Cache Control"):
  - On Ampere and later, part of L2 can be set aside for persisting accesses via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)`, then targeted with an access-policy window (`hitRatio`, `hitProp=persisting`).
  - The guide's example caps the set-aside at min(75% of `l2CacheSize`, `persistingL2CacheMaxSize`).
  - The set-aside is **disabled in MIG mode** and cannot be changed under MPS.
  - [SE] — [CUDA Programming Guide §4.13](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- Jetson AGX Orin and Orin Nano:
  - GPU L2 is **4 MB** (third-party wiki; not confirmed from NVIDIA's brief).
  - A separate 4 MB system cache is described as CPU-side.
  - NVIDIA forum threads leave it unresolved whether GPU traffic uses the system cache.
  - [2nd] — [RidgeRun AGX Orin](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/Introduction/SoM_Overview); [NVIDIA forum](https://forums.developer.nvidia.com/t/jetsonagx-orin-system-level-cache/314129)
- Anecdotal measured L2-pinning results:
  - An RTX 3060 Laptop showed a persisting budget of 2.16 MB, about 69% of L2. [P-repo] — [GauthamMK-0/cuda-kv-persistence](https://github.com/GauthamMK-0/cuda-kv-persistence)
  - An informal RTX 5090 post pinned a small model's weight pool in L2 and reached 1,255 tok/s, a "67% improvement over FP16". That figure confounds L2 pinning with IQ4 quantization. Attribution to the ai.rs post is from a search summary. [2nd] — [ai.rs](https://ai.rs/ai-developer/memory-wall-disappears-llm-inference-bottlenecks)
- A llama.cpp fork PR made L2 persistence of *recurrent states* opt-in. Results were mixed: one trace was slower with persistence on (241 µs vs 123 µs per token), while the title claims +2.6% in another setting. [P-repo via SE] — [torad-labs/llama.cpp PR #82](https://github.com/torad-labs/llama.cpp/pull/82)

**Mobile system caches**
- Apple A17 Pro: 24 MB system-level cache. [2nd] — [Wikipedia: Apple A17](https://en.wikipedia.org/wiki/Apple_A17)
- M3-family SLC sizes are unofficial estimates (8–64 MB, conflicting). [2nd] — [MacRumors thread](https://spy.macrumors.com/threads/m3-pro-max-l2-and-slc-sizes.2410275)
- No figure was found for Snapdragon 8 Gen 3.

### Inferences
- **Byte arithmetic:**
  - A transformer block with hidden size d has ~12d² parameters.
  - At d=576 (MobileLLM-125M), that is ≈4M params ≈ 8 MB FP16 (≈2 MB INT4). It fits in a phone SLC (A15/A17-class) and, at INT4, in Orin's 4 MB L2.
  - At d=2048 (~1–2B-class models), a block is ≈50M params ≈ 25 MB INT4. It exceeds Orin's L2 and roughly fills an A17 Pro SLC.
  - So residency is only a lever for **small shared blocks**, which the user's "few unique blocks" design favours.
- **Execution order:**
  - In A→B→A→B→A→B order, reuse distance is one block. A and B must fit in cache *together* for the second visit to hit.
  - In A→A→A→B→B→B (MobileLLM's immediate order), only one block must fit. But that is a different network function, so it needs training in that order.
  - A runtime cannot legally reorder A→B→A→B into A→A→B→B for one token, because of data dependence.
  - It *can* get the same reuse by running block A for several independent work items back-to-back: frames, speculative positions, or batch rows at loop r. This is depth-wise batching seen as a cache-locality trick.
- **Regimes:**
  - Residency matters most for batch-1 decode and low-res per-frame models, where weight bytes dominate traffic.
  - In prefill or high-res vision, activation traffic dominates and residency buys little.
- **Platforms:**
  - L2 persistence is unavailable in MIG mode on datacenter GPUs. On Jetson, no MIG constraint applies, but the budget is small.
  - No public API was found for pinning on Apple or Qualcomm SLCs; residency there relies on access order (i.e. back-to-back reuse).

### Gaps
- No measured `cudaAccessPolicyWindow` result for shared/looped weights on any GPU, and none for LLM weights on Jetson Orin. A device-side A/B test is needed.
- NVIDIA's official Orin GPU-L2 figure and `persistingL2CacheMaxSize` were not retrievable (nvidia.com blocked).
- MobileLLM's comparison of "repeat-all-over" (A→B→A→B) vs "immediate block-wise" (A→A→B→B) accuracy could not be re-verified this session. Recalled, unverified: repeat-all-over was slightly more accurate and immediate block-wise was chosen for cache locality. Confirm before citing.
- Whether Apple/Qualcomm GPU/NPU drivers keep a repeated block resident across dispatches has not been measured.

---

## Q5. Kernel/graph-level execution and runtime support (CUDA Graphs, persistent kernels, TensorRT ILoop, ONNX Loop, WebGPU dispatch, llama.cpp/MLC/vLLM status)

### Takeaway
Looping multiplies kernel launches and dispatches by R, just like an unshared model of the same depth, so launch overhead becomes a first-order cost.
- On a B200, eager looped speculation was launch-bound (~1,400 kernels per pass, GPU 13–16% busy).
- On WebGPU, each dispatch costs ~24–71 µs, and fusing 876→564 dispatches gave +53% throughput on Vulkan.
- CUDA Graph capture of the recurrent decode core is implemented in vllm-rlt.

Runtime support for loop-aware execution is thin:
- **llama.cpp** runs looped models by unrolling logical layers over shared weights (merged Jul 2026). Ouro support is an open PR. There is no early exit, depth batching or self-speculation.
- **vLLM** mainline dropped Ouro.
- **TensorRT's ILoop** supports loop fusion, unrolling and invariant hoisting, but no looped-transformer benchmarks were found.

### Cited Findings
**Launch and dispatch overhead**
- vllm-rlt RFC #43 measurements on B200 MIG at B≤8:
  - Eager speculative decoding is launch-bound: ~1,400 kernels per pass, GPU 13–16% busy.
  - With CUDA Graphs, host syncs were still 5–9% of decode time.
  - Cutting syncs per round from 73–110 to 5 gave +2.5% (B=1) and +5.3% (B=8), token-identical.
  - [P-repo] — [vllm-rlt issue #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43)
- Batch-1 decode, unshared 7–8B models, 4 NVIDIA GPUs (arXiv 2605.30571, May 2026):
  - L4 reaches ~81% of its analytic memory floor; H100 only 27%.
  - A CUDA-Graphs A/B gives **1.259× on H100 vs 1.028× on L4**. The gap is attributed to launch overhead that is hidden on slow-memory GPUs and exposed on fast ones.
  - Quantized paths (bnb-nf4, AWQ+Marlin) on L4 fall well short of the 4× weight-traffic reduction.
  - [SE] — [arXiv 2605.30571](https://arxiv.org/html/2605.30571v1)
- WebGPU at batch 1 (Qwen2.5-0.5B/1.5B; single-author 2026 study):
  - Raw API cost ~24–36 µs per dispatch on Vulkan and 32–71 µs on Metal; ~95 µs per op including Python/framework.
  - "Per-operation overhead dominates regardless of kernel quality".
  - Fusion cut dispatches 876→564 for **+53% throughput on Vulkan**, with no benefit on CUDA.
  - [SE] — [arXiv 2604.02344](https://arxiv.org/html/2604.02344v1); related [arXiv 2608.08730](https://arxiv.org/pdf/2608.08730). Which of the two papers carries each number is not fully certain from the excerpt.
- An ONNX Runtime WebGPU SiLU-fusion commit gave decode gains of +5.2% (Qwen3.5-0.8B int4) and +1.1% (4B, Phi-4-mini). Smaller models gain more, because overhead per step is fixed. [SE] — [onnxruntime commit a49da870](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/a49da870aaaeded4b7d8bd9e937fedb2cf321493)

**Graph-level loop constructs**
- **TensorRT `ILoopLayer` / `IRecurrenceLayer`:**
  - The loop construct "enables a rich set of automatic loop optimizations, including loop fusion, unrolling, and loop-invariant code motion".
  - Loop-invariant tensors (e.g. weights) can be used directly inside the loop.
  - The biggest gains often come from unrolling so that repeated MatrixMultiply instances are combined. This works best without a recurrent data dependence along the unrolled dimension, which a looped transformer's depth loop *does* have.
  - [SE] — [TensorRT "Working with Loops"](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-loops.html); [IRecurrenceLayer API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_recurrence_layer.html)
  - No benchmark was found.
- **ONNX `Loop`:**
  - No performance data was found.
  - Subgraphs are evaluated lazily by the parent op. [SE] — [Spox docs](https://spox.readthedocs.io/en/latest/guides/advanced.html)
  - ONNX Runtime only recently fixed run-level profiling of ops nested inside Loop/If/Scan. [SE] — [onnxruntime commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/318a2516fed6dcd58f7bc07c49e90ee4d3351020)
  - ORT execution providers on Jetson-class devices have slow eager subgraph construction at session setup. [SE] — [Jetson-ORB-SLAM3, arXiv 2608.17874](https://arxiv.org/pdf/2608.17874)

**Runtime status (as of Oct 2026)**
- **llama.cpp:**
  - PR #25994 (Nanbeige4.2 looped-model machinery) was merged 2026-07-27. It expands logical layers to `n_layer_phys × num_loops`, shares physical weights, and gives each logical layer its own KV slot, i.e. the loop is unrolled in the graph.
  - Open review notes: `num_loops` silently defaults to 1 if missing; quantized-KV paths are untested. [P-repo] — [llama.cpp PR #25994](https://github.com/ggml-org/llama.cpp/pull/25994)
  - The Ouro PR #29823 is **open** (opened 2026-10-01). Ouro-1.4B has 96 logical layers, and decode is "about 4× slower than a single pass".
  - Early exit is not implemented (the shipped gate threshold of 1.0 never fires). Loop count can be overridden at runtime with `--override-kv ouro.num_loops=int:N`.
  - Logit parity was checked on Apple M4 Metal. Wikitext-2 PPL: 9.53 (F16), 9.54 (Q8_0), 9.84 (Q4_K_M) at 4 loops. One user reports garbled output for 2.6B-Thinking.
  - [P-repo] — [llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823)
- **vLLM:**
  - Native Ouro support landed in v0.11.1 and was later removed from main (#49786). v0.26.0 is the last release with it, per the Ouro project page. Unverified against vLLM release notes. [SE] — [Ouro project page](https://ouro-llm.github.io/)
  - The Huginn repo ships a vLLM v1 plugin. [P-repo] — [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
  - vllm-rlt (standalone, not mainline) is the most complete loop-aware engine. See Q1/Q2.
- **SGLang:** LoopSpec's `sglang_recurrent` fork ([kaist-flexml-lab/loopspec](https://github.com/kaist-flexml-lab/loopspec)); "loop-sglang" is in development by the CDB authors ([continuous-depth-batching](https://github.com/ZH-Haoran/continuous-depth-batching)). [P-repo]
- **Apple Silicon (HF transformers/MPS):**
  - Nanbeige4.2-3B has five deployment bugs, e.g. a silently zeroed RoPE buffer.
  - Its layer reuse "effectively doubles peak attention memory". Chunked prefill extends usable context 2.7× on 32 GiB.
  - [SE] — [arXiv 2608.13987](https://arxiv.org/abs/2608.13987)
- **TensorRT-LLM, MLC/WebLLM, ExecuTorch:** no evidence found of looped-model support or loop-aware scheduling. The GitHub search API was blocked in this session.

### Inferences
- **Dispatch counts:**
  - Weight sharing does not reduce kernel launches. An R-loop model issues the same number of dispatches per token as an unshared model of R× depth.
  - On WebGPU (~25–70 µs per dispatch) and mobile GPUs, a 3-pass model with ~15 dispatches per block × 2 blocks × 3 passes ≈ 90 dispatches per token is ≥2–6 ms of pure overhead per token (illustrative arithmetic).
  - Fusion, graph capture and persistent kernels matter as much as for unshared models.
- **Loop-specific kernel opportunity:** One persistent "block kernel" can be relaunched R times, or iterate internally, without new weight bindings. On Vulkan/Metal/WebGPU this means bind groups and pipelines can be reused across passes, and the loop can be recorded once into a reusable command buffer or bundle. No study measures this for looped transformers (gap).
- **Speculative decoding and batching** (Q2, Q1) also amortize launch overhead: one dispatch sequence serves K+1 tokens. This is likely part of why eager vllm-rlt speedups track the ideal cost model.
- **Graph-level loop constructs:** In TensorRT/ONNX Runtime, using `ILoop`/`Loop` instead of unrolling keeps engine size small. TensorRT's documented big win (combining unrolled MatMuls) does not apply across depth iterations, which are data-dependent. Expect unrolled-with-shared-weights graphs (llama.cpp's approach) to perform at least as well. This is unbenchmarked.

### Gaps
- No benchmarks found for TensorRT `ILoop` vs unrolled looped transformers, or for ONNX `Loop` overhead.
- No WebGPU/MLC/ExecuTorch looped-model kernel studies.
- CUDA Graph speedup specifically for looped decode was not isolated; vllm-rlt reports only post-graph sync gains.
- No persistent/megakernel implementation for looped transformers was found.

---

## Q6. Regime map: which tricks help memory-bound batch-1 decode vs compute-bound prefill / per-frame vision

### Takeaway
Almost every loop-specific trick with measured gains targets **memory/launch-bound decode**: depth batching, self-speculation, wavefront/pipelined speculation, diffusion-forcing samplers, PLT and weight residency. Measured speedups are 1.3–7× at batch 1 on datacenter/workstation GPUs, and 1.5–2× in serving.

For **compute-bound prefill and per-frame vision**, the tricks either do nothing (batching is already saturated) or add wasted FLOPs (speculation, CLP). There, only launch/dispatch reduction, KV/token-sparse methods (FlashLoop, up to 1.64×), and fewer passes or tokens help.

### Cited Findings
**Memory-bound decode, batch 1 (measured unless noted)**
- vllm-rlt sync self-speculation: 1.33–1.80× (B300, Ouro-1.4B, concurrency 1). [P-repo] — [PR #44](https://github.com/ThinkFlowLab/vllm-rlt/pull/44)
- WaveFront Decoding, RTX A6000, batch 1. [SE] — [arXiv 2609.23033](https://arxiv.org/html/2609.23033)
  - 2.42× Ouro-2.6B.
  - 3.54× Huginn.
  - 4.81× with cross-recurrence KV sharing.
- LoopSpec: up to 6.83× (model unattributed); ~2.6–3.36× on Ouro. [SE] — [arXiv 2609.17184](https://arxiv.org/html/2609.17184)
- DAS: 4.00–6.96× mean throughput. [SE] — [arXiv 2609.34538](https://arxiv.org/html/2609.34538v1)
- Diffusion-forcing sampler: up to 5× (36.1→182.2 tok/s on GSM8K, Huginn). [SE] — [arXiv 2510.14961](https://arxiv.org/pdf/2510.14961)
- PLT-2 at batch 4: 4.9 ms/token vs 9.4 ms sequential loop-2 (≈1.9× faster than looping sequentially) vs 4.8 ms vanilla. The near-vanilla latency is premised on memory-bound decode. [SE] — [arXiv 2510.24824](https://arxiv.org/pdf/2510.24824)
- Weight residency (MobileLLM, iPhone 13): a second pass of a cached block costs +2.6% execution time vs +86% for a non-shared model of equal depth (29.0 vs ~15.6 ms). [SE] — [arXiv 2402.14905](https://arxiv.org/html/2402.14905v2)
- KV side of decode traffic:
  - Ouro keeping only the last loop's KV during decode: GSM8K 78.85 vs 78.92 (full cache) at **4× less KV memory**.
  - Keeping only the first loop's KV collapses accuracy to 18.73.
  - Sharing caches across loops during *prefill* hurts accuracy.
  - [SE] — [arXiv 2510.25741 (Ouro)](https://arxiv.org/pdf/2510.25741)
  - Nanbeige4.2 sharing KV across loops halves the KV cache but lowers gains. [SE] — [arXiv 2607.22083](https://arxiv.org/html/2607.22083)

**Serving, batch > 1**
- CDB: 1.5–1.9× offline throughput, 45–90% lower normalized latency. [SE] — [arXiv 2608.09444](https://arxiv.org/abs/2608.09444)
- MoR-4: up to 2.06× at max batch. [SE] — [arXiv 2507.10524](https://arxiv.org/pdf/2507.10524)
- vllm-rlt speculation at concurrency 8: 1.32–1.58×, regressing at K=8. [P-repo] — [PR #44](https://github.com/ThinkFlowLab/vllm-rlt/pull/44)
- PLT at batch 64: 1.04× vanilla latency. [SE] — [arXiv 2510.24824](https://arxiv.org/pdf/2510.24824)
- RRT: 2–3× (simulated, oracle exit). [SE] — [arXiv 2410.20672](https://arxiv.org/pdf/2410.20672)

**Compute-bound prefill / long context**
- Ouro-2.6B (4 loops) 32K prefill takes 27 s vs 3 s for LLaMA-3.1-8B. FlashLoop (token-sparse later loops + sparse attention + int4 KV residuals) gives up to 1.64× end-to-end and 6× less KV memory. [SE] — [arXiv 2609.29812](https://arxiv.org/html/2609.29812v1)
- llama.cpp Ouro: decode ≈4× slower than one pass, i.e. no loop-aware savings. [P-repo] — [PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823)

**Launch/dispatch-bound regime (small models, fast GPUs, browsers)**
- CUDA Graphs: 1.259× on H100 vs 1.028× on L4 (unshared models). [SE] — [arXiv 2605.30571](https://arxiv.org/html/2605.30571v1)
- WebGPU fusion: +53% on Vulkan. [SE] — [arXiv 2604.02344](https://arxiv.org/html/2604.02344v1)
- Eager looped speculation: GPU 13–16% busy. [P-repo] — [vllm-rlt #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43)

### Inferences
**Phones and browsers (LLM decode, batch 1)**
- Decode is bandwidth- and dispatch-bound.
- The highest-value loop-specific tricks:
  1. Self-speculation with draft = 1 pass, verify = remaining passes. Prefer the wavefront/pipelined form (WFD/LoopSpec) over synchronous DtV, which is capped at D/d.
  2. Back-to-back reuse of a small shared block so it stays in SLC.
  3. Fused or recorded command buffers, because R passes multiply dispatches.
- Expected gains are not measured. On datacenter GPUs the same methods gave 2.4–3.4× at R=4. With the user's R=3, the theoretical cap for draft=1 is ~3×.

**Jetson Orin**
- The same logic applies to LLM decode. CUDA Graphs are available, and L2 persistence is available without the MIG restriction.
- The 4 MB L2 means only INT4 blocks around hidden size 512–600 fit.
- Orin's lower bandwidth and slower clocks put it closer to the L4 end (launch overhead mostly hidden) than the H100 end, by analogy with arXiv 2605.30571. CUDA Graphs may therefore give smaller relative gains than on H100. Untested.

**Per-frame vision/video (compute-bound at realistic resolutions)**
- Speculation does not apply: there is no autoregressive dependence across tokens within a frame.
- Depth-wise batching across frames is exact but adds latency, and it raises throughput only if a single frame underutilizes the GPU/NPU (small inputs, small models).
- Weight residency helps only when weight bytes are a sizable share of traffic, i.e. low resolution with a large hidden size.
- The main loop-specific execution lever is **launch/dispatch amortization**: record one pass and replay it R times.
- Real compute savings must come from fewer passes or tokens (other researchers' slices).
- Warm-starting from the previous frame's state is adjacent evidence: RecSlowFast reports 279% higher average FPS by reusing prior-frame hidden states with fewer recurrent steps on a desktop GPU [SE, from search summary] — [PMLR v243](https://proceedings.mlr.press/v243/wang24a/wang24a.pdf). That is a fewer-passes technique.

**Prefill (LLM)**
- Already GEMM-saturated. Depth batching and speculation do not help; PLT's CLP does not remove FLOPs.
- KV-residual quantization and token-sparse later loops (FlashLoop) are the measured prefill wins.
- Chunked prefill controls the doubled peak attention memory seen on Apple Silicon (Nanbeige4.2: 2.7× more usable context).

### Gaps
- No loop-specific trick has a published measurement on a phone SoC, WebGPU browser or Jetson.
- No study quantifies the crossover batch size at which self-speculation or depth batching stops paying off on edge accelerators. On B300, vllm-rlt saw diminishing and negative returns at concurrency 8, K=8.
- No vision or video looped-model execution study (wavefront across frames, residency, graph capture) was found.
- Acceptance rates for the 2026 looped self-speculation papers were not retrievable, so their speedups cannot be decomposed into acceptance vs per-step cost on edge hardware.
