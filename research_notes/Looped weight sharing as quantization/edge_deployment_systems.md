# Edge deployment systems for looped (weight-shared) and "shared base + per-layer delta" transformers

## Q1. Does reusing the same weights across passes speed up bandwidth-bound edge decode, or does it only save storage? How big are on-chip caches compared with block sizes?

### Takeaway
Reusing weights saves storage, download size and resident weight memory. It only reduces DRAM traffic per token, and so speeds up decode, when the reused block is still in on-chip cache at the moment it runs again. The one measured case is MobileLLM's "immediate" repeat (the same block run twice back to back), which costs only +2.6% execution time against +86% for an untied model of the same depth, on iPhone 13. The large throughput claims from looped models come from early exit and depth-wise batching, not from weights staying in cache: MoR claims "up to 2×", and Relaxed Recursive claims 2–3× with an oracle exit policy. On-chip caches are 4–24 MB, while one transformer block of a 1–3B-class model is about 6–100 MB, so cross-pass residency is realistic only for very small blocks or immediate repeats.

### Cited Findings
**Measured or claimed speedups from looped and shared models**
- MobileLLM (Meta, ICML 2024) uses "immediate block-wise weight sharing": the same block runs twice in a row so its weights "can stay in the cache and be immediately computed twice". The authors picked this layout over other sharing layouts that scored higher on accuracy, for this cache reason. Sharing adds +0.7% / +0.8% accuracy on the 125M / 350M models — [MobileLLM, ICML 2024](https://proceedings.mlr.press/v235/liu24ce.html); [arXiv 2402.14905](https://arxiv.org/pdf/2402.14905)
- MobileLLM latency setup: FP16, ExecuTorch, iPhone 13 on iOS 17.2.1 with the Metal Performance Shaders backend (Table 7). MobileLLM-LS-125M adds about 2.2% to load and initialization time and about 2.6% to execution time over MobileLLM-125M. An untied model with doubled layers adds about 143% to load/init and about 86% to execution. The paper attributes the small overhead to data locality — [arXiv 2402.14905](https://arxiv.org/pdf/2402.14905) (figures from the paper's text as returned by search; I could not open the table itself)
- Mixture-of-Recursions (NeurIPS 2025): "up to 2× greater inference throughput compared to standard transformers at similar accuracy". It also reduces training FLOPs and memory — [MoR GitHub README](https://github.com/raymin0223/mixture_of_recursions). A secondary wiki gives "up to 2.06×" at the 360M scale with 4 recursions; I did not verify this against the paper — [aiwiki MoR](https://aiwiki.ai/wiki/mixture_of_recursions). The speedup comes from per-token early exit plus batching tokens at different depths through the shared block (depth-wise batching), not from cache residency — [MoR README](https://github.com/raymin0223/mixture_of_recursions)
- Relaxed Recursive Transformers (ICLR 2025): "Continuous Depth-wise Batching" is presented as an analysis projecting "2–3×" throughput. The estimates assume oracle exiting, where a token exits at the earliest loop whose prediction matches the final loop. They report up to 2.66× generation speedup over vanilla models and nearly 4× over vanilla Pythia, all simulated from measured per-token timings. The authors say practical early-exit algorithms are out of scope — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html); [arXiv 2410.20672](https://arxiv.org/pdf/2410.20672)
- ALBERT (cross-layer sharing) does not cut compute. ALBERT-xxlarge has about 70% of BERT-large's parameters (235M vs 334M) but is about 3× slower; ALBERT-large is about 1.7× faster. These are training-throughput numbers on TPU; I found no inference-latency measurement — [ALBERT arXiv 1909.11942](https://arxiv.org/pdf/1909.11942)
- 2026 looped-model systems papers, all from search abstracts:
  - FlashLoop: up to 1.64× end-to-end speedup and up to 6× KV-cache memory reduction, from lazy KV updates, not weight residency — [arXiv 2609.29812](https://arxiv.org/html/2609.29812)
  - LT2, a linear-time looped transformer: about 5.7× decode throughput at 8k context (125 vs 22 tok/s, batch 8), from removing quadratic attention — [arXiv 2605.20670](https://arxiv.org/html/2605.20670v1)
  - Looped Latent Attention: 2.3× decode speedup in a random-weight kernel benchmark at 262k context — [arXiv 2607.15456](https://arxiv.org/html/2607.15456v2)
  - The Fleet paper notes that on multi-die GPUs each die's L2 caches weights separately, which causes redundant HBM traffic during decode. This is a reminder that weights staying in L2 depends on the hardware — [arXiv 2604.15379](https://arxiv.org/pdf/2604.15379)

**Cache sizes and bandwidth of target hardware**
- Jetson Orin, Ampere iGPU: 4 MB GPU L2 and 192 KB L1 per SM, with up to 16 SMs — [Hot Chips 34 NVIDIA slides](https://hc34.hotchips.org/assets/program/conference/day2/ADAS%20and%20Grace/HC2022.NVIDIA.Mike_Ditty.v6.pdf); [NVIDIA forum](https://forums.developer.nvidia.com/t/orin-system-cache/223472)
  - The SoC also has a 4 MB system cache. Whether GPU traffic uses it is undocumented: NVIDIA staff declined to give details, and one user reads the TRM as saying it is a CPU-side L4 — [NVIDIA forum](https://forums.developer.nvidia.com/t/jetsonagx-orin-system-level-cache/314129); [Orin technical brief](https://nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)
  - The same 4 MB L2 is listed for Orin Nano — [RidgeRun](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_Nano/Introduction/SoM_Overview)
- Apple A18 Pro: 24 MB system-level cache (SLC), shared by all units including the NPU. The A18 has 12 MB. The P-core L2 is 16 MB on the A18 Pro vs 8 MB on the A18 — [heise](https://heise.de/-9960558); [A18 Pro wiki mirror](https://wiki.krisyotam.com/content/wikipedia_en_all_nopic_2026-03/A18_Pro). I found no M4 SLC figure.
- Snapdragon 8 Elite: 8 MB SLC/L3, 12 MB L2 per CPU cluster, and four 16-bit LPDDR5X channels — [Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Processor-Benchmarks-and-Specs.908499.0.html). Memory speed is quoted as "5.3GHz LPDDR5x" by [Android Central](https://www.androidcentral.com/phones/qualcomm-announces-snapdragon-8-elite-chip) and as "LPDDR5x 4800" by Notebookcheck; the sources conflict.
- Jetson memory bandwidth (spec peaks):
  - AGX Orin 64GB: 256-bit LPDDR5 at 204.8 GB/s, about 275 TOPS — [Connect Tech](https://connecttech.com/orin-module-comparison/)
  - Orin NX 16GB: 128-bit at 102.4 GB/s — [NVIDIA JP](https://www.nvidia.com/ja-jp/autonomous-machines/embedded-systems/jetson-agx-orin-campaign)
  - Orin Nano 8GB: 68 GB/s per NVIDIA JP; the Nano Super is listed at 102 GB/s by a reseller ([Viperatech](https://viperatech.com/product/nvidia-jetson-agx-orin-64gb-developer-kit)). These conflict.
  - Jetson T5000 (AGX Thor): 128 GB 256-bit LPDDR5X at 273 GB/s, 40–130 W — [Connect Tech](https://connecttech.com/product/nvidia-jetson-t5000-module-900-13834-0080-000/). The T4000 is listed at 64 GB, 273 GB/s, provisional — [Generation Robots](https://www.generationrobots.com/en/404454-nvidia-jetson-thor-module.html)
- Measured bandwidth is far below spec: a STREAM test on AGX Orin 32GB measured about 54.5 GB/s against the 204.8 GB/s spec — [NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-agx-orin-32gb-measured-memory-bandwidth-much-lower-than-theoretical-spec/331766)
- iPhone: A18 bandwidth is about 68 GB/s (unverified third-party figure). Community llama.cpp numbers on iPhone 16 Pro are 40–50 tok/s for 1B Q4 (~0.6 GB) and 22–28 tok/s for 3B Q4 (~1.7 GB), consistent with decode being bandwidth-bound — [ertas.ai benchmarks](https://www.ertas.ai/blog/llm-run-on-iphone-benchmarks)

### Inferences
- **Decode bytes do not drop without residency.** Bytes per token ≈ (number of passes) × (bytes per block) + KV reads. A looped model of depth N therefore has the same decode bandwidth ceiling as an untied model of depth N at the same bit-width, unless the block is still cached when it is reused.
- **Block sizes vs caches** (my arithmetic, using about 12·d² parameters per standard block: 4d² attention + 8d² FFN):

  | d_model | int4 per block | fp16 per block |
  |---|---|---|
  | 768 | ~3.5 MB | ~14 MB |
  | 1024 | ~6.3 MB | — |
  | 2048 | ~25 MB | ~100 MB |

  - Against Orin's 4 MB L2, only blocks at about d ≤ 768 in int4 could fit, and KV, activations and other traffic share that L2.
  - On A18 Pro (24 MB SLC) or Snapdragon 8 Elite (8 MB SLC), a d=2048 int4 block is at or over capacity.
- **Loop order matters for residency.**
  - In a cyclic loop (blocks 1..K, then repeat), the reuse distance is K blocks plus all the KV streamed in between. With K ≥ 2 and a 1B-class width, that reliably exceeds every listed cache.
  - In an "immediate repeat" schedule (block i run r times in a row, as in MobileLLM-LS), the reuse distance is one block, which is the only layout with measured near-free reuse.
  - For the user: if bandwidth gains matter, prefer immediate-repeat schedules with small int4 blocks. Otherwise, expect storage and RAM savings only.
- **Batched early exit and depth-wise batching** (MoR, Relaxed Recursive) are throughput features for serving many sequences. At batch 1 on a phone or browser, the realistic gain is skipping passes for easy tokens (fewer passes means fewer bytes), not batching.
- **Compute-bound phases gain nothing from sharing.** Prefill, and per-frame vision encoders, are compute-bound: FLOPs per token are unchanged by sharing, as ALBERT illustrates.

### Gaps
- I found no direct measurement on Jetson, Apple GPU/ANE, Adreno or Hexagon of DRAM traffic or speedup from cross-pass weight reuse in a looped LLM. Neither MoR nor Relaxed Recursive reports on-device decode latency.
- I found no official GPU-L2 figure for Orin in NVIDIA's own datasheets, only Hot Chips slides and the forum. The GPU's use of the 4 MB system cache is undocumented.
- M4 / M-series SLC sizes and official Apple memory bandwidth for the iPhone A18/A19 were not found from primary sources.
- Access caveat: arxiv.org, huggingface.co, nvidia.com and most docs sites were unreachable from this environment (proxy policy). Paper numbers come from search-engine extracts, not full-text reads; runtime facts were verified from GitHub source and docs at the commits linked below.

## Q2. What does it cost to reconstruct W = shared base + per-layer delta at runtime?

### Takeaway
At batch size 1, a fused dequant GEMV (matrix–vector multiply) on a 4-bit base gets close to the ideal speedup (Marlin: 3.87× theoretical at batch 1). Adding a separate low-rank or 1-bit delta path is not free on bandwidth-bound hardware. Every pass must read the base block plus that layer's delta, so a "shared base + delta" layer reads more bytes per pass than a plain int4 layer unless the base stays in cache. Unfused LoRA side paths measured at batch 1 cost roughly 10–40%.

There are two workable designs:
- **Materialize at load:** add each delta into the base once when the model loads. This gives the download savings, and runtime is identical to a dense model on every runtime.
- **Fused on-the-fly kernels:** apply the delta inside the matmul. This also saves RAM, but needs custom kernels, which only CUDA and ggml-style backends realistically allow.

### Cited Findings
**Fused dequant kernels on the base**
- Marlin, an FP16×INT4 kernel, achieves "close to ideal (4×) speedups up to batchsizes of 16–32". Prior kernels reach "close to the optimal 3.87× … at batchsize 1"; the 3.87× rather than 4× reflects the 0.125 bit/weight overhead of group scales. Weights and scales are reshuffled offline so they dequantize directly into tensor-core layout. On A10 with ECC enabled, achievable bandwidth is 10–15% lower — [IST-DASLab/marlin README](https://github.com/IST-DASLab/marlin)

**Low-rank (LoRA-style) side paths**
- Punica computes y = x·W + x·A·B, with a CUDA kernel (SGMV) for the batched low-rank add-on. Punica "adds only 2ms latency per token" and claims 12× throughput over other systems in multi-tenant serving — [punica-ai/punica README](https://github.com/punica-ai/punica); [arXiv 2310.18547](https://arxiv.org/pdf/2310.18547)
- S-LoRA's MBGMV kernel handles batches of adapters with mixed ranks — [S-LoRA arXiv 2311.03285](https://arxiv.org/pdf/2311.03285); [LMSYS blog](https://lmsys.org/blog/2023-11-15-slora)
- Measured unmerged-LoRA overhead:
  - On an H100 at batch 1, a single rank-48 unmerged LoRA decoded at 21.0 tok/s vs 33.4 tok/s merged (about 37% slower), with about 0.2 GB more peak memory — [CoLT-Drive arXiv 2609.00242](https://arxiv.org/pdf/2609.00242)
  - Fireworks documents 10–30% higher prompt-processing time for unmerged LoRA — [Fireworks docs](https://docs.fireworks.ai/guides/understanding_lora_performance)
  - The overhead comes from extra per-layer matmuls and kernel launches — [VaLoRA arXiv 2411.00915](https://arxiv.org/pdf/2411.00915)

**1-bit deltas (BitDelta)**
- BitDelta stores one fp16 base plus 1-bit deltas (sign × per-matrix scale), reducing GPU memory ">10×" in multi-tenant settings. Six Mistral-7B fine-tunes are served in 30 GB. The current kernel is Triton, with a faster BitBLAS W1A16 kernel listed as TODO — [FasterDecoding/BitDelta README](https://github.com/FasterDecoding/BitDelta)
- Kernel behaviour:
  - A fused binary GEMM keeps deltas packed until they reach on-chip memory, and gives about 2× faster decoding than running each fine-tuned model separately.
  - At batch size 1, the shared backbone dominates latency. The combined deltas overtake the base weights at about batch 6–8.
  - BitDelta slightly underperforms S-LoRA (rank 128, chosen to match memory at N=M=4096) at large batch.
  - The authors call the kernel "fairly slow relative to the theoretical bound".
  - Sources: [Princeton PLI blog](https://pli.princeton.edu/blog/2024/bitdelta-your-fine-tune-may-only-be-worth-one-bit); [arXiv 2402.10193](https://arxiv.org/pdf/2402.10193)

**How runtimes apply adapters today**
- llama.cpp applies runtime LoRA as an unfused side path in `build_lora_mm`: `res = W·x; res += scale · (B·(A·x))` as separate `ggml_mul_mat`, `ggml_scale` and `ggml_add` ops. Adapters are looked up by tensor name — [llama-graph.cpp L1615–1653 @ efe3626, 2026-10-12](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-graph.cpp#L1615); [llama-adapter.cpp L140](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-adapter.cpp#L140)
- TensorRT Edge-LLM (Jetson) offers two LoRA modes — [TensorRT-Edge-LLM lora.md @ 0f52705, 2026-10-06](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/features/lora.md):
  - Dynamic runtime LoRA: the export tool inserts LoRA inputs into the ONNX graph, adapters are converted to a sidecar format, and the engine is built with `--maxLoraRank` (example 64).
  - Static LoRA merge: the adapter is merged into the checkpoint before quantization and export.

### Inferences
**Delta overhead per pass, relative to an int4 base, when the base is not cache-resident** (my arithmetic)
- Low-rank delta of rank r on a d_in×d_out matrix: extra parameters = r(d_in+d_out).
  - d=2048, r=16: 1.56% of the matrix's parameters and 1.56% extra FLOPs. In fp16 that is about 6.25% extra bytes relative to an int4 base.
  - r=64: about 25% extra bytes in fp16.
- 1-bit sign delta: +25% bytes relative to an int4 base, or +6.25% relative to fp16.
- 2-bit delta: +50% relative to an int4 base.
- So at batch-1 decode, "looped base + per-layer delta" is slightly slower than an untied int4 model of the same depth unless the base is cache-resident. Its advantages are storage, download and RAM.

**Storage per layer-slot** (my arithmetic: 24 passes from 4 unique int4 bases)
- With 1-bit deltas: (4·4 + 24·1)/24 ≈ 1.67 bits/param, about 2.4× smaller than untied int4.
- With rank-16 fp16 deltas at d=2048: about 0.92 bits/param, about 4.4× smaller than untied int4.

**Implementation options**
- Materialize at load (W_l = base + Δ_l, re-quantized to int4 per layer) works on every runtime: ORT, TensorRT, Core ML, LiteRT, WebLLM. It keeps the download savings but loses the RAM savings, and it adds a one-time CPU/GPU pass at load. Note that re-quantizing base+Δ to int4 adds rounding error on top of the trained delta. A quantization-aware design would train or quantize in the final per-layer int4 grid.
- On-the-fly reconstruction is easiest to make efficient in two forms:
  - A low-rank delta as a separate tiny GEMV (llama.cpp's existing LoRA path; small overhead at small r).
  - A packed 1-bit/2-bit delta fused into the base's dequant loop (a Marlin- or k-quant-style kernel that reads base and delta tiles together). This needs a custom kernel per backend: CUDA, Metal, Vulkan, WGSL.
- Low-rank deltas have a kernel-launch cost on mobile GPUs and WebGPU, where dispatch overhead per op is significant. The CoLT-Drive and VaLoRA measurements show launch-dominated overhead even on H100.

### Gaps
- I found no published batch-1 measurement of LoRA side-path overhead on Jetson, Apple GPU, Adreno or WebGPU. I also found no fused "shared base + per-layer 1–2-bit delta" GEMV kernel for any mobile or web backend.
- I could not access Punica's rank-16 batch-1 microbenchmark values, or BitDelta's Figure 4 millisecond values.
- I did not verify llama.cpp k-quant dequant overhead percentages.

## Q3. KV cache in looped models: recursion-wise caching, KV sharing (MoR), and cross-layer KV sharing (CLA, YOCO, MiniCache)

### Takeaway
Sharing weights does not share KV. A looped model with N passes, cached naively, holds N layers' worth of KV, the same as an untied model of equal depth. The options for shrinking it:
- MoR recursion-wise caching gives about (N_r+1)/(2N_r) of baseline KV memory and IO under its routing schedule.
- Sharing first-pass or last-pass KV across recursions cuts KV by about the loop count. Ouro reports about 4× at decode "with minimal loss" (secondary source). MoR reports that sharing hurts expert-choice routing.
- Cross-layer schemes in standard models: CLA gives 2× over MQA at near-equal accuracy, MiniCache 1.53× alone (5.02× with 4-bit), YOCO up to 6.4× memory at 512K, and Gemma 3n gives 2× prefill.
- Runtimes: llama.cpp gives each loop pass its own KV slot today, and separately supports "first-N layers own KV" sharing.

### Cited Findings
**Looped and recursive models**
- MoR recursion-wise KV caching stores KV only for tokens active at each recursion depth. Memory and IO become roughly (N_r+1)/(2N_r) of the baseline across the model; the PDF extraction was garbled, so verify against the paper's table — [arXiv 2507.10524 via search extract](https://arxiv.org/pdf/2507.10524); [emergentmind summary](https://www.emergentmind.com/topics/recursion-wise-kv-caching)
- MoR recursive KV sharing:
  - KV from the first recursion is reused by all deeper recursions, which lowers memory and lets prefill skip deeper recursions.
  - It causes a mismatch, and "quite large performance degradation in expert-choice routing".
  - The authors recommend it when memory and prefill are the bottleneck.
  - Attention FLOPs fall only by k/N_ctx, and KV IO remains a decode bottleneck.
  - Sources: [arXiv 2507.10524](https://arxiv.org/pdf/2507.10524); [MoR README](https://github.com/raymin0223/mixture_of_recursions) (the README notes that in the KV-sharing implementation "we computed all sequences against a shared KV cache for simplicity")
- Ouro (ByteDance LoopLM, 2025):
  - Prefill needs the full cache for every loop step. At decode, sharing the final step's KV across loops costs minimal quality, cutting decode KV about 4× (= loop count). This is a secondary summary of arXiv 2510.25741 — [insiderllm guide](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/); [Raschka LLM architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/)
  - Config: `total_ut_steps` defaults to 4 — [Ouro-1.4B model card](https://huggingface.co/ByteDance/Ouro-1.4B)
  - Serving: reportedly no vLLM support for adaptive exit (always runs all steps), and the upstream repo only does post-hoc gating — [insiderllm](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/); [community fork](https://huggingface.co/KristianS7/Ouro-2.6B)
- Huginn-0125 (recurrent depth, 3.5B, 800B tokens) lists KV-cache sharing, per-token adaptive compute and self-speculative decoding as inference features its architecture supports naturally — [arXiv 2502.05171](https://arxiv.org/html/2502.05171v2)

**Cross-layer KV sharing in standard (untied) models**
- CLA (NeurIPS 2024): sharing KV heads between adjacent layers (CLA2) halves KV again relative to MQA, with accuracy close to MQA, in 1B and 3B models trained from scratch — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9e23d020c18e4c40d81c6a0fc7a46f68-Abstract.html); [arXiv 2405.12981](https://arxiv.org/html/2405.12981v1)
- YOCO (NeurIPS 2024): the KV cache is built once by the self-decoder and reused by the cross-decoder.
  - About 3× faster prefill at 32K.
  - At 512K (Figure 1): about 6.4× lower GPU memory, 30.3× higher throughput, 9.6× lower prefill latency.
  - The prefill speedup comes from exiting early before the cross-decoder.
  - Sources: [arXiv 2405.05254](https://arxiv.org/html/2405.05254v2); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract.html)
- MiniCache (NeurIPS 2024): training-free merging of similar KV states across adjacent middle and deep layers.
  - 1.53× compression alone (LLaMA-2-7B, ShareGPT); 5.02× combined with 4-bit quantization.
  - About 5× throughput and 41% memory reduction vs an FP16 full cache, "near-lossless".
  - Source: [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/93380)
- Gemma 3n:
  - The middle layer's local and global attention KV is shared with the top layers, giving 2× faster prefill vs Gemma 3 4B (Google's claim) — [InfoQ](https://www.infoq.com/news/2025/07/gemma-3n-architecture)
  - Raschka estimates about 2.7 GB (E2B) and about 6 GB (E4B) of KV saved at 128K in bf16 for Gemma-4-style KV sharing — [Raschka](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures)

**Runtime support**
- llama.cpp supports cross-layer KV sharing through `n_layer_kv_from_start` ("the first n_layer_kv_from_start layers have KV cache") and the GGUF key `%s.attention.shared_kv_layers` — [llama-hparams.h L86](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-hparams.h#L86); [llama-arch.cpp L310](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-arch.cpp#L310)
- llama.cpp's looped architectures keep a separate KV slot per pass:
  - Nanbeige: "Share physical weights across loops; each slot still has its own KV index" — [nanbeige.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/nanbeige.cpp)
  - HRM-Text: "the GGUF block count is the expanded cache-slot count" — [hrm-text.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/hrm-text.cpp)
- TensorRT Edge-LLM has FP8 KV and multi-turn KV-cache reuse features (FP8 not on Orin). I found no cross-pass or cross-layer KV sharing feature in its docs — [TRT Edge-LLM README](https://github.com/NVIDIA/TensorRT-Edge-LLM); [support matrix](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/getting_started/support-matrix.md)

### Inferences
**KV size arithmetic** (mine)
- KV per token per pass = 2 · n_kv_heads · head_dim · bytes.
- A 1B-class GQA config (8 KV heads, head_dim 64, fp16) needs 2 KB per token per pass.
  - 16 passes at 4K context: about 128 MB.
  - Looping 4 blocks × 8 = 32 passes: about 256 MB.
- This is comparable to the int4 weights of a heavily shared model, so on phones and browsers KV can dominate memory once weights are shared. KV sharing across loops (final-pass or first-pass) is likely needed to realize the RAM benefit of looping.

**Bandwidth**
- KV reads per decode step grow with passes × context. With weights shared, KV IO becomes a larger fraction of bytes per token. Cross-loop KV sharing therefore helps decode speed as well as memory, which is where FlashLoop and LLA report gains.

**Choosing a KV strategy**
- Training-time strategies (MoR-style routing with recursion-wise caching, Ouro-style last-step sharing, CLA-style sharing) generally preserve quality better than training-free merging (MiniCache) when sharing more aggressively.
- For a "quantization-style" compression applied to an existing model, MiniCache-style merging plus KV quantization is the training-free option.

### Gaps
- Exact MoR KV-memory and throughput tables, and Ouro's own measured KV-sharing quality deltas, could not be read from primary sources (arXiv blocked).
- I found no runtime (llama.cpp, MLC, ORT, TRT Edge-LLM) that implements MoR-style recursion-wise or first-pass-shared KV for looped models out of the box. llama.cpp's `shared_kv_layers` is a layer-index scheme designed for Gemma-3n-style sharing; whether it can be configured to make loop passes share KV without code changes is unverified.

## Q4. Runtime and file-format support for tied or shared tensors and per-pass parameters

### Takeaway
As of October 2026:
- **llama.cpp/GGUF** natively runs looped models. The `{arch}.num_loops` key (Nanbeige) and HRM-Text alias layer structs to the same tensors, so GGUF stores only the physical blocks, and the graph is unrolled with a KV slot per pass. Sharing is coded per architecture; GGUF has no generic tensor aliasing. Per-pass LoRA deltas are not expressible because adapters are keyed by tensor name.
- **ONNX** natively lets many nodes and Loop bodies reference one initializer. Exporters historically duplicated tied weights, so dedupe passes exist.
- **Core ML** (iOS 18+) dedupes identical weights across functions by hashing.
- **ExecuTorch** shares weights across methods and programs via `.ptd` named data.
- **TensorRT Edge-LLM** documents LoRA (dynamic and static) but nothing on looped or shared-block models.
- **LiteRT** and **MLC/WebLLM** have no documented looped-model path.

### Cited Findings
**llama.cpp / GGUF** (master @ efe3626, 2026-10-12)
- GGUF key `NUM_LOOPS = "{arch}.num_loops"` with writer `add_num_loops()` — [gguf-py constants.py L154](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/gguf-py/gguf/constants.py#L154)
- The Nanbeige architecture reads `num_loops` and expands the logical layer count to `n_layer_phys * n_loops` before KV allocation. It aliases `layers[i + j*n_phys] = layers[i]`, with the comment "Share physical weights across loops; each slot still has its own KV index". There is an optional final norm between loops (`skip_loop_final_norm`) — [nanbeige.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/nanbeige.cpp)
- HRM-Text runs alternating low and high stacks over H×(L+1) cycles. Later passes "alias" the layers created in the first pass. Prefix-LM prefill is not implemented — [hrm-text.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/hrm-text.cpp)
- Tied embeddings use the `TENSOR_DUPLICATED` flag: `output` falls back to `token_embd`. The loader returns the existing tensor if it is already allocated in the same buffer-type context, otherwise it creates a duplicate allocation — [llama-model-loader.cpp L1370–1386](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-model-loader.cpp#L1370)
- I found no tensor-alias concept in gguf-py's writer or reader (grep for "alias" returned nothing), so sharing lives in each architecture's C++ loader — [gguf_writer.py](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/gguf-py/gguf/gguf_writer.py)
- The LoRA lookup is `ab_map.find(w->name)`, by tensor name — [llama-adapter.cpp L140](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-adapter.cpp#L140)

**ONNX / ONNX Runtime**
- ONNX IR: nested subgraphs (Loop and Scan bodies) see names "from the outer scopes that are visible in the nested subgraph", and shadowing is disallowed. A Loop body can therefore reference outer-graph initializers (weights) without copying them — [onnx/onnx docs/IR.md](https://github.com/onnx/onnx/blob/main/docs/IR.md); [onnx-ir subgraphs doc](https://onnx.ai/ir-py/subgraphs_and_functions.html)
- PyTorch's TorchScript-based ONNX export "introduces duplicated ONNX initializers for shared weights". This was fixed for eval-mode export by comparing data_ptr, strides and sizes — [PyTorch commit (semanticdiff mirror)](https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/cc792746d28d90b020de7313995f134688398919)
- onnx-ir `DeduplicateInitializersPass` and `DeduplicateHashedInitializersPass` dedupe by shape, dtype and content within each graph. Across subgraphs, initializers must first be lifted to the main graph — [onnx-ir passes](https://onnx.ai/ir-py/api/ir_passes_common.html); [source](https://onnx.ai/ir-py/_modules/onnx_ir/passes/common/initializer_deduplication.html)
- An ORT commit notes O(n²) initializer comparison (with a greedy option) in a Whisper export tool. Another ORT constant-sharing pass needed a size threshold because merging hurt DirectML — [ORT commits via semanticdiff](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/47f136e2d3240e725b1c4b2bedeb5de0c2db240b)
- AMD Quark ships the reverse tool (`convert_shared_initializer_to_unique`) for compilers or quantizers that cannot handle shared initializers — [Quark docs](https://quark.docs.amd.com/release-0.8.2/autoapi/quark/onnx/tools/convert_shared_initializer_to_unique/index.html)
- ORT C API (@ 988cc07, 2026-10-12):
  - `AddInitializer` lets a user-owned OrtValue replace a named initializer, "useful when you want to share the same initializer across sessions".
  - `CreatePrepackedWeightsContainer` shares pre-packed buffers of shared initializers between sessions.
  - Source: [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/988cc07aed84347a7d9b94583d15c099e6407fb7/include/onnxruntime/core/session/onnxruntime_c_api.h)
  - Inference: this is a route for per-pass parameters (e.g., swap deltas without re-loading the base).

**Core ML**
- Multifunction models (iOS 18 / macOS 15) are merged with `save_multifunction`, and "Core ML Tools deduplicates shared weights by calculating the hash of the weight values".
- In the LoRA example, a 72M-parameter base is shared and the rank-32 adapters add 0.768M parameters per function.
- Source: [coremltools multifunction-models.md @ a207b1a, 2026-09-30](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/docs-guides/source/multifunction-models.md); [WWDC24 notes](https://wwdcnotes.com/documentation/wwdc24-10161-deploy-machine-learning-and-ai-models-ondevice-with-core-ml/)
- An ExecuTorch PR adds positional weight sharing across methods for multifunction Core ML (prefill and decode graphs) — [executorch PR #16552](https://github.com/pytorch/executorch/pull/16552)

**ExecuTorch**
- `.ptd` named data: "Multiple named data blobs to reference the same data segment". Use cases include "Deduplication: sharing model weights between multiple executable PTE files" — [ptd-file-format.md @ ad1c24d, 2026-10-12](https://github.com/pytorch/executorch/blob/ad1c24d648b5a12bb51d488573fab94d83dac39e/docs/source/ptd-file-format.md)
- Multi-method PTEs (prefill and decode) share data segments. Program-data separation lets LoRA programs share a foundation-weight `.ptd` — [ExecuTorch paper arXiv 2605.08195](https://arxiv.org/pdf/2605.08195)
- There is a C++ LoRA weight-sharing example that reuses foundation weights through XNNPACK's weight cache — [success-stories.md](https://github.com/pytorch/executorch/blob/ad1c24d648b5a12bb51d488573fab94d83dac39e/docs/source/success-stories.md)
- The export IR supports higher-order control-flow operators — [concepts.md](https://github.com/pytorch/executorch/blob/ad1c24d648b5a12bb51d488573fab94d83dac39e/docs/source/concepts.md)

**TensorRT Edge-LLM** (v0.11.0, 2026-09; docs @ 0f52705)
- Pipeline: HF checkpoint → ONNX → TensorRT engine, plus an experimental direct engine builder without ONNX (0.10.0) — [README](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- LoRA is supported as dynamic sidecar adapters with a max-rank build flag, or as a static merge — [lora.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/features/lora.md)
- A grep of the user guide found no mention of tied or shared-weight looped models.
- Jetson Orin is "Official" on JetPack 7.2 / CUDA 13.2 with "FP16, INT8, and INT4 only"; FP8 and FP4 engines do not run on Orin — [support-matrix.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/getting_started/support-matrix.md)
- Third-party: v0.6.0 was released 2026-03-16 — [Hackster](https://www.hackster.io/shahizat/getting-started-with-nvidia-tensorrt-edge-llm-on-jetson-thor-14735e)

**LiteRT (TFLite)**
- Multiple signatures since 2.7. Signature runners from one interpreter must not run concurrently — [LiteRT signatures](https://developers.google.cn/edge/litert/conversion/tensorflow/signatures)
- I found no documentation on weight dedup across signatures or on WHILE-op support in the GPU delegate.

**MLC-LLM / WebLLM**
- The TVM WebGPU runtime requests a 1 GB `maxBufferSize` and falls back to 256 MB — [apache/tvm web/src/webgpu.ts](https://github.com/apache/tvm/blob/main/web/src/webgpu.ts)
- WebLLM warns if `maxStorageBufferBindingSize` is under 1 GB, and has `low_resource_required` / `buffer_size_required_bytes` flags per model — [web-llm src/engine.ts](https://github.com/mlc-ai/web-llm/blob/main/src/engine.ts); [web-llm src/config.ts](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts)
- I found no looped-model support.

**Other evidence**
- MobileLLM-LS ran on ExecuTorch (iPhone 13, MPS) with only +2.2% load/init time vs 143% for doubled layers, consistent with shared weights not being duplicated in the exported program — [arXiv 2402.14905](https://arxiv.org/pdf/2402.14905)

### Inferences
**Expressing a looped graph without duplicating weights, per runtime**
- **llama.cpp: yes.**
  - Use or adapt the `num_loops` (Nanbeige) pattern. The file holds only K physical blocks, and RAM holds one copy.
  - Per-pass deltas need new architecture code: extra per-slot tensors such as `blk.{slot}.attn_q_delta`, plus an add in `build_lora_mm`-like code. Stock LoRA files cannot target individual passes of an aliased block, because every alias has the same tensor name.
  - Exact deltas could be materialized at load in the loader, giving download savings only.
- **ONNX: yes, two ways.**
  - Unrolled graph where every pass's MatMul references the same initializer name. Size stays small if the exporter or a dedupe pass merges tied tensors.
  - A Loop/Scan body referencing outer-scope weights, with per-pass deltas as stacked tensors indexed by the iteration counter.
  - Whether each execution provider (WebGPU, CoreML, NNAPI/QNN, TensorRT) keeps a Loop on-device or falls back to CPU control flow is unverified; unrolling is safer for EP coverage.
- **TensorRT: likely possible but undocumented.** Whether the engine builder stores one copy when many layers reference one ONNX initializer is unverified, and engines may bake per-layer copies after tactic or layout transforms. Expect to test engine size; the Edge-LLM exporter targets standard HF architectures.
- **Core ML and ExecuTorch: dedupe exists,** but documented only across functions/methods/programs, not within a single graph that reuses a block. Within-graph reuse depends on torch.export lifting a shared parameter once; this is plausible from the MobileLLM-LS numbers, but undocumented.
- **Download format.** The most portable approach for browsers and phones is to ship base and delta tensors in a custom container (or GGUF with new tensor names) and reconstruct per-layer weights at load. Every runtime then sees a standard dense model.

### Gaps
- No primary-source documentation was found on:
  - (a) TensorRT engine-level dedupe of shared initializers;
  - (b) LiteRT/TFLite dedupe and WHILE support on the GPU or NNAPI delegates;
  - (c) MLC-LLM/TVM support for tied blocks or loops beyond tied embeddings;
  - (d) torch.export / ExecuTorch handling of one parameter used by many unrolled calls;
  - (e) ORT WebGPU EP and mobile EP handling of Loop/Scan.
- I could not search GitHub issues for llama.cpp, onnxruntime, executorch or mlc-llm: the GitHub search API was blocked in this session, so findings come from source and docs at the commits linked.

## Q5. Download, storage and memory budgets: browser limits, phone RAM, Jetson memory

### Takeaway
Browser delivery is bounded by several hard limits:
- WebGPU guaranteed buffer limits are 128 MiB per storage binding and 256 MiB per buffer. Adapters often expose 1–4 GB; MLC requests 1 GB.
- A Chrome ArrayBuffer tops out at about 2 GB.
- wasm32 addresses 4 GB. Memory64 ships in Chrome 133+ and Firefox 134+ but not Safari.
- ONNX Runtime Web cannot run models over 4 GB.

Phones: iOS apps report a de facto ceiling of about 6 GB even on 8–12 GB iPhones. Jetson Orin Nano and NX have 8 and 16 GB of unified memory, and measured TRT Edge-LLM footprints are about 1.9–3.6 GB for 0.6–2B int4 models.

Looping plus int4 can shrink transformer-block weights by K/N on top of the ~4× from int4. Embeddings are untouched and often dominate small models.

### Cited Findings
**Browser and WebGPU**
- WebGPU spec defaults: `maxStorageBufferBindingSize` 134,217,728 bytes (128 MiB) and `maxBufferSize` 268,435,456 bytes (256 MiB). The binding size must be ≤ maxBufferSize — [gpuweb spec index.bs](https://github.com/gpuweb/gpuweb/blob/main/spec/index.bs); [MDN GPUSupportedLimits](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)
- Dawn (Chrome) reports limit tiers of 128 MB, 1 GB, 2 GB−4 and 4 GB−4, plus a 512 MB tier added for Qualcomm — [Dawn commit](https://dawn.googlesource.com/dawn/+/716339fa9fe10b4383899bbe112479967b585d3c)
- ChromeOS telemetry: 128 MB appears in 100% of reports, 1 GB in 99%, 2 GB−4 in 34% — [web3dsurvey](https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize/platform/Chromium%20OS)
- ONNX Runtime Web — [ORT Web large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html):
  - Chrome's maximum ArrayBuffer is 0x7fe00000 bytes (~2 GB); ORT-Web works around this via `new WebAssembly.Memory()`.
  - Protobuf is limited to 2 GB, so larger models need external data.
  - WebAssembly has a 4 GB limit, and "no way for ONNX Runtime Web to run models larger than 4GB", with WASM64 or direct GPU weight loading named as possible futures.
  - ORT's JS source now contains wasm64-build code paths — [session-options.ts](https://github.com/microsoft/onnxruntime/blob/988cc07aed84347a7d9b94583d15c099e6407fb7/js/web/lib/wasm/session-options.ts)
- Memory64 is supported in Chrome 133+ and Firefox 134+, and not in Safari (through the 26.5 TP). It is part of Wasm 3.0, standardized September 2025 — [caniuse](https://caniuse.com/wf-wasm-memory64); [platform.uno](https://platform.uno/blog/state-of-webassembly-2024-2025/)
- The memory64 spec changes limits to u64 — [WebAssembly/memory64 Overview.md](https://github.com/WebAssembly/memory64/blob/main/proposals/memory64/Overview.md)
- An Emscripten write-up says wasm32 builds are capped at about 2 GB in practice unless configured for 4 GB — [zenn.dev](https://zenn.dev/syoyo/articles/b78c5c3fb607d8)

**WebLLM footprints** (`vram_required_MB`: weights, KV and buffers at default context) — [web-llm src/config.ts](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts)

| Model | Variant | VRAM (MB) | Notes |
|---|---|---|---|
| SmolLM2-135M | q0f16 | 359.69 | |
| SmolLM2-360M | q4f16 | 376.06 | q0f16: 871.99 |
| Qwen2.5-0.5B | q4f16 | 944.62 | q0f16: 1624.12 |
| Qwen3-0.6B | q4f16 | 1403.34 | |
| Llama-3.2-1B-Instruct | q4f16_1 | 879.04 | q0f16: 2573.13; flagged `low_resource_required: true` |
| Qwen3-1.7B | q4f16 | 2036.66 | |
| gemma-2-2b | q4f16 | 1895.3 | |
| Llama-3.2-3B | q4f16 | 2263.69 | |
| Phi-3.5-mini | q4f16 | 3672.07 | 1k-context variant: 2520.07 |
| Llama-3.1-8B | q4f16 | 5001.0 | |

**ExecuTorch WebGPU backend**
- A July 2026 M4 Pro measurement reports 188.3 decode tok/s for a 4-bit Llama 3.2 1B at 128-token context — [ExecuTorch success-stories.md](https://github.com/pytorch/executorch/blob/ad1c24d648b5a12bb51d488573fab94d83dac39e/docs/source/success-stories.md)

**Phones**
- Apple: the increased-memory-limit entitlement is "only available on some device models", and apps should check `os_proc_available_memory` — [Apple entitlement doc](https://developer.apple.com/tutorials/data/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit.md)
- Developer forum reports (anecdotal):
  - About 6 GB per-app ceiling on iPhone 16 Pro Max (8 GB) and 17 Pro Max (12 GB) even with the entitlement — [Apple forum 834805](https://developer.apple.com/forums/thread/834805)
  - About 15 GB on an iPad dev build vs about 6 GB from the App Store — [Apple forum 770868](https://developer.apple.com/forums/thread/770868)
  - Metal `recommendedMaxWorkingSetSize` vs RAM causes LLM load failures — [Apple forum 805161](https://developer.apple.com/forums/thread/805161)
- All iPhone 16 models have 8 GB RAM — [aphnetworks](https://aphnetworks.com/news/29221-apple-exec-confirms-all-iphone-16-models-will-have-8gb-ram)
- Gemma 3n Per-Layer Embeddings (PLE) are "cacheable to fast local storage", reducing accelerator memory: E2B has ~5B total parameters vs ~2B in accelerator memory (secondary source). This is a production precedent for keeping per-layer parameters outside the hot working set — [Gemma 3n docs](https://ai.google.dev/gemma/docs/gemma-3n); [InfoQ](https://www.infoq.com/news/2025/07/gemma-3n-architecture)

**Jetson** — [TRT Edge-LLM performance-benchmarks.md v0.10.0](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/performance/performance-benchmarks.md)
- Orin Nano 8GB, Orin NX 16GB and AGX Orin 64GB are the benchmarked Orin targets.
- Measured peak GPU memory, batch 1, INT4 AWQ:

| Model | Orin Nano 8GB | Orin NX 16GB |
|---|---|---|
| Qwen3-0.6B | 1,917 MB | 1,914 MB |
| Qwen3-1.7B | 2,992 MB | 2,999 MB |
| Qwen3.5-2B | 3,642 MB | — |
| Qwen3-8B | — | 7,167 MB |

- Decode at 2,048 past-KV (`llm_bench`, batch 1, INT4 AWQ):

| Model | Orin Nano | Orin NX | AGX Orin |
|---|---|---|---|
| Qwen3-0.6B | 61.0 tok/s | 98.9 tok/s | 162.5 tok/s |
| Qwen3-1.7B | 32.3 tok/s | 52.2 tok/s | 89.5 tok/s |
| Qwen3-8B | — | 17.5 tok/s | 30.6 tok/s |

### Inferences
**Download-size arithmetic** (mine)
- A 1B-parameter model in int4 is about 0.5–0.6 GB of weights. Looping K=4 unique blocks over N=16 passes divides block weights by 4.
- Small models are embedding-heavy. A 128K vocab × 2048 dimensions ≈ 262M parameters, about 131 MB in int4 if tied. So total shrinkage is less than K/N; for example a 1B model might go from about 0.55 GB to about 0.25–0.3 GB.
- Sub-1-bit-per-param effective budgets are realistic with rank-16 deltas, per Q2.
- All of this sits well under the 2 GB ArrayBuffer and 4 GB wasm32 limits, removing the need for Memory64 (unsupported in Safari).

**WebGPU binding limits**
- With per-tensor buffers, int4 matrices of 1B–3B models are well below 128 MiB per binding. The 1 GB request by MLC concerns packed parameter shards or KV buffers.
- Sharing makes it easier to fit in low-tier (128 MiB / 256 MiB) mobile adapters, because fewer, smaller buffers are needed. KV per pass still scales with passes × context (see Q3).

**Where RAM goes**
- On phones with a ~6 GB app ceiling, and on Orin Nano 8 GB, which shares memory with the OS/CPU, the measured footprints show runtime buffers and KV are a large share at small model sizes.
- For example, Qwen3-0.6B INT4 is about 0.3–0.4 GB of weights but about 1.9 GB peak on Orin. So weight sharing alone yields a modest fraction of total RAM savings; KV and workspace reductions matter as much.

### Gaps
- No official Apple or Google per-app memory budget for LLMs is published. Android per-app limits (largeHeap, low-memory killer) were not researched to a primary source.
- Practical maximum `WebAssembly.Memory` with Memory64 per browser is undocumented. The "16 GB" figure appears only in low-quality sources.
- Safari/WebKit WebGPU limit tiers on iOS were not found.

## Q6. Vision and video transformers on Jetson: per-frame compute and TensorRT INT8 on ViTs

### Takeaway
Per-frame ViT encoding is compute-bound, so looping or weight sharing does not reduce per-frame latency: FLOPs are unchanged. Sharing helps only with storage and with switching between models. INT8 on Orin gives mixed results for ViTs:
- An RF-DETR transformer detector gains 1.26× (14.83 → 11.77 ms) on Orin Nano Super.
- A ViT-S/DPT encoder gained nothing in isolation (30.75 → 30.34 ms), and the full encoder was 2.7× slower with ModelOpt INT8.
- FP16 ViT-B/16 at 224 takes about 17–20 ms on Orin Nano.

TRT Edge-LLM's measured VLM vision-encoder times on Orin are about 25–141 ms per image (265 tokens).

### Cited Findings
- FastPose-ViT on Orin Nano: ViT-B/16 at 224×224 takes about 17–20 ms for the ViT alone and about 75 ms end-to-end (~13 FPS) sequentially, up to about 33 FPS when pipelined at batch 1. FP16 helped at 384×384 (49.2 → 40.7 ms) but not at 224 — [arXiv 2512.09792](https://arxiv.org/pdf/2512.09792)
- An NVIDIA forum bug report on Orin Nano 4GB with a ViT-S + DPT encoder:
  - Isolated ViT-S: 30.75 ms in FP16 vs 30.34 ms with ModelOpt INT8.
  - Full encoder: 114.5 ms with ModelOpt INT8 vs 41.8 ms in FP16, a 2.7× regression; one ModelOpt version failed to compile.
  - Source: [NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)
- Embedl RF-DETR Base on Orin Nano Super: TensorRT FP16 14.83 ms vs INT8 11.77 ms (1.26×), GPU-only with clocks locked — [HF embedl/rf-detr-base](https://huggingface.co/embedl/rf-detr-base)
- A Jetson Thor forum thread reports "Poor INT8 performance" — [NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-thor-agx-poor-int8-performance/363884)
- TRT Edge-LLM v0.10.0 vision-encoder times (265 image tokens per run, FP16 ViT, INT4 AWQ LLM) — [performance-benchmarks.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/performance/performance-benchmarks.md):

| Vision encoder (model) | Orin NX 16GB | Orin Nano 8GB |
|---|---|---|
| Qwen3-VL-2B | 77.1 ms | 141.3 ms |
| Qwen3-VL-8B | 110.5 ms | — |
| Qwen3.5-0.8B | 25.1 ms | 45.4 ms |
| Qwen3.5-2B | 72.6 ms | 133.4 ms |

  - VLM prefill on Orin Nano for Qwen3-VL-2B: 296 tokens in 167.2 ms.
- On Orin, TensorRT Edge-LLM supports only FP16, INT8 and INT4 (no FP8/FP4). The INT4 runs use "externalized INT4 weights" — [support-matrix.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/getting_started/support-matrix.md); [performance-benchmarks.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/performance/performance-benchmarks.md)
- Orin AI throughput (sparse INT8 TOPS): Orin Nano 8GB 40 TOPS (67 TOPS Super), Orin NX 16GB 100 TOPS (157 Super), AGX Orin 64GB about 275 TOPS — [Connect Tech module comparison](https://connecttech.com/jetson/jetson-module-comparison/)

### Inferences
- **Weight sharing and per-frame compute.**
  - A ViT patch encoder processes hundreds of tokens per frame, which makes it compute-bound like LLM prefill. A looped ViT with the same effective depth costs the same per frame.
  - To cut per-frame compute, the levers are fewer passes per frame (adaptive depth, early exit), fewer tokens (TRT Edge-LLM has a visual-token-pruning feature), or INT8/INT4 compute — not sharing.
- **Video.** Temporal redundancy suggests running fewer loop passes on frames that change little (MoR-style per-token depth applied per patch or per frame). This is speculative and has no Jetson measurements.
- **Jetson weight residency.** With a 4 MB L2 and a ViT-B block of about 7M parameters (~7 MB in INT8), a shared ViT-B block will not stay in L2 across passes. The memory benefit on Jetson is footprint, which matters most on the 8 GB Orin Nano shared with the OS and camera pipeline.
- **INT8 on Orin.** Results are kernel- and graph-dependent: Q/DQ placement around LayerNorm, Softmax and GELU often blocks fusion. Plan to benchmark per model with trtexec.

### Gaps
- I found no clean published benchmark of a plain ViT (e.g., ViT-B/16, DeiT) with TensorRT INT8 vs FP16 on Orin, and no measurement of a looped or weight-shared ViT on any Jetson.
- I found no data on whether TensorRT's ViT tactics change when blocks share weights. Engines are built per layer, so expect no runtime speedup.
