# Precision and Edge Hardware: Making Each Looped-Transformer Pass Cheaper (phones, browsers, Jetson)

Research date: 2026-10-12. Source-type tags: **[primary]** = vendor docs, runtime source code or benchmark tables read directly from the GitHub repos (cloned 2026-10-12); **[search excerpt]** = numbers seen only in web-search snippets (arxiv.org, nvidia.com, developer.chrome.com, developers.googleblog.com, forums.developer.nvidia.com and the Qualcomm and Apple sites could not be fetched through the proxy); **[secondary]** = blogs, reviews, community benchmarks and forum posts. Peak TOPS are labelled as peak. Measured end-to-end numbers say which device, runtime and model they come from.

---

## Q1. Weight-only vs weight+activation quantization: which actually reduces compute time (W8A8, W4A8, W4A4, FP8/MXFP8/MXFP4/NVFP4), and what are the measured speedups at batch 1 vs prefill/vision?

### Takeaway
Weight-only quantization (W4A16, INT4 AWQ, palettization) speeds up memory-bound batch-1 decode. It does almost nothing for compute-bound prefill or per-frame ViT work, and on some stacks it makes prefill slower. Only quantizing the activations as well moves the matmuls onto faster math units (INT8/FP8/FP4 tensor cores, the ANE int8 path, Hexagon fixed point). The clearest edge measurement is on Jetson Thor: NVFP4 (W4A4) prefill is **3–6.3× faster than INT4 AWQ (W4A16) on the same model**, while decode speed is identical.

### Cited Findings
**Same-model comparisons on edge hardware (the best evidence)**
- **Jetson AGX Thor, TensorRT Edge-LLM v0.10.0, `llm_bench`, BS=1, 2,048-token prefill / 2,048 past-KV decode [primary]:**
  - Qwen3-8B: prefill INT4 AWQ 3,087 tok/s vs NVFP4 19,517 tok/s (**6.3×**); decode 46.4 vs 44.6 tok/s (same).
  - Qwen3-4B-Instruct-2507: prefill 5,617 vs 33,030 tok/s (**5.9×**); decode 76.5 vs 73.4.
  - Qwen3-1.7B: prefill 14,082 vs 73,858 (**5.2×**); decode 135.1 vs 135.5.
  - Qwen3-0.6B: prefill 37,350 vs 114,242 (**3.1×**); decode 253 vs 244.
  - Qwen3.5-27B: prefill 870 vs 4,564 (**5.2×**); decode 15.2 vs 14.7.
  - — [TensorRT-Edge-LLM performance-benchmarks.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md) (JetPack 7.2, CUDA 13.2, TensorRT 10.16; doc commit 2026-10-06)
- **Same source, gemma-4-31B-it on Thor, BS=1 [primary]:**
  - FP8 (W8A8) prefill 1,976 tok/s vs INT4 AWQ 706 tok/s: FP8 is **2.8× faster** at prefill.
  - At decode INT4 AWQ is faster: 12.8 vs 7.4 tok/s (1.7×), because it moves fewer weight bytes.
  - The NVFP4 build gave 2,602 tok/s prefill and 7.7 tok/s decode.
  - — [TensorRT-Edge-LLM benchmarks](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md)
- **NVIDIA Model Optimizer docs [primary]:** "INT4 AWQ is an INT4 weight only quantization … particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself." They also say NVFP4 "can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint." — [Model-Optimizer examples/onnx_ptq/README.md](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md)
- **Apple Core ML, measured on iPhone 15 Pro (A17 Pro), iOS 17, Xcode 15, Xcode Performance-tab median, compute units = all [primary]:**

  | Model | FP16 | Weight-only (8-bit) | W8A8 | W8A8 vs FP16 |
  |---|---|---|---|---|
  | ResNet50 | 1.38 ms | 1.50 ms (slower) | 0.77 ms | 1.79× faster |
  | MobileViTv2-1.0 (hybrid ViT) | 1.36 ms | 1.37 ms | 1.03 ms | 1.32× faster |
  | MobileNetV2 | 0.49 ms | 0.44 ms | 0.20 ms | 2.45× faster |

  On iPhone 14 Pro (A16), W8A8 gave ResNet50 1.52→0.94 ms and MobileViTv2 1.38→1.18 ms. — [coremltools opt-quantization-perf.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-quantization-perf.md)
- **Apple's guidance [primary]:**
  - Activation quantization can cause "a considerable slowdown … for the compute units (CPU and sometimes GPU) that employ load-time weight decompression". Apple therefore recommends it "only when your model is fully or mostly running on the Neural Engine."
  - W8A8 can "lead to considerable latency benefits on the Neural Engine by leveraging the faster int8-int8 compute path supported in newer hardware (A17 pro, M4)."
  - — [coremltools opt-overview.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-overview.md)
- **Qualcomm Hexagon NPU, Qualcomm AI Hub, Llama-3.2-3B-Instruct via Genie, context 4096, Galaxy S25 (Snapdragon 8 Elite) [primary]:**
  - "w4" (4-bit weights, **fp16 activations**): prefill 396 tok/s, decode 14.3 tok/s.
  - "w4a16" (4-bit weights, **int16 fixed-point activations**): prefill 728 tok/s, decode 26.5 tok/s.
  - So quantizing activations to int16 gave **1.84× prefill and 1.85× decode**. On the Galaxy S26 the figures were 443→776 prefill (1.75×) and 15.6→28.2 decode (1.81×).
  - — [ai-hub-models llama_v3_2_3b_instruct/perf.yaml](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/llama_v3_2_3b_instruct/perf.yaml); precision definitions ("w4 + w8 (few layers) with fp16 activations and w4a16 + w8a16") in [manifest.yaml](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/llama_v3_2_3b_instruct/manifest.yaml) (repo commit 2026-10-12)
- **Mobile CPU, W4A8 with dynamic int8 activations [primary]:**
  - Setup: ExecuTorch + XNNPACK/KleidiAI on a OnePlus 12, 64-token prompt.
  - Llama-3.2-1B, BF16 → SpinQuant: prefill 60.3→260.5 tok/s (**4.3×**); decode 19.2→50.2 tok/s (2.6×).
  - Llama-3.2-3B: prefill 21.2→89.7 tok/s (**4.2×**); decode 7.6→19.7 tok/s (2.6×).
  - SpinQuant here is "4-bit weight … and 8-bit dynamic activation quantization."
  - — [ExecuTorch examples/models/llama/README.md](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md); [SpinQuant README](https://github.com/facebookresearch/SpinQuant)

**Datacenter reference points for each scheme (not edge, but they show what each scheme can do)**
- **SmoothQuant (W8A8):** "up to 1.56x speedup and 2x memory reduction" in FasterTransformer [primary]. — [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
- **QServe/QoQ (W4A8KV4) [primary]:**
  - Existing INT4 methods "suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs." They "only accelerate low-batch, edge LLM inference."
  - QServe's throughput gain over TensorRT-LLM: Llama-3-8B 1.2× on A100 and 1.4× on L40S; Qwen1.5-72B 2.4× on A100 and 3.5× on L40S.
  - — [mit-han-lab/omniserve README](https://github.com/mit-han-lab/omniserve)
  - Separately, QServe reports 2.5–2.9× higher throughput than Atom and QuaRot on A100 [search excerpt]. — [QServe arXiv 2405.04532](https://arxiv.org/abs/2405.04532)
- **Atom (W4A4):** "improves end-to-end throughput by up to 7.73× compared to the FP16 and by 2.53× compared to INT8 quantization, while maintaining the same latency target" (serving setting) [primary]. — [efeslab/Atom](https://github.com/efeslab/Atom)
- **QuaRot (W4A4KV4):** "all matrix multiplications are performed in 4-bits"; LLaMa2-70B loses at most 0.29 WikiText perplexity and keeps 99% of zero-shot performance [primary]. I could not verify the commonly quoted 3.33× prefill speedup (arXiv not reachable). — [spcl/QuaRot](https://github.com/spcl/QuaRot)
- **SpinQuant (W4A4KV4):** narrows the gap to full precision "to merely 2.9 points for the LLaMA-2 7B model on zero-shot reasoning tasks" [primary]. — [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)

**Format support on hardware**
- **NVFP4** needs SM100+ (Blackwell, which includes Thor at SM110). **FP8** needs SM89+. INT4 AWQ/GPTQ run on all platforms [primary]. — [Edge-LLM benchmarks precision key](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md)
- **Edge-LLM's Thor guidance:** "NVFP4 is the recommended precision for Thor (SM110+)" [search excerpt]. Do "not select FP8, MXFP8, FP4, or NVFP4 checkpoints for Orin" [search excerpt]. — [Jetson AI Lab Edge-LLM tutorial](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/); [Edge-LLM supported models](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/supported-models.html)

### Inferences
- **For a looped model, the regime decides which quantization helps:**
  - Batch-1 LLM decode is weight-bandwidth bound. Each loop pass re-reads the shared block's weights, unless they stay resident on-chip. Weight-only 4-bit is enough there; activation precision barely matters (Thor AWQ ≈ NVFP4 at decode).
  - Prefill and per-frame ViT encoding are compute bound. Only W8A8/FP8/W4A4-class schemes, or int16 fixed point on Hexagon, reduce the time per pass.
  - The user's "int8 work" mattered for compute because it quantized activations. That is consistent with this evidence.
- **Where activation quantization should not be done:** the CPU and GPU paths that dequantize weights at load time (Core ML CPU/GPU, Metal WebGPU). There it adds Q/DQ overhead without faster math.

### Gaps
- I found no same-device, same-model edge measurement of W4A8 vs W8A8 vs W4A4 for a ViT encoder. The only per-frame ViT evidence is on the ANE (Core ML table), on Hexagon (Q4) and in Jetson forum anecdotes (Q3).
- I could not open the QuaRot, Atom and QServe papers to confirm their batch-1 decode vs prefill breakdowns (arXiv blocked).

---

## Q2. Looped-model quantization specifically (LoopQ, "Quantizing Looped Transformers", Loopy, recursive reasoners at 4-bit activations), and what per-loop scales cost at runtime

### Takeaway
Four 2026 papers agree that quantization error compounds across passes of a shared block:
- One activation scale shared by all loops fits poorly.
- Calibrating only on the first pass misses directions that matter later.
- 4-bit activations with per-tensor scales can collapse weight-tied reasoners to 0% accuracy, while per-block (MX) scaling or 8-bit activations recover.

Per-loop static scales cost O(loops × layers) scalars, which is negligible memory. On static-graph NPUs, however, they effectively require unrolling the loop. Per-token dynamic scales (Loopy) need runtime reductions that GPUs and CPUs support but most NPUs do not.

All accuracy results are simulated quantization. None of these papers report edge-device latency.

### Cited Findings
- **LoopQ (Fang et al., arXiv 2605.16343, May 2026) [search excerpt]:**
  - What it does: post-training quantization (PTQ) for looped/recursive transformers that "preserves a shared quantized backbone while introducing lightweight loop-dependent adaptations."
  - Components: loop-aware activation scaling ("a shared activation scale cannot fit all loops when activation magnitudes vary", which the paper calls distribution shift across loops), selective loop-aware transforms, a cross-loop transition/state-alignment adapter, and trajectory-aware calibration that suppresses "trajectory-level error accumulation".
  - Overhead: the adaptations touch "<5% of transformation parameters, corresponding to <0.12% of total model parameters". The loop-dependent scales add "only O(TL) scalars" (T loops, L layers), described as "minimal memory and deployment cost".
  - Headline results vs the strongest static PTQ baseline (baselines include QuaRot and SpinQuant): W4A4 average downstream accuracy +68.8% and average perplexity −87.7%; W4A8 accuracy +9.8% and perplexity −17.1%.
  - No measured latency was reported in anything I could see.
  - — [arXiv 2605.16343](https://arxiv.org/abs/2605.16343)
- **"Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness" (arXiv 2609.30820, Sep 2026) [search excerpt]:**
  - **Feedback exposure:** a quantized layer perturbs the recurrent state with no identity path, so the error is fed back at later steps.
  - **Calibration blindness:** one-step GPTQ estimates its Hessian from first-step activations, so input directions that matter at later recurrence steps get little weight.
  - One-step GPTQ was *worse than round-to-nearest (RTN)* on the main metric for 5 of 9 checkpoints.
  - Accumulating the GPTQ Hessian across all recurrence steps beats both one-step GPTQ and RTN on all 9 checkpoints (seven looped architectures, including Ouro and LoopFormer). It recovers bf16-level accuracy on Huginn.
  - — [arXiv 2609.30820](https://arxiv.org/abs/2609.30820)
- **Loopy (Li et al., arXiv 2610.05265, Oct 2026) [search excerpt]:**
  - Problem statement: "errors introduced by a quantized shared core affect subsequent cores"; "quantization configuration candidate rankings can change with recurrent depth", so configurations are selected at the target deployment depth.
  - Uses **per-token dynamic activation scales** rather than per-loop static ones.
  - Evaluated on Ouro, Huginn-0125 and Recurrent-Llama at W4A8 and W4A4. On Ouro-1.4B W4A4, LAMBADA perplexity is 36.5% lower than adapted SpinQuant. It beats LoopQ on W4A4 WikiText-2 perplexity and on 5-task average accuracy on both Ouro checkpoints.
  - — [arXiv 2610.05265](https://arxiv.org/abs/2610.05265v1)
  - A secondary write-up says that on Huginn (32 loops) "four of five adapted baselines collapsed", and that Loopy's speed figures come from a datacenter GPU. [secondary] — [smallaimodel.substack.com](https://smallaimodel.substack.com/p/quantize-at-the-depth-you-run)
- **"Quantizing Recursive Reasoning Models" (Ingolfsson et al., ETH Zürich, arXiv 2607.16237, Jul 2026) [search excerpt]:**
  - Models: HRM/TRM-style weight-tied recursive reasoners.
  - Collapse: per-tensor 4-bit activation quantization drops Sudoku exact accuracy from 84.1% to **0.0%**. The authors attribute this to activation-scale *granularity*, not bit-width or format.
  - Fix: per-block MXInt4 gives Sudoku 80.1% and Maze 84.7%; MXFP4 gives 72.7% and 85.1%. Integer and float block formats are about equal.
  - 8-bit integer is a small perturbation. Weight-only 4-bit costs "a few points".
  - The error "concentrates in the most-reused recursive layers and scales with reuse depth". The equilibrium model (EqR) is the most sensitive.
  - — [arXiv 2607.16237](https://arxiv.org/abs/2607.16237)
- **"What Survives When You Compress a Recursive Reasoner for the Edge?" (arXiv 2606.26488) [search excerpt]:** argues that quantization noise compounds at each carry-state update across recursive cycles, a failure mode distinct from single-pass transformers. It frames this as an open hypothesis. — [arXiv 2606.26488](https://arxiv.org/pdf/2606.26488)
- **Hyperloop Transformers (arXiv 2604.21254) [search excerpt]:** looped models "can be more sensitive to low-precision quantization". The Hessian for a looped layer aggregates activations from all loops. — [arXiv 2604.21254](https://arxiv.org/pdf/2604.21254)
- **Related, KV-side:** ResidualQuant (arXiv 2610.10381) quantizes the KV cache of looped transformers with 2-bit residuals [search excerpt, title only]. — [arXiv 2610.10381](https://arxiv.org/html/2610.10381)
- **How each runtime handles activation scales [primary]:**
  - ONNX Runtime WebGPU's DP4A path quantizes activations on the fly. A quantize kernel produces per-block A-scales of shape `{batch, 1, M, K/kBlockSizeA}`, so dynamic per-block activation scales are a runtime feature on GPU. — [onnxruntime dp4a_matmul_nbits.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.cc)
  - ExecuTorch/XNNPACK uses "8-bit dynamic activation" for W4A8. — [SpinQuant README](https://github.com/facebookresearch/SpinQuant)

### Inferences
- **Runtime cost of per-loop static scales (LoopQ-style):**
  - On GPU and CPU (TensorRT, XNNPACK, WebGPU), each pass's GEMM receives a different scalar or vector, so the cost in FLOPs or latency is essentially zero.
  - On static-graph NPUs (QNN/HTP, Core ML ANE, LiteRT int8, DLA), activation encodings are compile-time tensor properties. A single "rolled" shared block called k times can carry only one encoding. Per-loop scales therefore force an unrolled graph with k copies of the block's activation encodings. The weights can still be shared through deduplication (Q6).
  - Conclusion: per-loop scales are free in compute but cost graph size and compile complexity on NPUs.
- **Runtime cost of per-token dynamic scales (Loopy-style):**
  - Each GEMM input needs an absmax reduction, which is cheap and fused on GPU and CPU (ORT WebGPU, XNNPACK).
  - These do not map cleanly onto NPUs that use static fixed-point encodings (Hexagon int16/int8, Core ML W8A8 with constant scales). On phones, Loopy-style schemes fit CPU/GPU paths; LoopQ-style static per-loop scales fit NPU paths once unrolled.
- **4-bit activations for looped models need block scaling:**
  - Block-scaled hardware formats (MXFP4/NVFP4) exist on Jetson Thor (Blackwell).
  - In this device set, W4A4 for a looped model is realistic only on Thor. Orin, phone NPUs and WebGPU should target 8-bit (or Hexagon int16) activations.
- **Calibration for the user's distillation pipeline:** collect calibration data from every pass (A→B→A→B…), not just the first. Accumulate per-block Hessians or activation statistics across all passes that reuse the block. Evaluate at the deployed loop count, because the best configuration changes with depth.

### Gaps
- I found no measured on-device latency for LoopQ, Loopy or the Hessian-accumulation method. All reported accuracy is simulated quantization.
- I found no per-loop error-growth curves; arXiv full texts were blocked, so I saw only abstract-level claims.
- I found no study of looped *vision* encoders (per-frame ViT) under quantization. All looped-quantization papers are on LMs or puzzle reasoners.

---

## Q3. NVIDIA Jetson Orin and Thor: INT8 vs FP16 TOPS, DLA constraints, 2:4 sparsity, TensorRT INT8 for ViTs (including regressions), TensorRT-LLM / TensorRT Edge-LLM precision support

### Takeaway
- **Orin (Ampere SM87)** has INT8 tensor cores at about 2× FP16 peak and no FP8/FP4. Edge-LLM supports only FP16, INT8 and INT4 on Orin, and every published Orin LLM row is INT4-AWQ weight-only.
- **The DLA** is a CNN engine: INT8/FP16 only, with ViT attention not running on it.
- **2:4 sparsity** delivers 1.24–1.47× on DLA CNNs. On the GPU, users report zero to marginal gain.
- **TensorRT INT8 for ViTs on Orin and Thor** is frequently a no-op or a regression, because of Q/DQ fusion and softmax/LayerNorm costs.
- **Thor (Blackwell SM110)** is where 4-bit compute pays off: NVFP4 prefill is 3–6× faster than INT4 AWQ.

### Cited Findings
**Peak hardware numbers (peak, not measured)**
- **AGX Orin 64GB:** 275 sparse INT8 TOPS total = GPU 171 sparse INT8 TOPS + 2× DLA 105 sparse INT8 TOPS (MAXN). In 15 W mode: GPU 14 and DLA 40 sparse TOPS, so the DLA share is 38% at MAXN and 74% at 15 W [primary]. — [NVIDIA Deep-Learning-Accelerator-SW README](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW)
- **AGX Orin dense:** 138 TOPS for the whole module [secondary]; the GPU alone is about 85 dense INT8 TOPS and about 43 dense FP16 TFLOPS [secondary]. — [Hackster](https://www.hackster.io/news/nvidia-launches-275-tops-jetson-agx-orin-developer-s-kit-at-1-999-bbb5ff80e050); [flopper.io](https://flopper.io/gpu/nvidia-jetson-agx-orin-64gb)
- **Orin Nano 8GB Super:** 67 sparse / 33 dense INT8 TOPS, 17 FP16 TFLOPS, no DLA [search excerpt]. **Orin NX 16GB Super:** 157 sparse / 78 dense TOPS total; GPU 77 sparse / 38 dense; DLA 80 sparse / 40 dense; GPU FP16 about 19 TFLOPS [search excerpt]. — [NVIDIA JetPack 6.2 Super Mode blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules); [Connect Tech](https://connecttech.com/jetson-orin-nx-nano-super-mode/)
- **Jetson AGX Thor T5000 [search excerpt]:**
  - Compute: 2,070 FP4 TFLOPS sparse (1,035 dense); FP8 1,035 sparse (517 dense).
  - Other specs: 2,560 Blackwell cores, 128 GB LPDDR5X at 273 GB/s, 40–130 W.
  - A Transformer Engine switches between FP4 and FP8.
  - T4000: 1,200 FP4 TFLOPS sparse.
  - — [NVIDIA Jetson Thor blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/); [flopper.io T5000](https://flopper.io/gpu/nvidia-jetson-agx-thor-t5000-128gb)

**Edge-LLM precision support [primary]**
- Support matrix: "Jetson Orin … FP16, INT8, and INT4 only … Jetson Orin does not run FP8 or FP4 model engines." Thor (JetPack 7.0–7.2, SM110) is "model-dependent". — [Edge-LLM support-matrix.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md)
- The quantizer offers NVFP4 for the LLM and LM head, an FP8 KV cache, and FP8 for the **visual tower** (`--visual_quantization fp8`, calibrated on MMMU). — [Edge-LLM quantization.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/quantization.md)

**Measured Edge-LLM v0.10.0 numbers, INT4 AWQ, BS=1, 2,048-token prefill [primary]**
| Device | Model | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| AGX Orin 64GB | Qwen3-8B | 1,480 | 30.6 |
| AGX Orin 64GB | Qwen3-1.7B | 6,644 | 89.5 |
| AGX Orin 64GB | Qwen3-0.6B | 17,277 | 162.5 |
| Orin NX 16GB | Qwen3-8B | 696 | 17.5 |
| Orin NX 16GB | Qwen3-1.7B | 3,039 | 52.2 |
| Orin Nano 8GB | Qwen3-1.7B | 2,006 | 32.3 |
| Orin Nano 8GB | Qwen3-0.6B | 4,601 | 61.0 |

- On the same model and the same INT4 AWQ precision, Thor is about 2.1× AGX Orin on prefill (3,087 vs 1,480 tok/s for Qwen3-8B) and about 1.5× on decode (46.4 vs 30.6).
- With NVFP4, Thor prefill is about 13× AGX Orin's INT4 AWQ (19,517 vs 1,480).
- — [Edge-LLM benchmarks](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md)

**Measured ViT encoder time, Qwen3-VL-2B visual encoder at FP16, about 265 image tokens per run, Edge-LLM runtime dashboard [primary]**
| Device | ViT time per run |
|---|---|
| Thor | 11.7 ms |
| AGX Orin | 36.2 ms |
| Orin NX | 77.1 ms |
| Orin Nano | 141.3 ms |

The ViT runs at FP16 in all published rows. — [Edge-LLM benchmarks](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md)

**DLA**
- "NVIDIA DLA hardware is a fixed-function accelerator engine targeted for deep learning operations … full hardware acceleration of convolutional neural networks". Orin has up to two second-generation DLAs. Reference networks are all CNNs (ResNet-50, RetinaNet, SSD) [primary]. — [Deep-Learning-Accelerator-SW](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW)
- DLA supports FP16 and INT8; INT8 on DLA requires *implicit* quantization, which TensorRT 10 deprecates [search excerpt, NVIDIA forum]. — [NVIDIA forum](https://forums.developer.nvidia.com/t/is-there-a-plan-to-support-dla-on-the-next-tensorrt-version/313130)
- Transformer and ViT attention layers do not run on DLA as of JetPack 6.2 [secondary]. Attention's 3-D (B×S×C) tensors are "incompatible with the DLA's fixed 4D addressing scheme" [search excerpt]. LayerNorm and non-channel softmax force GPU partitions [secondary]. — [ProventusNova blog](https://proventusnova.com/blog/tensorrt-vs-dla-jetson-orin/); [arXiv 2607.10942](https://arxiv.org/pdf/2607.10942)

**2:4 sparsity**
- DLA dense→sparse speedups measured on AGX Orin with JetPack 5.1.1, all CNNs [primary]: RetinaNet ResNeXt-50 1.31× (MAXN) to 1.43× (15 W); ResNet-34 backbone 1.35–1.45×; SSD-ResNet-34 1.24–1.36×. Lower power modes give larger gains, because more layers become math-bound. — [Deep-Learning-Accelerator-SW](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW)
- On the GPU: an Orin user saw identical latency (about 22 ms) with `--sparsity=enable/force` on a 2:4-pruned YOLOv7 under TensorRT 8.5.2 [secondary]. Typical 2:4 speedups on GPUs are 1.1–1.3×, not 2× [search excerpt]. — [NVIDIA forum](https://forums.developer.nvidia.com/t/stuctured-sparsity-2-4-does-not-improve-inference-performance-on-jetson-orin/263685); [V:N:M sparsity paper, arXiv 2410.16135](https://arxiv.org/pdf/2410.16135v2)

**TensorRT INT8/FP8 for ViTs: measured regressions [secondary, forum]**
- ViT-S + DPT on Orin Nano 4GB: the Model Optimizer INT8 engine took 114 ms vs about 42 ms for FP16, a **2.7× regression**. The isolated ViT-S gained only about 1.3% with INT8; trtexec `--fp16 --int8` gave about 1%. — [NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)
- Jetson Thor: INT8 "always ends up about the same or worst than the FP16" on RT-DETR and Swin-UNet with both implicit and explicit quantization. Another user reports no INT8 gain on Thor vs about 37% on AGX Orin. NVIDIA staff acknowledged an FP8/INT8 performance report on Thor. — [NVIDIA forum 353308](https://forums.developer.nvidia.com/t/jetson-thor-int8-quantization-show-no-performance-gain-over-fp16/353308); [NVIDIA forum 363884](https://forums.developer.nvidia.com/t/jetson-thor-agx-poor-int8-performance/363884)
- Thor ViT at FP8: Q/DQ insertion is needed to trigger the MHA fusion, and base ViT then gains only about 20% latency. In NVIDIA's own test, EfficientSAM went from 140.7 to 166.3 qps at FP8. NVIDIA staff say the attention problem size is "bound by the softmax throughput, which is done in FP32. Hence, FP16 and FP8 are not far apart." — [NVIDIA forum 349329](https://forums.developer.nvidia.com/t/low-vit-performance-gain-on-jetson-thor-using-fp8-vs-fp16/349329); [NVIDIA forum 360866](https://forums.developer.nvidia.com/t/jetson-thor-int8-quantization-show-no-performance-gain-over-fp16-2/360866)
- Other unfused-Q/DQ cases:
  - FasterViT INT8 1.46 ms vs FP16 1.36 ms on a T4 [secondary]. — [TensorRT issue #3186](https://github.com/NVIDIA/TensorRT/issues/3186)
  - QAT-INT8 still 1.66× slower than FP16 on TensorRT 10.7 because of unfused Q/DQ (CNN) [search excerpt]. — [arXiv 2607.26238](https://arxiv.org/pdf/2607.26238)

### Inferences
- **On Orin, INT8 GPU tensor-core peak is 2× FP16**, but ViT-scale encoders realize close to 1.0× unless Q/DQ placement lets TensorRT fuse INT8 MHA and GEMM+bias+GELU. Budget 1.0–1.4× for a per-frame ViT encoder on Orin GPU from INT8. This is consistent with the forum data; I found no NVIDIA-published Orin ViT INT8 table.
- **The DLA is not a viable target for transformer blocks.** Use it only for convolutional stems or CNN heads running alongside, freeing the GPU.
- **Orin LLM compute** is limited to INT8 (SmoothQuant-style W8A8) or INT4 weight-only. Edge-LLM publishes no W8A8 Orin rows, so the INT8 prefill benefit on Orin is unmeasured in the official tables.
- **Thor** is the only Jetson where 4-bit activations (NVFP4) are a real compute lever (3–6× prefill vs W4A16). Per Q2, NVFP4's block scaling is exactly what looped and recursive models need to avoid 4-bit collapse.

### Gaps
- I could not open NVIDIA's official TensorRT DLA layer-support tables (docs.nvidia.com blocked), so the DLA claims about softmax, LayerNorm and MatMul rest on secondary sources.
- I found no NVIDIA-published Orin INT8 vs FP16 ViT benchmark table.
- I found no Edge-LLM INT8 (W8A8 SmoothQuant) Orin rows to quantify INT8 prefill gains on Orin.
- I found no measured 2:4-sparse transformer GEMM on the Orin GPU.

---

## Q4. Phones: Apple Neural Engine, Qualcomm Hexagon (QNN/Genie), MediaTek APU, Google Tensor/LiteRT — precision support and measured NPU vs GPU numbers

### Takeaway
- Phone NPUs are fixed-point engines. Apple's ANE gained a faster int8×int8 path from A17 Pro/M4. Hexagon runs LLMs as w4a16 (int16 activations), roughly 1.8× faster than fp16 activations. FP16 matmul on older mobile NPUs is very slow.
- Measured NPU prefill is 2–10× GPU prefill on Snapdragon and MediaTek. Decode gains are smaller because decode is bandwidth bound.
- ViT results are mixed: on Hexagon, AI Hub's own w8a8 ViT-B/16 is *slower* than fp16, while w8a16 is 1.4× faster.

### Cited Findings
**Apple Neural Engine**
- **Peak:** A17 Pro ANE is 35 TOPS INT8 vs A16 at 17 TOPS. The precisions likely differ (A16 figure possibly FP16/INT16), so the doubling is partly a change in how TOPS are counted [secondary/search excerpt]. A19 Pro adds "Neural Accelerators" in the GPU, with no published TOPS [search excerpt]. — [Notebookcheck A17 Pro](https://www.notebookcheck.net/Apple-A17-Pro-Processor-Benchmarks-and-Specs.756287.0.html); [Medium analysis](https://medium.com/@willrarcher/whats-wrong-with-apple-a17-pro-and-a18-series-3b3e3cb4a714); [Mac Observer](https://www.macobserver.com/news/a20-pro-vs-a19-pro-a18-pro-a17-pro-core-counts/)
- **Measured W8A8 vs FP16:** see Q1 (iPhone 15 Pro ResNet50 1.79×, MobileViTv2 1.32×) [primary]. — [coremltools perf](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-quantization-perf.md)
- **Community microbenchmark, 128 chained 1×1 convs at 512 channels [secondary]:**
  - M5: FP16 17.99 TOPS vs INT8 W8A8 33.86 TOPS (**1.88×**).
  - A commenter reports A18 Pro: FP16 15.86 vs INT8 30.53 TOPS.
  - The author attributes the gain to int8 activations halving inter-tile and L2 traffic, but the test also halves weight bytes and does not prove native int8 MACs.
  - — [Anemll gist, Mar 2026](https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c)
- **Int8 speedup by chip generation [secondary/search excerpt]:** about 1.0× on M1–M3 and about 1.9× on M4/M5 (Hoxton Macs). An ANE architecture monograph (arXiv 2606.22283) instead claims about 1.4× on M1; the sources conflict. — [Hoxton Macs](https://www.hoxtonmacs.co.uk/blogs/news/your-mac-has-a-hidden-ai-chip-developers-are-finally-unlocking-it); [arXiv 2606.22283](https://arxiv.org/pdf/2606.22283)
- **Weight encoding decides ANE placement [search excerpt]:** a measurement study found an fp16 LM export ran on the CPU despite allowing the ANE, while its int8 and 2-bit-LUT versions ran on the ANE. The int8 export cut warm forward latency 1.9×; another version reports 1.8–2.2× and residency rising from 0% to about 83%. "Successful conversion and prediction … do not establish that an exported language model executes on the ANE." Decode "streams the entire weight set every token." — [arXiv 2608.22110](https://arxiv.org/abs/2608.22110); [Pith summary](https://pith.science/paper/2608.22110)
- **Palettization vs linear quantization [primary]:**
  - Palettization's latency gain "is available only when palettized weights are … decompressed 'just in time' for computation", which is more likely on the ANE from iOS 17.
  - Some backends "decompress the weights fully before runtime, leading to a latency identical to that of the float16 model".
  - "Weight palettization (all bits from 1 to 8) typically works the best on the Neural Engine for runtime memory and latency gains"; W8A8 gives compute gains on A17 Pro/M4; INT4 per-block suits Mac GPUs.
  - Per-grouped-channel LUTs can slow runtime as LUT count grows.
  - Palettization is therefore a bandwidth and storage lever; only W8A8 changes the math.
  - — [coremltools opt-overview.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-overview.md); [opt-palettization-perf.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-palettization-perf.md)
- **Older reference:** Apple's `ane_transformers` reference gives "up to 10 times faster and 14 times lower peak memory" vs baseline transformer implementations on the ANE (distilbert on iPhone 13; 2.84× at seq 128, BS 1) [primary]. — [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers)
- **Core ML stateful models** (iOS 18) keep the KV cache as on-device state. Apple's Llama-3.1-8B Core ML example targets the **GPU** on an M1 Max because it is memory-bandwidth bound [search excerpt]. I could not retrieve its tokens/s. — [Apple ML Research](https://machinelearning.apple.com/research/core-ml-on-device-llama)

**Qualcomm Hexagon**
- **Precisions:** the Snapdragon 8 Elite Gen 5 product brief lists INT2, INT4, INT8, INT16, FP8 and FP16 including mixed precision [search excerpt]. The HMX matrix unit's INT4/INT8/INT16/FP16 support is attributed to Qualcomm; its FP16 instructions are undocumented [search excerpt]. — [Qualcomm product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-8-Elite-Gen-5-product-brief.pdf); [arXiv 2509.23324 (EuroSys 2026)](https://arxiv.org/abs/2509.23324)
- **FP16 on mobile NPUs:** llm.npu (ASPLOS 2025) found FP16 MatMul on a mobile NPU up to 159× slower than CPU INT8. It reaches >1,000 prefill tok/s for a ~1B model on a Redmi K70 Pro using INT8 NPU prefill [search excerpt]. — [arXiv 2407.05858](https://arxiv.org/pdf/2407.05858)
- **Measured LLM, Qualcomm AI Hub, Galaxy S25 (8 Elite) / S26 (8 Elite Gen 5), context 4096 [primary]:**

  | Model | Precision / runtime | Unit | Prefill (tok/s) | Decode (tok/s) |
  |---|---|---|---|---|
  | Llama-3.2-3B | w4a16, Genie (S25) | NPU | 728 | 26.5 |
  | Llama-3.2-3B | w4a16, Genie (S26) | NPU | 776 | 28.2 |
  | Qwen3-0.6B | w4a16, Genie (S25) | NPU | 2,262 | 125 |
  | Qwen3-0.6B | w4a16, Genie (S26) | NPU | 2,681 | 133 |
  | Qwen3-0.6B | q4_0, llama.cpp (S25) | CPU | 173 | 15.8 |
  | Qwen3-0.6B | q4_0, llama.cpp (S25) | GPU | 504 | 32.9 |
  | Qwen3-0.6B | q4_0, llama.cpp (S25) | NPU | 2,783 | 30.8 |

  NPU prefill is about 4.5–5.5× GPU prefill here. — [ai-hub-models qwen3_0_6b/perf.yaml](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_0_6b/perf.yaml); [llama_v3_2_3b_instruct/perf.yaml](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/llama_v3_2_3b_instruct/perf.yaml)
- **Measured ViT-B/16 (torchvision `vit_b_16`), Qualcomm AI Hub, NPU, ONNX Runtime path [primary]:**

  | Device | FP16 ("float") | w8a16 | w8a8 |
  |---|---|---|---|
  | Galaxy S25 | 4.42 ms | 3.16 ms (1.40× faster) | 5.81 ms (0.76×, slower) |
  | Galaxy S26 | 3.43 ms | 2.31 ms | 4.02 ms |
  | Galaxy S24 | 6.33 ms | 4.06 ms | 7.03 ms |

  The QNN-DLC path on S25: float 6.11, w8a16 3.93, w8a8 5.83 ms. — [ai-hub-models vit/perf.yaml](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml)
- **Google LiteRT on Qualcomm (FastVLM demo) [search excerpt]:** an int8/int16-quantized FastVLM on the Snapdragon 8 Elite Gen 5 NPU reaches TTFT about 0.12 s on 1024×1024 images, >11,000 prefill tok/s and >100 decode tok/s. LiteRT claims up to 100× vs CPU and 10× vs GPU across 72 models. — [InfoQ](https://www.infoq.com/news/2025/11/litert-snapdragon-accelerator); [Edge AI Vision](https://www.edge-ai-vision.com/2025/11/google-announces-litert-qualcomm-ai-engine-direct-accelerator/)
- **Contrary results [search excerpt]:**
  - A stage-level study on Snapdragon 8 Gen 3 found CPUs beat NPUs at prefill and NPU decode gains were limited, blamed on scheduling and fallback overheads. — [arXiv 2605.27435](https://arxiv.org/pdf/2605.27435)
  - On a Snapdragon X Elite laptop, NPU prefill was about 787 tok/s vs about 43 (CPU) and about 25 (GPU); decode was about 14 vs about 8 tok/s. — [arXiv 2606.11257](https://arxiv.org/html/2606.11257v1)

**MediaTek APU**
- Google's LiteRT NeuroPilot numbers for Dimensity 9500: Gemma 3n E2B at about 1,600 prefill tok/s and 28 decode tok/s at 4K context; up to 12× CPU and 10× GPU for LLM workloads [search excerpt]. — [Google Developers Blog](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/); [MarkTechPost](https://www.marktechpost.com/2025/12/09/google-litert-neuropilot-stack-turns-mediatek-dimensity-npus-into-first-class-targets-for-on-device-llms/)
- Community note: Gemma 3 1B on MediaTek NPU decode is pinned at about 16 tok/s, dispatch-bound [search excerpt]. Known LiteRT NPU bugs: Gemma 3 prefill chunks lost on Qualcomm, and MediaTek NPU compile aborts. — [LiteRT-LM issue #3508](https://github.com/google-ai-edge/LiteRT-LM/issues/3508); [LiteRT issue #9250](https://github.com/google-ai-edge/LiteRT/issues/9250)

**Google Tensor**
- Only LiteRT community NPU builds exist (e.g., [Gemma3-1B-IT-Tensor-NPU](https://huggingface.co/litert-community/Gemma3-1B-IT-Tensor-NPU)); I found no measured numbers.

### Inferences
- **Hexagon, LLM blocks:** activation precision is the compute lever. int16 fixed-point activations (w4a16/w8a16) are the production format and run about 1.8× faster than fp16 activations. Int8 activations (w8a8) are not reliably faster for transformer blocks: AI Hub's ViT-B/16 w8a8 is slower than fp16.
- **Hexagon, looped ViT encoder:** target w8a16 first (measured 1.4×). Treat w8a8 as an experiment that needs per-op profiling, probably because softmax and LayerNorm requantization dominate.
- **ANE:** W8A8 is the only compute lever. Expect about 1.3× for ViT-like blocks (MobileViTv2) up to about 1.8–1.9× for GEMM/conv-heavy blocks on A17 Pro/A18/M4+. Expect about 1.0× on M1–M3 class engines.
- **Palettization** (including 2–4-bit LUTs) reduces bytes streamed per decode step, and therefore per loop pass, but not MACs.

### Gaps
- I found no measured A17 Pro/A18 ANE LLM prefill/decode numbers for W8A8 vs W4A16 in Apple documentation.
- I found no MediaTek APU int8 vs int16 vs fp16 breakdown.
- I found no Google Tensor G4/G5 NPU numbers.
- I could not explain AI Hub's w8a8 ViT slowdown. The likely causes are int8 fallbacks or requantize ops in softmax/LayerNorm, which is unverified.

---

## Q5. Browser: WebGPU shader-f16, subgroups, DP4a (`dot4I8Packed`), WebNN status in 2026, measured int8 vs fp16 speedups (ONNX Runtime Web / WebLLM / transformers.js)

### Takeaway
WebGPU in Chrome has shader-f16, packed-int8 dot products (Chrome 123) and subgroups (Chrome 134). Subgroup-matrix (cooperative matrix) support is still experimental.

ONNX Runtime Web's int4/int8 kernels use DP4A only when activations are quantized (accuracy_level 4) and M ≥ 4 (prefill), on non-Apple GPUs. Metal has no native DP4A, so on Apple GPUs the int8 path is skipped.

Google reports 1.6–2.9× for 8-bit over f16 in its own testing. Browser LLMs reach about 70–80% of native speed.

WebNN is still in origin trial in Chrome/Edge as of 2026, with no stable shipping found, so browser NPU access is not production-ready.

### Cited Findings
- **Packed int8 dot product [search excerpt]:** `packed_4x8_integer_dot_product` (`dot4I8Packed`/`dot4U8Packed`) shipped in Chrome 123. It is detected via `navigator.gpu.wgslLanguageFeatures`, and browsers polyfill it when the hardware lacks it. — [Chrome WebGPU 123](https://developer.chrome.com/blog/new-in-webgpu-123); [MDN WGSLLanguageFeatures](https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures)
  - Native paths in wgpu (Firefox): DX12 SM ≥ 6.4 and Vulkan `VK_KHR_shader_integer_dot_product` [search excerpt]. — [wgpu PR #7494](https://github.com/gfx-rs/wgpu/pull/7494)
- **Google's int8 numbers [search excerpt]:** "compared to 16-bit floating point, 8-bit is 1.6 to 2.8 times faster … with packed integer dot products, it's 1.7 to 2.9 times faster." These are Google's own consumer-GPU tests, and a second search could not re-confirm the wording. — [Chrome I/O 2024 WebGPU part 2](https://developer.chrome.com/blog/io24-webassembly-webgpu-2)
- **Subgroups [search excerpt]:** "now available" in Chrome 134 (Feb 2025). Google Meet saw "2.3–2.9×" speedups vs packed integer dot products during the origin trial. Chrome 155–156 (2026) expanded subgroup support and added f16 on Linux with recent NVIDIA drivers. — [Chrome WebGPU 134](https://developer.chrome.com/blog/new-in-webgpu-134); [Chrome WebGPU 155-156](https://developer.chrome.com/blog/new-in-webgpu-155-156)
  - `chromium-experimental-subgroup-matrix` shows 0% availability in web3dsurvey [secondary]. — [web3dsurvey](https://web3dsurvey.com/webgpu/features/chromium-experimental-subgroup-matrix)
- **Other browsers [search excerpt]:** WebGPU shipped in Safari 26 (June 2025) and Firefox 141 (July 2025). Subgroup and shader-f16 status in Firefox and Safari is unconfirmed. — [Wikipedia WebGPU](https://en.wikipedia.org/wiki/WebGPU); [web3dsurvey shader-f16](https://web3dsurvey.com/webgpu/features/shader-f16)
- **ONNX Runtime WebGPU MatMulNBits kernel selection [primary]:**
  - The DP4A path requires `accuracy_level == 4` (int8 activation quantization), `block_size % 32 == 0`, `K % 128 == 0`, `N % 16 == 0`, and the Subgroups feature.
  - It is disabled when the adapter vendor is "apple": "Avoid using dp4a on Metal, as it does not appear to have native dp4a support."
  - It runs when M ≥ `kMinMForTileOptimization` (= 4), or always on FP32-only GPUs and Qualcomm GPUs ("integer math is faster than FP32").
  - A separate subgroup-matrix kernel needs 4- or 8-bit weights, block size 32, M ≥ 4, K % 32 == 0, N % 64 == 0 and batch 1.
  - So M < 4 (decode) uses dequantize-to-float kernels, which are weight-only in effect.
  - — [dp4a_matmul_nbits.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_nbits.cc); [subgroup_matrix_matmul_nbits.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc); [matmul_nbits_common.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_common.h) (main, 2026-10-12)
- **WebLLM [search excerpt]:** "can retain up to 80% native performance on the same device". Secondary reports of its Table 1 on an M3 Max with 4-bit models: Llama-3.1-8B 41.1 vs 57.7 tok/s native (71%); Phi-3.5-mini 71.1 vs 89.3 (80%). — [WebLLM arXiv 2412.15803](https://arxiv.org/html/2412.15803v1); [tianpan.co summary](https://tianpan.co/blog/2026-04-17-browser-native-llm-inference-webgpu)
- **transformers.js:** q4f16 (4-bit weights, fp16 compute) is the usual WebGPU recommendation, which is weight-only for compute purposes [secondary]. An issue reports architecture-dependent slowdowns at the same dtype: Qwen3.5-4B q4f16 is about 3× slower to decode and about 20× slower to first token than Qwen3-4B, on 4.0.0-next.7 [primary issue]. — [promptfoo docs](https://www.promptfoo.dev/docs/providers/transformers/); [transformers.js #1599](https://github.com/huggingface/transformers.js/issues/1599)
- **WebNN [search excerpt]:**
  - W3C Candidate Recommendation updated 22 Jan 2026.
  - Chrome origin-trial windows are reported inconsistently: M146–M148 per blink-dev, M147–M149 per an April 2026 guide, or 156–160 per webnn.io. Android is excluded; Edge follows Chrome.
  - Production as a default path is estimated around 2027. Only Chromium has implementations.
  - — [blink-dev Intent to Experiment](https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98/m/xMw0U5NkAAAJ); [Phoronix Chrome 146 beta](https://www.phoronix.com/news/Chrome-146-Beta); [webnn.io OT registration](https://webnn.io/en/learn/get-started/ot_registration); [utsubo 2026](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready)

### Inferences
- **What works in browsers for a looped model in 2026:**
  - Prefill and per-frame ViT: W4/W8 weights with int8 activations via DP4A, on Windows/Android/Linux GPUs only (not Apple Metal).
  - Decode: f16 (shader-f16) compute with 4-bit weights.
- **Shared weights across loop passes help browsers twice:** downloads are smaller, and the same storage buffers can be bound repeatedly. Per-pass compute is unchanged.
- **NPU access from the browser (WebNN)** should be treated as experimental for 2026 deployments.

### Gaps
- I found no independent, methodology-documented int8 (DP4A) vs f16 ViT or LLM-prefill benchmark in ORT Web or transformers.js on phones.
- I found no Safari/Firefox shipping status for `packed_4x8_integer_dot_product`, subgroups or shader-f16 on iOS.

---

## Q6. Do NPUs/DLAs support loops (a shared block executed repeatedly) without unrolling, and does looping help or hurt on NPUs?

### Takeaway
I found no evidence that any phone NPU (Hexagon HTP, ANE, MediaTek APU) or the Jetson DLA executes a rolled loop over a shared transformer block. Their graphs are static and compiled, so a looped model is deployed unrolled.

What these stacks do support is **weight deduplication across graphs or functions**:
- QNN weight sharing (up to 64 graphs per context binary).
- Core ML multifunction models (hash-based dedup).
- ANEMLL-Dedup for palettized ANE models.

So looping keeps its memory/storage benefit but not a compute benefit. TensorRT on the GPU does support an ONNX `Loop` op, with constraints.

### Cited Findings
- **TensorRT GPU [primary]:** ONNX `Loop` is supported with "FP32, FP16, BF16, INT32, INT64, BOOL" and the constraint that "the shape of Loop carried dependencies must be the same across all loop iterations". `If` and `Scan` are also supported. INT8/FP8 do not appear in the Loop type list. — [onnx-tensorrt operators.md](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- **DLA [primary]:** the DLA is a fixed-function CNN engine (convolution, deconvolution, FC, activation, pooling, BN), and unsupported layers fall back to the GPU. Per Q3, attention does not run on the DLA. — [Deep-Learning-Accelerator-SW](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW)
- **Core ML [primary]:** ML programs can include "functions, blocks, control flow". Converting data-dependent loops needs `torch.jit.script`, which has "limited support". Multifunction models (iOS 18/macOS 15) dedupe "shared weights by calculating the hash of the weight values". — [comparing-ml-programs-and-neural-networks.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/comparing-ml-programs-and-neural-networks.md); [model-scripting.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/model-scripting.md); [multifunction-models.md](https://github.com/apple/coremltools/blob/main/docs-guides/source/multifunction-models.md)
- **Core ML placement [search excerpt]:** placement is a cost-driven plan that segments the model across CPU, GPU and ANE; the compute-unit setting is a hint, not a guarantee. — [arXiv 2606.22283](https://arxiv.org/pdf/2606.22283)
- **ANEMLL-Dedup [primary]:**
  - Core ML's byte-level dedup misses palettized weights that are semantically identical. Different graph shapes produce different LUTs, causing 15–40% bloat.
  - ANEMLL re-points them to anchor blobs. Qwen2.5-0.5B LUT6 multifunction package: 862→472 MB. "The deduped combined package is essentially the same size as a single function."
  - — [ANEMLL docs/anemll-dedup.md](https://github.com/Anemll/Anemll/blob/main/docs/anemll-dedup.md)
- **QNN [search excerpt]:** QNN weight sharing lets "one context binary hold multiple graphs that share common weights, across up to 64 graphs" (Linux host only, March 2025 KBA). ExecuTorch added multi-graph contexts with HTP weight sharing for prefill/decode. — [Qualcomm KBA-250306025001](https://docs.qualcomm.com/doc/KBA-250306025001/KBA-250306025001_REV_1_QNN_Weight_Sharing_Feature.pdf); [ExecuTorch PR #6657](https://github.com/pytorch/executorch/pull/6657)
- **Decode bandwidth on the ANE [search excerpt]:** "decode streams the entire weight set every token" (measurement study on the ANE). — [arXiv 2608.22110](https://arxiv.org/abs/2608.22110)
- **Dispatch-bound decode [search excerpt]:** MediaTek NPU decode for Gemma 3 1B is "dispatch-bound" at about 16 tok/s. — [HF litert-community](https://huggingface.co/litert-community/Gemma3-1B-IT-Tensor-NPU)

### Inferences
- **Looping helps on NPUs in three ways:**
  - Smaller model files and RAM, provided the toolchain dedupes the unrolled copies: within-graph dedup is unverified; across graphs, QNN, Core ML and ANEMLL do dedupe.
  - A smaller unique-weight working set, which could let the shared block stay in on-chip SRAM/caches across passes and cut DRAM traffic per decode token. This is unverified, since I found no on-chip SRAM sizes or reuse measurements.
  - Fewer distinct encodings to calibrate.
- **Looping can hurt on NPUs in four ways:**
  - Per-loop activation scales (LoopQ) or per-loop adapters force unrolling.
  - Dynamic loop counts (adaptive halting/early exit) are incompatible with static graphs. They need one graph per depth, or CPU-orchestrated multiple calls with per-call dispatch overhead (decode is already dispatch-bound on some NPUs).
  - A k-pass model has the same FLOPs as the unrolled model, so NPU latency is the same as an equally deep non-looped model.
  - Error compounding (Q2) pushes toward higher-precision activations (int16 rather than int8), which are about 2× slower on fixed-point units.
- **On the Jetson GPU**, a TensorRT `Loop` could keep a rolled graph. But the Loop type list excludes INT8/FP8, and Edge-LLM uses layer-unrolled engines, so unrolling is the safe assumption for quantized deployment.

### Gaps
- I found no documentation or measurement of QNN/HTP, ANE or MediaTek executing a rolled loop, or of within-graph weight dedup in an unrolled graph.
- I found no public on-chip SRAM sizes (HTP VTCM, ANE caches) to judge whether a shared block can stay resident.
- I found no measurement of a looped model on any NPU.

---

## Q7. Realistic compute speedups per platform from int8/int4 activations for (a) per-frame ViT encoders and (b) LLM prefill/decode

### Takeaway
The range runs from about 1.0× (Orin or Thor GPU with INT8 ViTs, Apple GPU/Metal) to about 6× (Thor NVFP4 prefill vs W4A16).

For per-frame ViT encoders, plan on:
- 1.3× on ANE W8A8 (A17 Pro+).
- 1.4× on Hexagon w8a16 vs fp16 (w8a8 may regress).
- 1.0–1.2× on Jetson GPU INT8/FP8, unless attention fusion is achieved.

For LLM prefill:
- About 4× on mobile CPU W4A8 vs BF16.
- About 1.8× from int16 vs fp16 activations on Hexagon.
- 3–6× from NVFP4 vs INT4-AWQ on Thor.

Decode gains come from weight bits, not activation bits.

### Cited Findings (summary table; every row cited above)
| Platform | Workload | Comparison | Measured speedup | Source |
|---|---|---|---|---|
| iPhone 15 Pro ANE (A17 Pro) | MobileViTv2 (hybrid ViT) | W8A8 vs FP16 | 1.32× (1.36→1.03 ms) | [coremltools](https://github.com/apple/coremltools/blob/main/docs-guides/source/opt-quantization-perf.md) [primary] |
| iPhone 15 Pro ANE | ResNet50 | W8A8 vs FP16; weight-only | 1.79×; weight-only 0.92× | same [primary] |
| iPhone 14 Pro ANE (A16) | MobileViTv2 | W8A8 vs FP16 | 1.17× | same [primary] |
| M5 / A18 Pro ANE | 1×1 conv chain | W8A8 vs FP16 | 1.88× / about 1.9× | [Anemll gist](https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c) [secondary] |
| Galaxy S25 Hexagon | ViT-B/16 | w8a16 vs fp16; w8a8 vs fp16 | 1.40×; 0.76× (slower) | [AI Hub vit](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml) [primary] |
| Galaxy S25 Hexagon | Llama-3.2-3B prefill / decode | w4a16 (int16 act) vs w4 (fp16 act) | 1.84× / 1.85× | [AI Hub llama 3B](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/llama_v3_2_3b_instruct/perf.yaml) [primary] |
| Galaxy S25 | Qwen3-0.6B prefill | NPU vs GPU vs CPU (q4_0, llama.cpp) | NPU 2,783 / GPU 504 / CPU 173 tok/s (ctx 4096) | [AI Hub qwen3](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_0_6b/perf.yaml) [primary] |
| OnePlus 12 CPU (XNNPACK) | Llama-3.2-1B/3B prefill / decode | W4A8 dynamic (SpinQuant) vs BF16 | 4.2–4.3× / 2.6× | [ExecuTorch](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md) [primary] |
| Dimensity 9500 APU | Gemma 3n E2B | NPU vs CPU/GPU | up to 12× / 10× (about 1,600 prefill, 28 decode tok/s) | [Google blog](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/) [search excerpt] |
| Jetson AGX Thor | Qwen3-8B prefill / decode | NVFP4 (W4A4) vs INT4 AWQ (W4A16) | 6.3× / 0.96× | [Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md) [primary] |
| Jetson AGX Thor | gemma-4-31B prefill / decode | FP8 (W8A8) vs INT4 AWQ | 2.8× / 0.58× | same [primary] |
| Jetson AGX Thor | ViT (base) | FP8 vs FP16 | about 1.2× (about 20% latency cut) | [NVIDIA forum](https://forums.developer.nvidia.com/t/low-vit-performance-gain-on-jetson-thor-using-fp8-vs-fp16/349329) [secondary] |
| Jetson Thor | RT-DETR / Swin-UNet | INT8 vs FP16 | about 1.0× or worse | [NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-thor-int8-quantization-show-no-performance-gain-over-fp16/353308) [secondary] |
| Jetson Orin Nano | ViT-S (+DPT) | INT8 vs FP16 | ViT-S about 1.01–1.013×; full model 0.37× (regression) | [NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835) [secondary] |
| Jetson AGX Orin DLA | CNNs | 2:4 sparse vs dense INT8 | 1.24–1.47× | [DLA-SW](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW) [primary] |
| Browser WebGPU (Chrome, consumer GPUs) | ML kernels | int8 / packed dot vs f16 | 1.6–2.8× / 1.7–2.9× | [Chrome I/O 2024](https://developer.chrome.com/blog/io24-webassembly-webgpu-2) [search excerpt] |
| Browser vs native (M3 Max) | LLM decode, 4-bit | WebGPU vs Metal | 71–80% of native | [WebLLM](https://arxiv.org/html/2412.15803v1) [search excerpt/secondary] |

### Inferences
- **(a) Per-frame ViT encoders run as loops:**
  - Each pass is compute bound at realistic token counts (196–1,024 tokens).
  - Best levers per platform:
    - iPhone ANE: W8A8, about 1.3–1.8×.
    - Snapdragon NPU: w8a16, about 1.4×.
    - Orin GPU: FP16 is usually best unless INT8 attention fusion is engineered.
    - Thor: FP8 about 1.2×. NVFP4 for ViTs is not shown in the Edge-LLM tables; the visual tower there is FP16 or FP8.
  - Looped ViTs are sensitive to activation error accumulating across passes (Q2). Prefer 8/16-bit activations with per-pass calibration over per-tensor 4-bit.
- **(b) LLM prefill vs decode:**
  - Prefill gains require activation quantization: Thor NVFP4 3–6×; phone CPU W4A8 about 4×; Hexagon int16 vs fp16 activations about 1.8×.
  - Decode gains come from fewer weight bytes. A k-pass looped model re-reads the shared weights k times per token, so 4-bit weights (or 2–4-bit palettization on ANE) give about 2–2.6× vs 16-bit. Activation bits barely matter for decode (Thor AWQ ≈ NVFP4).
- **Combining with looping:** the per-pass precision speedup multiplies with whatever pass-count reduction other researchers find.
  - The binding constraint is accuracy: looped and recursive models tolerate W4A8/W8A8 (LoopQ W4A8 is within reach; 8-bit integer is "a small perturbation" in recursive reasoners).
  - W4A4 needs block-scaled formats and loop-aware calibration, which in this device set only Thor supports in hardware.

### Gaps
- I found no measurement anywhere of a quantized *looped* model's latency on Jetson, phone NPUs or WebGPU.
- I found no NVFP4 ViT encoder measurement on Thor.
- Several phone and browser numbers are vendor claims seen only in search snippets (Google LiteRT, Chrome I/O 2024). Verify them on target devices.
