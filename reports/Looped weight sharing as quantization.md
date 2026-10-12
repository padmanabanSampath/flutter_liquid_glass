# Calibration alone cannot loop a pretrained transformer

**No published method turns a pretrained transformer into a looped, weight-shared model with 2–3× fewer unique weights using calibration data alone. Every training-free attempt on record either collapses or barely compresses.** Re-using one middle layer's weights at several depths of a frozen Llama-2 drives accuracy toward the random baseline. The training-free cross-layer sharing that does work removes 20% of parameters at +2.1 WikiText-2 perplexity on LLaMA-1-7B and +9.4 on Llama-3.1-8B, while 4-bit GPTQ cuts bytes about 4× at roughly +0.5. One measurable quantity explains the gap *[derived]*: coding a layer as a prediction from an aligned partner layer saves **½·log₂(1/(1−R²)) bits per weight**, where R² is the activation-weighted share of the layer that its partner explains. A pure 3× loop at int4 fidelity needs R² ≈ 0.996, but the evidence puts bulk weights near 0.05–0.15 after alignment, with only a minority of attention heads near 0.98. The looping principle therefore survives as post-training compression only in reframed form, which this report calls **depth-shared residual quantization**. Each group of tied layers gets an aligned shared base, per-pass scales and gains (the closed-form counterparts of per-pass norms, LARM-style FiLM and RRT-style per-pass LoRA), and per-layer residuals whose bits follow measured R². All of it is fitted by one Hessian-summed objective, which is exactly the closed-form version of "gradients from every pass land on the shared block." With calibration alone this should trim only about **2–10% off int4 bytes on modern LLMs**, with ViT and per-frame video encoders the most promising and least measured case *[inference]*. A real 2–3× looped student remains a distillation job costing between 3×10⁷ tokens (DeltaLLM's 12–24% cut) and 6×10¹⁰ tokens (RRT's 2×). The dependable payoff is download size and storage, since decode speed, per-frame latency and KV cache do not shrink from weight sharing on its own. A redundancy census that costs GPU-hours and no training should decide whether to build any of it.

## Paper sites were blocked, so treat several numbers as provisional

The five research notes behind this report were compiled on 2026-10-12 in an environment whose proxy blocked arXiv, OpenReview, ACL Anthology, the ICLR and NeurIPS proceedings, NVIDIA's site and Hugging Face. As a result, paper figures come from three kinds of source: search-engine excerpts of those pages, GitHub READMEs, configs and released data that the researchers read directly, and occasionally third-party summaries. Many 2026 papers are known only from abstracts or from one-line summaries in the Awesome-Loop-Models catalog ([catalog](https://github.com/huskydoge/Awesome-Loop-Models)).

The labels work as follows. A claim with a source link and no tag is a published or released result, as excerpted; *[secondary]* means the number appeared only in a third-party summary; *[unreviewed]* flags single-author preprints, unreviewed repositories and catalog summaries; *[derived]* marks algebra or arithmetic done by the research team or in this report; and *[inference]* marks a judgment that no source has tested. The numbers in the table below carry the most weight in the argument and should be checked against the PDFs before anyone commits engineering time.

| Number used here | How it was obtained | What to verify |
|---|---|---|
| Relaxed Recursive Transformer (RRT) Gemma: 58.4% vs 58.6% (rank-512 LoRA, 60B tokens with KD) | Search excerpt of [arXiv HTML](https://arxiv.org/html/2410.20672v1) | The table itself, the parameter count of the rank-512 model, and any zero-token row |
| RRT: SVD-initialized LoRA beats zero init by +6.5 points | Secondary review | The paper's initialization figure and appendix |
| MiniViT: DeiT-B at 9M parameters scores 79.8% | Secondary | The CVPR table |
| Basis Sharing on LLaMA-2-7B at 20% removal: 7.70 vs 7.57 PPL | Conflicting excerpts ([OpenReview](https://openreview.net/forum?id=gp32jvUquq) vs [D-Rank](https://arxiv.org/pdf/2509.25622)) | The paper's own table |
| Llama-2-7B at 2 bits: QuIP# 6.19 vs 6.66; QTIP 5.91 vs 6.82 | Conflicting extracts, probably different context lengths | The primary tables |
| MoR KV memory ≈ (N_r+1)/(2N_r) of baseline | Garbled PDF extraction | MoR's KV table |
| Ouro: about 4× decode-KV saving from last-step sharing | Secondary guide | The Ouro paper |
| MCWC beats GPTQ-W4 at ~4.2 bits; LegoLM reaches 4.41× at +0.03% PPL | Unreviewed single-author preprints; MCWC's model is unnamed | Bit accounting and baselines |
| Ouro-1.4B under INT4 round-to-nearest: WikiText-2 10.68 → 12.48 | Unreviewed GitHub repo | Reproduce it |
| Retrofitted Recurrence training budgets (~26B and ~50B tokens) | Derived by the research team from logged FLOPs | The paper text |

## Looping converts parameters into depth, not into speed or smaller KV

A looped (recursive, weight-shared) transformer stores K unique blocks and runs each of them several times, so a 3× loop such as A→B→A→B→A→B gets six layers of depth from two blocks of weights. Trained from scratch, such models are strong. ALBERT shares every layer and shrinks ALBERT-base from **108M to 31M parameters** while its GLUE average falls from 82.3 to 79.8, with almost all of that loss coming from sharing the FFN rather than attention ([ALBERT](https://arxiv.org/abs/1909.11942)). Mixture-of-Recursions' 167M model with two recursions beats a 315M vanilla Transformer at equal training FLOPs, 43.1% vs 42.3% few-shot ([MoR](https://arxiv.org/pdf/2507.10524)). Ouro's 1.4B and 2.6B looped models, pretrained on 7.7T tokens, match 4B models and beat models of up to 8B on reasoning-heavy tasks, gains the authors attribute to knowledge manipulation rather than knowledge capacity ([Ouro](https://arxiv.org/abs/2510.25741)). LARM, which the research identifies as Idiap's depth-conditioned looped ASR encoder (the only looped model by that name it found *[inference]*), reaches **11.34 test-clean WER with 7.7M parameters** against 14.43 for a 28.9M unshared 16-block encoder on LibriSpeech-100h ([LARM](https://github.com/idiap/LARM)). All of these were trained into existence; none compresses a model that already exists.

| Quantity | K unique blocks run as N passes, vs an untied N-layer model | Evidence |
|---|---|---|
| Unique parameters, file and download size | Block weights shrink N/K×; embeddings and head unchanged | ALBERT-base 108M → 31M ([ALBERT](https://arxiv.org/abs/1909.11942)); a 128K×2048 embedding table is ~262M parameters, ~131 MB in int4 *[derived]* |
| Resident weight RAM | Shrinks N/K× only if the runtime aliases one copy | llama.cpp `num_loops` aliases physical layers ([nanbeige.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/nanbeige.cpp)) |
| FLOPs per token or frame | Unchanged at equal depth | ALBERT-xxlarge has ~70% of BERT-large's parameters but trains ~3× slower ([ALBERT](https://arxiv.org/pdf/1909.11942)) |
| DRAM bytes per decoded token | Unchanged unless the block is still cached when it runs again | MobileLLM's back-to-back repeat costs +2.6% execution time vs +86% for doubled untied depth ([MobileLLM](https://arxiv.org/pdf/2402.14905)) |
| KV cache | Unchanged: each pass keeps its own KV unless the model is trained to share it | llama.cpp: "each slot still has its own KV index" ([nanbeige.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/models/nanbeige.cpp)) |
| Quality | Knowledge bounded by unique parameters; loops help reasoning | Ouro; retrofit data below |
| Quantization robustness | Worse, because sharing and quantization remove overlapping redundancy | ALBERT keeps 93.4% at 4-bit, 72.5% at 3-bit ([quantization study](https://arxiv.org/pdf/2010.07109)) |

The best public data on converted loops show what unique parameters buy. In the Retrofitted Recurrence release, TinyLlama converted to a 4-8-4 prelude/core/coda model, keeping 16 of its 22 layers unique and trained on about 26B tokens *[derived from logged FLOPs]*, scores **52.5 HellaSwag** when run as 24 layer passes, against 61.4 for the static model given the same continued training. It needs four recurrences (40 passes, about twice the compute) to reach 60.5, and a 2-4-2 variant with only 8 unique layers **stalls at 42.7 even at 132 passes** ([retrofit data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)). A September 2026 analysis likewise finds that extra loops help reasoning but hurt knowledge ([arXiv 2609.36636](https://arxiv.org/abs/2609.36636)) *[unreviewed]*. Unique parameters set the knowledge ceiling and extra passes do not buy it back *[inference]*, a limit that any scheme removing unique weights from a token model must respect.

### The payoff is bytes at rest, not speed

For download size, a 1B-parameter int4 model holds about 0.55 GB of weights, and looping 4 unique blocks over 16 passes brings it to roughly 0.25–0.3 GB rather than 0.14 GB, because embeddings dominate small models *[derived]*. That matters most where storage ceilings are hard. Browsers default to WebGPU limits of 128 MiB per storage binding and 256 MiB per buffer ([WebGPU spec](https://github.com/gpuweb/gpuweb/blob/main/spec/index.bs)), ONNX Runtime Web cannot run models above 4 GB ([ORT Web](https://onnxruntime.ai/docs/tutorials/web/large-models.html)), and Safari still lacks Memory64 ([caniuse](https://caniuse.com/wf-wasm-memory64)). On iPhones, developers report a per-app ceiling of about 6 GB even on 8–12 GB phones ([Apple forum](https://developer.apple.com/forums/thread/834805)) *[secondary]*. On Jetson, weights are a small share of the footprint: Qwen3-0.6B in INT4 peaks at **1,917 MB** on Orin Nano although its weights are only about 0.3–0.4 GB ([TRT Edge-LLM benchmarks](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/performance/performance-benchmarks.md)).

Speed is another matter. Batch-1 decode on these devices is bandwidth-bound, and their on-chip caches are far smaller than a 1B-class block of about 6–100 MB *[derived]*:

| Device | Cache |
|---|---|
| Orin | 4 MB GPU L2 ([Hot Chips](https://hc34.hotchips.org/assets/program/conference/day2/ADAS%20and%20Grace/HC2022.NVIDIA.Mike_Ditty.v6.pdf)) |
| Snapdragon 8 Elite | 8 MB SLC ([Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Processor-Benchmarks-and-Specs.908499.0.html)) |
| A18 Pro | 24 MB SLC ([heise](https://heise.de/-9960558)) |

So a cyclic loop re-reads every block from DRAM, and only small blocks repeated back to back stay resident. The headline speedups from looped models come from early exit and depth-wise batching, not from cached weights. RRT's 2–3× throughput, for instance, is a simulation with oracle exits ([RRT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html)). Per-frame ViT encoders are compute-bound, so sharing leaves their latency untouched *[inference]*.

## Per-pass parameters are cheap; conversion budgets are not

### Norms and FiLM differentiate passes almost free; per-pass LoRA costs real bits

Shared weights force every pass to compute the same function of its input, so looped models add small per-pass parameters that let one pass behave differently from another. The lightest is the user's per-pass normalization, a separate gain and shift per pass costing about 2d numbers per block per pass. LARM's FiLM is the smooth version: after each application of the shared encoder it sets h⁽ᵏ⁾ = γ(d̄) ⊙ â⁽ᵏ⁾ + β(d̄) with normalized depth d̄ = (k−1)/(K−1), and together with a clock embedding and input re-injection this lowers 100h test-clean WER from **12.70 for naive looping to 11.34** ([LARM README](https://github.com/idiap/LARM); [LARM paper](https://arxiv.org/html/2606.04678v2)). Because γ and β are functions of normalized depth, FiLM can interpolate to pass counts it never trained on, which per-pass lookup tables cannot *[inference]*. RD-ViT sits between the two, pairing a shared LoRA with a per-loop scale vector that costs about 1.5K parameters in its Tiny model ([RD-ViT](https://arxiv.org/pdf/2605.03999)). The heaviest option is RRT's per-position LoRA, initialized by truncated SVD (ranks 64–512) of the residual between each original layer and the shared weights ([rrt-lora](https://github.com/avramdj/rrt-lora); [RRT](https://arxiv.org/pdf/2410.20672)).

On the post-training-quantization side, LoopQ shows the same pattern for already-looped models. It keeps one shared quantized backbone plus loop-dependent activation scales and cross-loop adapters that touch **under 0.12% of parameters**, and at W4A4 it improves average accuracy by 68.8% over the strongest static PTQ baseline ([LoopQ](https://arxiv.org/abs/2605.16343)).

Two cautions carry into any design. Hidden-state scale grows with depth, which is why naive recursion fails in VLMs; RecursiveVLM fixes it with an RMSNorm connector and per-recursion supervision for about +7% over vanilla recursive baselines ([RecursiveVLM](https://arxiv.org/html/2602.09080v1)). And adapters shared across loops drift toward late loop positions unless training randomly drops them ([Loop Dropout](https://arxiv.org/abs/2609.34218)) *[unreviewed]*.

| Mechanism | Extra parameters | Extrapolates to new pass counts | Closed-form fit from calibration |
|---|---|---|---|
| Per-pass norm gain and shift (user) | ~2d per block per pass | No | Yes: the gain folds into input-channel scales *[derived]* |
| FiLM of normalized depth (LARM) | Small γ/β maps | Yes | Fit per-pass values, then regress them on depth *[inference]* |
| Shared LoRA with per-loop scale (RD-ViT) | ~1.5K (Tiny) | No | Partly |
| Per-position LoRA (RRT) | r(m+n) per matrix per position | No | Yes: truncated SVD of the residual |
| Loop-dependent scales and adapters (LoopQ) | <0.12% of model | No | Yes |

### Every conversion that removed unique weights trained on 30M to 60B tokens

Training budgets for converting a pretrained model span four orders of magnitude, and the reduction in unique weights grows with the budget.

| Method | What it ties | Training | Tokens | Result |
|---|---|---|---|---|
| DeltaLLM ([arXiv](https://arxiv.org/abs/2501.18596)) | Later blocks reuse earlier ones, plus low-rank deltas | Deltas only, progressive module replacement | ~30–40M | 12% fewer parameters at ~90% of performance; a 24% variant matches a 12% one |
| Verma et al. FFN merging ([TMLR](https://openreview.net/pdf?id=t8iuiH46g0)) | Permutation-aligned FFN sublayers | QLoRA, 3,000 updates (OLMo3-7B) | Small | Over a third of FFNs tied at comparable performance; ViT: >21% of parameters removed at 99% accuracy |
| FlexiGPT ([NAACL](https://aclanthology.org/2025.naacl-long.31/)) | Pruned blocks replaced by shared copies plus LoRA | Recovery training | ~1B | 30% compression at PPL 6.55 *[secondary]* |
| LiSA ([repo](https://github.com/takagi97/lisa)) | Attention across adjacent layers | 0.46–1.64% of parameters | ~4.2B | Q/K compressed 6× in 53–84% of layers |
| RRT ([arXiv](https://arxiv.org/abs/2410.20672)) | Whole model looped 2× | Full uptraining, with KD for the best results | 15–60B | Recursive Gemma 1B beats a same-size baseline by 13.5 points; relaxed rank-512 model at 58.4 vs 58.6 |
| Retrofitted Recurrence ([data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots/data)) | Prelude/core/coda | Full training with a recurrence curriculum | ~26–50B *[derived]* | Matches the static model only at r ≥ 4 |
| Ouro, from scratch ([arXiv](https://arxiv.org/abs/2510.25741)) | Whole stack, 4 loops | Pretraining | 7.7T | 1.4B ≈ 4B models |

Initialization matters, and it interacts with relaxation. In RRT, **Stepwise** initialization (pick real layers at intervals, keeping the first and last) is best for strict ties, while **Average** initialization is best once per-position LoRA is added ([RRT](https://arxiv.org/html/2410.20672v1)).

The headline near-parity result also needs a parameter-count check *[derived]*. A rank-512 factor pair holds half as many parameters as a 2048×2048 matrix, and 31% of a 2048×8192 matrix. With one adapter per tied position, the rank-512 relaxed model therefore gives back much of what tying removed. The paper's exact counts were not retrieved, so read 58.4% as a lightly compressed point, and the unrelaxed recursive Gemma 1B, which "recovers most" of Gemma 2B, as the clean 2× point.

Vision follows the same pattern. MiniViT compresses DeiT-B 9.7× (86M → 9M) at 79.8% vs 81.8% top-1 *[secondary]*, and makes Swin-B 48% smaller while gaining 1.0 point. Both results rely on per-layer weight transformations plus distillation, and the paper reports that naive sharing destabilizes training and can collapse it ([MiniViT](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html)).

The user's pipeline is the standard heavy path. It distills a looped student from the full teacher: stage A matches the teacher's internal outputs, stage B fits a small head, and gradients from every pass are summed on the shared block. That is the MiniViT/RRT recipe. LiSA's 0.75·CE + 0.25·MSE loss on internal states is a close relative ([LiSA](https://github.com/takagi97/lisa)). Reassessing Layer Pruning supports stage B directly: refitting only the LM head plus the last three layers beats LoRA healing by 5.62–17.27% on Llama-3.1-8B-It ([arXiv 2411.15558](https://arxiv.org/html/2411.15558)).

## No calibration-only conversion survives, and one number explains why

### Every training-free attempt on record collapses or barely compresses

The negative evidence is consistent across language and vision. In "Transformer Layers as Painters," replacing middle layers of a frozen Llama-2 with copies of the middle-most layer is the worst variant tested: it **degrades quickly to the random baseline**, worse than simply skipping those layers ([Painters](https://arxiv.org/abs/2407.09298)). Training-free re-application of frozen blocks works only with damped sub-steps and keeps every original weight, and its authors note that naive block re-application "usually degrades performance" ([Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1)). Freshly converted Llama-3.2-1B 2-4-2 retrofits start training at **12.4–14.3 nats**, above uniform-random (ln 128K ≈ 11.8), with the caveat that the first step runs at the sampled training recurrence rather than r=1 ([retrofit CSVs](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots)). CHERRY's averaged blocks "start with high loss before training recovers it" ([CHERRY](https://arxiv.org/abs/2606.31796)) *[unreviewed]*, and MiniViT reports training collapse as the number of shared vision layers grows ([MiniViT](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.pdf)). Weight-space fit is not model quality either: a 2026 study that coupled cross-layer SVD subspaces **cut Frobenius error by 37%** on Pythia-1.4B, yet perplexity rose from **50.17 to 159.14** ([arXiv 2605.30836](https://arxiv.org/abs/2605.30836)).

The most valuable missing number is RRT's accuracy right after SVD initialization, before any uptraining. No accessible source reports it.

The training-free sharing that works compresses little. Basis Sharing fits a shared SVD basis per layer group from 256×2048 calibration tokens. It shares q, k, v, up and gate, and keeps o and down private ([config](https://raw.githubusercontent.com/TUDa-HWAI/Basis_Sharing/main/tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml)). Under LowRankArena's matched protocol at 80% of parameters kept, its WikiText-2 perplexity compares with dense as follows ([LowRankArena](https://github.com/Zishan-Shao/lowrankarena)):

| Model | Basis Sharing at 80% kept | Dense |
|---|---|---|
| LLaMA-1-7B | 7.74 | 5.67 |
| Llama-3.1-8B | 15.61 | 6.24 |
| Qwen3-8B | 11.05 | 7.00 |

GPTQ at 4 bits reaches **6.21 at 3.9 GB** on LLaMA-7B ([SVD-LLM](https://arxiv.org/pdf/2403.07378)). Byte for byte, quantization wins by roughly 20× *[derived]*, and sharing degrades sharply on newer, more heavily trained models.

Head-level sharing does a little better. DirectShare re-uses pretrained head weights across layers with no retraining and keeps most of the performance below a 30% sharing ratio ([Head-wise Shareable Attention](https://aclanthology.org/2024.findings-emnlp.144.pdf)). That points to where the redundancy actually lives.

### Predicting a layer from its partner saves ½·log₂(1/(1−R²)) bits per weight

Treat the shared base as a predictor and the per-layer residual as what must still be coded *[derived]*. Under the Gaussian rate-distortion model behind quantization theory, coding a matrix to distortion D costs ½·log₂(σ²/D) bits per weight; the best aligned, rescaled prediction from a partner layer leaves residual energy (1−R²)·σ², so prediction saves **½·log₂(1/(1−R²)) bits per predicted weight**.

Here `R² = 1 − ‖(Wᵢ − Ŵᵢ)Xᵢ‖² / ‖WᵢXᵢ‖²`, measured in the activation-weighted metric that GPTQ-style quantizers optimize.

A pure loop codes that residual with zero bits. To match int4 fidelity, the residual must therefore fall below int4's own quantization noise, about 2⁻⁸ of the weight energy, which means R² ≈ 0.996. The research notes reach the same conclusion another way. After subtracting the mean of N tied layers, the relative residual energy is about (1−ρ)(1−1/N), so unless the pairwise correlation ρ exceeds about 0.8, the "delta" is nearly as large as the weight itself *[derived]*.

| R² (correlation ρ) | Bits saved per predicted weight *[derived]* | Where this level appears |
|---|---|---|
| 0.05 (0.21) | 0.04 | Aligned ViT attention activations, as a proxy ([Verma et al.](https://openreview.net/pdf?id=t8iuiH46g0)) |
| 0.15 (0.385) | 0.12 | Aligned ViT FFN neurons, as a proxy (same source) |
| 0.75 (0.87) | 1.0 | Needed for a 3-bit residual to reach int4 fidelity |
| 0.81 (0.90) | 1.2 | Most similar layer pairs on Q‖K head weights ([Head-wise](https://aclanthology.org/2024.findings-emnlp.144.pdf)) |
| 0.98 (0.99) | 2.8 | The most similar individual head pairs (same source) |
| 0.996 (0.998) | 4.0 | A pure loop (zero-bit residual) at int4 fidelity |

For bulk weights, trained models sit in the top rows of that table. DOCS finds that standard similarity indices cannot tell different layers' weights apart, because the weights stay near their orthogonal initialization, and adjacent layers are only modestly more alike ([DOCS](https://arxiv.org/html/2501.16650v1)). Even after optimal permutation, ViT FFN neurons correlate at only **0.385** across layers and attention at **0.210** ([Verma et al.](https://openreview.net/pdf?id=t8iuiH46g0)); these are activation correlations, a proxy for the weight-space R² the formula needs. LegoLM names large between-layer scale differences as the main obstacle to sharing ([LegoLM](https://arxiv.org/abs/2608.08652)) *[unreviewed]*, an obstacle that per-pass scales remove cheaply.

A worked estimate for a modern LLM *[derived]* assumes that the middle 60% of layers are candidates, attention is a third of block parameters, 30% of candidate heads reach R² ≈ 0.9 and FFN units sit at R² ≈ 0.15. With one keyframe per three-layer group, two-thirds of candidate layers are predicted and the saving is about 0.1 bpw, roughly 2.5% of int4's 4.125 bpw. If per-pass scales lift FFN R² to 0.5, the saving roughly doubles to about 5%. The 2–10% range quoted above brackets this arithmetic, leaving room for the census to find more redundancy than the literature suggests.

### A guaranteed floor and a trap

Give every tied pass a fitted branch gain αᵢ on its residual update, the scalar core of FiLM, and the method acquires a floor *[inference]*: setting α→0 turns that pass into a skipped layer, so a calibration-fitted loop can never be worse than dropping the layers whose unique weights it removed. That floor is measurable. Dropping 10 of Llama-2-13B's 40 layers keeps MMLU at 52.22 vs 55.0 ([ShortGPT](https://arxiv.org/html/2403.03853v1)), but a 25% cut on Llama-2-7B drops the zero-shot average from **62.8 to 47.4** ([FinerCut](https://arxiv.org/pdf/2405.18218)), and training-free weight blending (MKA) loses 2.82% MMLU at 43.75% layer compression on Llama-3-8B ([MKA](https://aclanthology.org/2024.emnlp-main.987/)). The trap is that savings on low-importance layers are equally available from mixed-precision bit allocation or plain layer dropping, with no sharing at all, so any sharing result has to beat those baselines and not only uniform int4.

### Verdict: infeasible as a 2–3× post-training loop, modest as a residual coder, valuable as stage 0

The evidence supports three verdicts at different confidence levels. With high confidence, **a calibration-only conversion of a pretrained LLM into a 2–3× looped model at usable quality is not feasible**: independent results show collapse, the rate-distortion arithmetic says the required redundancy is missing by orders of magnitude, and no source reports a counterexample. With medium confidence, **a calibration-fitted cross-depth residual coder on top of int4 is feasible but small for LLMs**, saving about 2–10% of bytes at int4-level quality with the gain concentrated in similar attention heads; the census can move that figure either way. Also with medium confidence, **the same machinery is valuable as a closed-form stage 0 for the user's distillation pipeline**, because it gives an aligned, Hessian-weighted initialization that beats Average and arrives with per-pass parameters already fitted. That should shorten stage A, but it cannot remove stage A for real 2–3× loops.

The contrast with fine-tunes is telling. BitDelta compresses a fine-tune's delta to one bit per weight at MT-Bench 6.47 vs 6.56, while a same-memory rank-128 delta reaches only 5.85 ([BitDelta](https://arxiv.org/html/2402.10193v3)). "Base plus 1-bit delta" works superbly across models that share a training history, but not across layers that never did. If the user ships several task variants of one backbone, cross-model delta compression is the proven route.

## Depth-shared residual quantization is the version that can work

The name is this report's. Every component has a published precedent, but no one has tested the combination.

### Represent each tied layer as an aligned base, per-pass scales and a coded residual

For a group g of tied layers, each shareable linear map in layer i is stored as:

`Ŵᵢ = diag(sᵢ) · Q(B_g) · diag(tᵢ) + Rᵢ`

Here Q(B_g) is a shared base quantized at 4 bits (3 at most) after a Hadamard rotation; sᵢ and tᵢ are fp16 per-pass output- and input-channel scales; Rᵢ is the per-layer residual, which is rank-r with 4-bit factors, a dense 1–3-bit code fitted against the decoded base, or zero (a pure loop for that unit); and αᵢ is a scalar or vector gain on the sublayer's residual-branch output. The user's three differentiation mechanisms map onto these components *[derived]*. Per-pass norm gains become tᵢ, because an RMSNorm gain folds into the next linear map's input columns, the computational invariance SliceGPT exploits ([SliceGPT](https://arxiv.org/abs/2401.15024)). FiLM-style γ and β on a branch become sᵢ, αᵢ and a bias, and per-pass LoRA becomes a rank-r Rᵢ. All three can be fitted in closed form, and if the student must run at unseen pass counts, regressing the fitted per-pass values on normalized depth yields a LARM-style smooth FiLM *[inference]*.

**Base.** Two choices compete. A keyframe base is one real layer coded directly, with the others predicted from its decoded form. It never loses to direct quantization in the Gaussian model, so it is the default when R² is low *[derived]*. A Hessian-weighted mean pays off only when R² is high and the residuals are low-rank. This mirrors RRT's finding that Stepwise wins for strict ties and Average for relaxed ones ([RRT](https://arxiv.org/html/2410.20672v1)).

**Residual form.** The census decides this from the whitened residual's spectrum. Sign structure beat low rank at equal memory for fine-tune deltas ([BitDelta](https://arxiv.org/html/2402.10193v3)), and 4-bit factors at rank 256 work best beside a 2-bit backbone ([CALDERA](https://github.com/pilancilab/caldera)). Nobody has measured which form inter-layer residuals favor.

**Granularity.** Tie heads and FFN neuron groups, not whole matrices, because R² varies widely between heads.

| Component | Storage cost *[derived]* | Fitted by |
|---|---|---|
| Shared base, int4 with g128 scales | 4.125/N bpw per tied layer, for N layers per base | GPTQ/QTIP with the summed Hessian H_Σ |
| Per-pass sᵢ, tᵢ in fp16 | ~0.008 bpw on a 4096×4096 matrix | Alternating least squares |
| Branch gain αᵢ and norm gains | Negligible | Least squares against the teacher's residual update |
| Rank-64 residual, 4-bit factors | 0.125 bpw (4096²) / 0.086 bpw (11008×4096) | QERA closed form, a whitened SVD ([QERA](https://proceedings.iclr.cc/paper_files/paper/2025/file/21718991f6acf19a42376b5c7a8668c5-Paper-Conference.pdf)) |
| Dense residual at b bits | b + 0.125 bpw | GPTQ on Wᵢ − ŝQ(B)t̂ with Hᵢ, coded against the decoded base |

### Share late-middle input-side projections; keep the edges and output-side writes private

Every source agrees on depth. The first one to three layers and the last one to two are distinct and sensitive and should stay unique ([Painters](https://arxiv.org/abs/2407.09298); [Lad et al.](https://arxiv.org/pdf/2406.19384); [Gromov et al.](https://arxiv.org/html/2403.17887v2)); MoR's best scheme, Middle-Cycle, keeps first and last unique ([MoR](https://arxiv.org/html/2507.10524v1)), and LegoLM protects boundary blocks for the same reason ([LegoLM](https://arxiv.org/abs/2608.08652)). The late middle is the place to tie. Second-half layers of Llama 3.1 and Qwen 3 contribute much less than first-half layers ([Csordás et al.](https://arxiv.org/html/2505.13898v1)), and in the Llama-3.2-1B retrofit a core built from layers 10–13, just before the coda, reached final loss **2.91** against 3.15–3.18 for early-middle layers and **3.84** for evenly spread layers 2, 5, 8 and 11 ([retrofit data](https://github.com/mcleish7/retrofitting-recurrence/tree/main/paper_plots)). ViTs follow suit, with blocks 8–12 of 12 the most prunable ([Elastic ViTs](https://arxiv.org/html/2510.17700v1)).

By sublayer, three published configurations together favor sharing projections that read from the residual stream and keeping private the ones that write layer-specific updates into it *[inference]*.

| Sublayer | Evidence | Recommendation |
|---|---|---|
| Q, K | Some head pairs reach ~0.99 cosine ([Head-wise](https://aclanthology.org/2024.findings-emnlp.144.pdf)); LiSA compresses Q/K 6× after training ([LiSA](https://github.com/takagi97/lisa)) | Share first, per head, with RoPE-compatible alignment |
| V | Shared in Basis Sharing's config ([config](https://raw.githubusercontent.com/TUDa-HWAI/Basis_Sharing/main/tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml)) | Share, aligned as a V/O pair |
| O | Private in Basis Sharing; MASA calls the output projection the bottleneck ([MASA](https://arxiv.org/abs/2508.04581)) | Keep private, or give a larger residual |
| Gate, up | Shared in Basis Sharing; DeltaLLM finds MLP deltas compress best ([DeltaLLM](https://arxiv.org/html/2501.18596v1)); GlowQ shares one factor across modules with the same input ([GlowQ](https://iclr.cc/virtual/2026/poster/10007786)) | Share after neuron-permutation alignment |
| Down | Private in Basis Sharing; FFN sharing is ALBERT's costly part ([ALBERT](https://arxiv.org/abs/1909.11942)) | Keep private, or give a larger residual |
| Norm gains | Per-pass by construction | Always per-pass |

The tie pattern needs testing *[inference]*. The user's A→B→A→B→A→B ties layers two apart: A at positions 1, 3, 5 and B at 2, 4, 6. Two findings favor tying contiguous layers instead (AAABBB). DOCS finds similarity highest between adjacent layers ([DOCS](https://arxiv.org/html/2501.16650v1)), and MobileLLM's back-to-back repeat is the only layout with measured near-free weight reuse at runtime ([MobileLLM](https://arxiv.org/pdf/2402.14905)). The census should score both patterns. Execution order never changes; only the alias table mapping depth slots to bases does.

### Align with free symmetries before averaging anything

Averaging several nearly orthogonal matrices shrinks and blurs them, which is the likely reason Stepwise beats Average for strict ties *[inference]*. LiSA reports that sharing attention without rearranging heads "proves to be ineffective" ([LiSA](https://github.com/takagi97/lisa)).

Four symmetries leave a layer's function unchanged and can be applied to each layer before tying: permuting FFN hidden units (gate/up rows together with down columns), permuting heads within a GQA group, per-head invertible maps on the V/O pair, and per-head maps on the Q/K pair, which under RoPE must commute with the rotary blocks, leaving only per-frequency 2×2 rotations or permutations *[inference]*.

The global residual-stream rotation used by QuaRot and SliceGPT cannot make two layers more alike, because it applies to all of them at once. It stays compatible with sharing for the same reason *[inference]*.

The tools exist. Verma et al. solve one linear assignment per FFN sublayer on activation correlations from about 10K tokens ([repo](https://github.com/nverma1/merging-ffs-compression)); head-wise sharing matches heads on concatenated Q‖K weights ([Head-wise](https://aclanthology.org/2024.findings-emnlp.144.pdf)); an ICML 2025 paper aligns attention by closed-form orthogonal Procrustes over its continuous rotation symmetry ([rotation symmetry](https://arxiv.org/pdf/2502.00264)); and SLOA warns that Q–K and V–O are composed operators that must be aligned as pairs and merged sequentially along the execution path ([SLOA](https://arxiv.org/abs/2610.06415)).

No published work combines these alignments with a looped tie schedule. Whether aligned Average beats Stepwise even for strict ties is an open, cheap experiment *[inference]*.

### One Hessian-summed objective is stage A in closed form

Take a group of layers i that share one tensor Ŵ, with calibration inputs Xᵢ and Hessians Hᵢ = XᵢXᵢᵀ. Their joint layer-wise objective Σᵢ‖(Wᵢ − Ŵ)Xᵢ‖² rearranges exactly into one standard quantization problem *[derived]*:

`tr((Ŵ − W*)·H_Σ·(Ŵ − W*)ᵀ) + C`, where `H_Σ = Σᵢ Hᵢ` and `W* = (Σᵢ WᵢHᵢ)·H_Σ⁻¹`

Three consequences follow. Existing solvers work unchanged, since GPTQ, AWQ, QuIP# or QTIP quantize the shared tensor when handed (W*, H_Σ). The constant C is the irreducible sharing error and the census's core number, and RRT's Average initialization is the special case Hᵢ = I. Most usefully, the gradient is a sum over tied positions, 2Σᵢ(Ŵ − Wᵢ)Hᵢ: **this is the closed-form counterpart of "gradients from every pass land on the shared block,"** so the PTQ fit solves stage A's per-layer linearized loss without a single optimizer step.

There is direct precedent. "Quantizing Looped Transformers" shows that GPTQ Hessians from the first recurrence step alone underperform round-to-nearest on 5 of 9 checkpoints. Accumulating Hessians across steps beats both on all 9 and recovers bf16-level accuracy on Huginn ([arXiv 2609.30820](https://arxiv.org/abs/2609.30820)) *[unreviewed]*.

Two refinements are worth adding *[inference]*. Weight each tied position by downstream sensitivity rather than activation energy alone, to resist the late-pass drift seen in looped training ([Loop Dropout](https://arxiv.org/abs/2609.34218)); per-row output scales make the Hessian row-dependent, and GuidedQuant is a precedent for keeping separate Hessians per row group ([GPTQ overview](https://www.emergentmind.com/topics/generative-pretrained-transformer-quantization-gptq)). And because each Xᵢ depends on the already-compressed layers before it, re-collect inputs through the compressed model and refit sequentially ([error propagation](https://arxiv.org/html/2504.09629v1)).

| Step | What is fitted | How | Closed form? |
|---|---|---|---|
| 1. Collect | Hᵢ for every linear map; per-layer importance (KL when the layer is dropped) | Forward passes over ~0.5M tokens or ~1K frames | Yes |
| 2. Group | Which layers, heads and neuron groups tie | R² census plus matching (GeoPair uses Blossom max-weight matching for pairs ([GeoPair](https://arxiv.org/abs/2609.25963v1))) | Yes |
| 3. Align | Permutations, V/O maps, RoPE-commuting Q/K maps | Linear assignment on activation correlations; Procrustes | Yes |
| 4. Initialize base | Keyframe, or W* = (ΣwᵢWᵢHᵢ)H_Σ⁻¹ | Pick whichever has the lower irreducible error C | Yes |
| 5. Scales and gains | sᵢ, tᵢ (with folded norm gains), αᵢ | Alternating least squares in the whitened metric; αᵢ matched to the teacher's residual update | Yes |
| 6. Residuals | Rᵢ, low-rank or dense low-bit | QERA, or GPTQ with Hᵢ against the decoded base | Yes |
| 7. Update and quantize base | B ← (Σwᵢ(Wᵢ−Rᵢ)Hᵢ)H_Σ⁻¹; Hadamard; GPTQ/QTIP with H_Σ | CALDERA-style alternation of steps 5–7, 3–5 rounds | Yes |
| 8. Re-collect | Student-side Xᵢ, then refit steps 5–6 in layer order | Propagation-aware pass | Yes |
| 9. Optional stage A-lite | Per-pass parameters and residual factors only, base frozen | Block-wise teacher-output matching, OmniQuant-style ([OmniQuant](https://arxiv.org/html/2308.13137v3)) | No (calibration-scale gradients) |
| 10. Select | Group sizes, bit budgets, ranks | End-to-end loss at deployment depth, as Loopy does ([Loopy](https://arxiv.org/abs/2610.05265)) | — |

Step 10 is mandatory because a better proxy can still mean a worse model ([arXiv 2605.30836](https://arxiv.org/abs/2605.30836)).

### Sharing beats int3 and int4 only where R² clears about 0.75

The comparison targets come from protocols that are not mutually comparable. On Llama-2-7B, WikiText-2, context 4096, with FP16 at 5.12:

| Baseline | Bits per weight | Perplexity | Source |
|---|---|---|---|
| 4-bit QTIP / QuIP# / AQLM | ~4 | 5.17–5.21 | [extract](https://arxiv.org/pdf/2510.20984) |
| 3-bit QuIP# | ~3 | 5.41 | [QuIP#](https://arxiv.org/abs/2402.04396) |
| 2-bit AQLM 1×16 | ~2 | 5.92 | [AQLM](https://github.com/Vahe1994/AQLM) |
| 2-bit AQLM with PV fine-tuning | ~2 | 5.68 | [AQLM](https://github.com/Vahe1994/AQLM) |
| ~1-bit AQLM 1×8g8 | ~1 | 7.85 | [AQLM](https://github.com/Vahe1994/AQLM) |

Scalar GPTQ on LLaMA-7B at context 2048 gives 6.21 at 4 bits and 16.28 at 3 bits ([SVD-LLM](https://arxiv.org/pdf/2403.07378)). Group scales add about 0.125 bpw, which is why Marlin's ideal batch-1 speedup is 3.87× rather than 4× ([Marlin](https://github.com/IST-DASLab/marlin)). This report therefore counts int4 as 4.125 bpw and int3 as 3.125.

| Configuration, three-layer groups *[derived]* | bpw | Fidelity target | R² needed on predicted units | Calibration-only outcome |
|---|---|---|---|---|
| Pure 3× loop, int4 base, per-pass scales | ~1.39 | int4 | ~0.996 | Collapse, as the evidence predicts |
| 3× loop + rank-64 4-bit residual | ~1.5 | int4 | Near 0.996 unless residuals are low-rank | Needs RRT-scale healing |
| Keyframe + two layers at 2-bit residuals | ~2.79 | int3 | ≥0.75 | Beats int3 by ~11%, only on high-R² units |
| Keyframe + two layers at 3-bit residuals | ~3.46 | int4 | ≥0.75 | Beats int4 by ~16%, only on high-R² units |
| Head-granular sharing of high-R² units, rest plain int4 | ~3.7–4.0 | int4 | ≥0.75 on shared units | 2–10% saving |

Codebook sharing is not the lever. Fixed lattice and trellis codebooks such as QuIP#'s E8P are already shared by every layer, and codebook storage is negligible at 2 bpw, so the win must come from sharing the codes themselves ([QuIP#](https://arxiv.org/abs/2402.04396)) *[inference]*.

The closest published cross-depth codec is MCWC. It treats depth like video, aligning layers, predicting each from already-decoded ones and entropy-coding quantized residuals. It reports perplexity 15.95 at about 4.2 bits against 16.40 for GPTQ-W4 and 15.33 dense, on an unnamed model ([MCWC](https://arxiv.org/html/2605.24754v1)) *[unreviewed]*. That gain is about the size the R² arithmetic predicts *[inference]*.

### Stack on int4 with loop-aware calibration rather than replacing it

Shared weights are more fragile under quantization, so the base needs at least the bits an untied layer would get. ALBERT keeps 93.4% of performance at 4 bits but only 72.5% at 3 bits ([quantization study](https://arxiv.org/pdf/2010.07109)). On Ouro-1.4B, group-128 INT8 is lossless (WikiText-2 10.69 vs 10.68), while INT4 round-to-nearest reaches 12.48 and cuts GSM8K at 8 loops from 78.5 to 60.0 ([repo](https://github.com/Nemo0412/Loop_Transformer_Quantization)) *[unreviewed]*. In weight-tied recursive reasoners, per-tensor 4-bit activations take Sudoku from 84.1% to 0% until per-block scaling restores it ([arXiv 2607.16237](https://arxiv.org/html/2607.16237)). The stacking rules follow *[inference]*: quantize the base at int4 or higher with a Hadamard rotation and the summed Hessian, keep per-pass scales, gains and norms in fp16, use MX-style block scaling if activations go to int4 or int8, give extra bits to the non-shared entry points that "Quantizing Looped Transformers" finds most exposed ([arXiv 2609.30820](https://arxiv.org/abs/2609.30820)), and budget all gains as sub-multiplicative.

Structure stacked on quantization has a mixed record:

| Combination | Size and perplexity | Compared with | Source |
|---|---|---|---|
| SVD-LLM (20% removed) + GPTQ-4bit, LLaMA-7B | 2.1 GB, 13.29 | GPTQ-3bit at 2.8 GB / 16.28; GPTQ-4bit at 6.21 | [SVD-LLM](https://arxiv.org/pdf/2403.07378) |
| Dobi-SVD + 4-bit, Llama-2-7B | 3.0 GB, 6.91 | 4-bit alone at 3.2 GB / 6.97 | [Dobi-SVD](https://arxiv.org/pdf/2502.02723) |

Structure plus int4 can beat a lower bit width at the same memory, but loses badly to plain int4. At equal quality, the gains are a few percent.

### ViT and per-frame video encoders are the most promising, least measured targets

Vision has the strongest near-PTQ evidence. ViTs have a much more uniform layer-to-layer representation structure than CNNs ([Raghu et al.](https://proceedings.neurips.cc/paper/2021/file/652cf38361a209088302ba2b8b7f51e0-Paper.pdf)). FiPS, a cross-layer shared base with sparse factors initialized by SVD and refined by block-wise reconstruction on calibration data, compresses DeiT-B and Swin-L MLPs to 25–40% of their parameters within 1 point of top-1 ([FiPS](https://arxiv.org/abs/2411.09816)), which is essentially rung one of the healing ladder below. Training-free entropy-based pruning removes up to 91.7% of VideoMAE-B blocks with competitive transfer, probably measured after downstream fine-tuning ([arXiv 2602.03918](https://arxiv.org/html/2602.03918v1)), which suggests masked video encoders are heavily over-deep.

Calibration is cheap for vision *[inference]*. Hessians come from patch tokens, so about 1K representative frames already yield hundreds of thousands of tokens. Many ViTs use learned or absolute position embeddings rather than RoPE, which frees Q/K alignment to use any per-head invertible map.

Per-frame encoding is compute-bound, so sharing buys storage and model switching, not frame rate. ViT-B/16 at 224 pixels takes about 17–20 ms on Orin Nano ([FastPose-ViT](https://arxiv.org/pdf/2512.09792)), and INT8 results on Orin are erratic: a ViT-S encoder gained nothing in isolation and its full encoder ran 2.7× slower ([NVIDIA forum](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835)), while RF-DETR gained 1.26× ([embedl](https://huggingface.co/embedl/rf-detr-base)).

The per-pass gains αᵢ also offer a natural knob for running fewer passes on frames that barely change, but no source has measured that *[inference]*.

## Download size is the dependable win; RAM needs kernels and speed needs cache residency

There are two deployment modes.

**Mode A is a transport codec** *[inference]*. It ships the base, residuals and scales, then rebuilds ordinary per-layer int4 weights at load time. It works on every runtime (ONNX Runtime, TensorRT, Core ML, LiteRT, WebLLM) at exactly int4's RAM and speed. To avoid rounding twice, code the residual against the final per-layer int4 grid so that reconstruction is exact.

**Mode B keeps the base resident.** One copy of the base stays in RAM alongside per-slot residual and scale tensors. RAM falls, but each pass now reads the base plus its residual. A dense 1-bit residual adds 25% more bytes than an int4 base, and a 2-bit residual adds 50% *[derived]*. Decode therefore slows unless the base stays cached.

Unfused low-rank side paths are measurably costly. A rank-48 unmerged LoRA on an H100 at batch 1 decoded at **21.0 vs 33.4 tok/s** ([CoLT-Drive](https://arxiv.org/pdf/2609.00242)), and llama.cpp applies runtime LoRA as separate matmul, scale and add operations ([llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-graph.cpp#L1615)).

| Target | Binding constraint (published) | Recommended mode |
|---|---|---|
| Browser (WebGPU + Wasm) | 128 MiB / 256 MiB default binding/buffer limits ([spec](https://github.com/gpuweb/gpuweb/blob/main/spec/index.bs)); ORT Web cannot exceed 4 GB ([ORT Web](https://onnxruntime.ai/docs/tutorials/web/large-models.html)); WebLLM's Llama-3.2-1B q4f16 needs 879 MB ([config](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts)) | Mode A, decoded in JS or WGSL into a standard graph. MLC/WebLLM has no looped-model path; download time and cache quota shrink |
| iPhone / Android | ~6 GB app ceiling *[secondary]*; 8–24 MB system cache; 1B Q4 decodes at 40–50 tok/s on iPhone 16 Pro, bandwidth-bound ([benchmarks](https://www.ertas.ai/blog/llm-run-on-iphone-benchmarks)) | Mode A by default. Mode B only where RAM binds, via Core ML's hash-based weight dedupe across functions ([coremltools](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/docs-guides/source/multifunction-models.md)) or ExecuTorch `.ptd` named data ([ExecuTorch](https://github.com/pytorch/executorch/blob/ad1c24d648b5a12bb51d488573fab94d83dac39e/docs/source/ptd-file-format.md)); use back-to-back repeat order if chasing cache reuse |
| Jetson Orin | 4 MB L2; Orin Nano's 8 GB is shared with the OS; measured 54.5 of 204.8 GB/s on AGX ([forum](https://forums.developer.nvidia.com/t/jetson-agx-orin-32gb-measured-memory-bandwidth-much-lower-than-theoretical-spec/331766)); TRT Edge-LLM supports only FP16/INT8/INT4 on Orin ([support matrix](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/getting_started/support-matrix.md)) | Mode A into TensorRT engines, for storage and fleet-update size *[inference]*. Edge-LLM documents LoRA sidecars ([lora.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/0f52705723abfa8c6702654bfdcc082a728ed353/docs/source/user_guide/features/lora.md)) but no looped models |
| llama.cpp / GGUF | `{arch}.num_loops` aliases physical blocks with a KV slot per pass ([constants.py](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/gguf-py/gguf/constants.py#L154)); LoRA is looked up by tensor name, so it cannot target aliased passes ([llama-adapter.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-adapter.cpp#L140)) | Mode B needs a new architecture: an arbitrary alias table, per-slot tensors (e.g. `blk.{slot}.attn_q_delta`), and fused or side-path residual kernels for CUDA, Metal, Vulkan and WGSL |
| ONNX | Loop bodies can read outer-scope initializers ([ONNX IR](https://github.com/onnx/onnx/blob/main/docs/IR.md)); dedupe passes merge identical initializers ([onnx-ir](https://onnx.ai/ir-py/api/ir_passes_common.html)) | An unrolled graph with shared initializers is safer for execution-provider coverage than Loop ops, which are unverified on mobile and WebGPU providers |

### Weight sharing leaves the KV cache untouched

A converted model keeps its full KV cache, because tied layers still see different inputs and so produce different keys and values. KV grows with passes times context. For a 1B-class GQA model that is about 2 KB per token per pass, roughly 128 MB for 16 passes at 4K context *[derived]*, which is comparable to the weights of a heavily shared model.

The training-free remedies are KV quantization and MiniCache, which merges similar KV across adjacent middle and deep layers for 1.53× compression alone and 5.02× with 4-bit ([MiniCache](https://neurips.cc/virtual/2024/poster/93380)). Cross-loop KV sharing belongs to models trained as loops: MoR shares first-recursion KV, Ouro shares last-step KV for about 4× at decode ([guide](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/)) *[secondary]*, and ResidualQuant stores an anchor plus 2-bit residuals across loops for 80.7% less KV storage ([ResidualQuant](https://arxiv.org/abs/2610.10381)) *[unreviewed]*. That last case is instructive, because cross-depth residual coding already works for KV in genuinely looped models, whose KV across loops is highly similar ([MoR](https://arxiv.org/html/2507.10524v1)). Per-frame vision encoders keep no autoregressive KV at all.

## Expected quality climbs a healing ladder from calibration to stage A and B

The ranges below are this report's estimates, anchored to the cited analogues. None has been measured for the proposed technique.

| Rung | What is fitted | Data | Expected LLM outcome *[inference]* | Expected ViT / per-frame outcome *[inference]* | Anchors |
|---|---|---|---|---|---|
| 0. Closed form only | Base, alignment, scales, gains, residuals | ~0.5M tokens or ~1K frames | As residual coder: int4-level quality at 2–10% fewer bytes. As strict 3× loop: collapse. Gains only: no worse than layer dropping (e.g. zero-shot 47.4 vs 62.8 at 25% of layers) | Similar single-digit savings; floor set by block dropping | [Painters](https://arxiv.org/abs/2407.09298); [FinerCut](https://arxiv.org/pdf/2405.18218); [MCWC](https://arxiv.org/html/2605.24754v1) |
| 1. Stage A-lite | Per-pass parameters and residual factors, base frozen; block-wise teacher-output matching | 10⁶–10⁷ tokens | Closes part of the proxy-to-end-to-end gap; perhaps 10–15% fewer bytes than int4 at int4 quality | FiPS-like: MLPs at 25–40% of FP parameters within 1 point, before int4 stacking | [OmniQuant](https://arxiv.org/html/2308.13137v3); [LoopQ](https://arxiv.org/abs/2605.16343); [FiPS](https://arxiv.org/abs/2411.09816) |
| 2. Stage A | Base unfrozen, gradients summed over tied positions; internal-feature MSE plus KL | 10⁷–10⁸ tokens | 12–24% fewer unique parameters at ~90% of performance | >21% of parameters removed at 99% accuracy | [DeltaLLM](https://arxiv.org/abs/2501.18596); [Verma et al.](https://openreview.net/pdf?id=t8iuiH46g0) |
| 3. Stage B | LM head plus last few layers, or the task head | Small | Beats LoRA healing after layer removal | Linear or task-head refit | [Reassessing Layer Pruning](https://arxiv.org/html/2411.15558) |
| 4. Uptraining with KD | Full looped student | 10⁹–6×10¹⁰ tokens | 30% at 1B tokens; 2× near parity at 15–60B | 2× at +1.0 top-1; 9.7× at −2.0 | [FlexiGPT](https://aclanthology.org/2025.naacl-long.31/); [RRT](https://arxiv.org/abs/2410.20672); [MiniViT](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MiniViT_Compressing_Vision_Transformers_With_Weight_Multiplexing_CVPR_2022_paper.html) |

The ladder maps directly onto the user's pipeline. Rung 0 is stage A solved in closed form, layer by layer, with the summed Hessian standing in for the summed gradient and the teacher's per-layer outputs as targets. Rung 1 is stage A restricted to the cheap parameters, which PTQ papers call block-wise reconstruction and which needs nothing beyond the user's existing feature-matching loss. Rung 2 is the user's current stage A on a far smaller token budget, started from an initialization that is already aligned, Hessian-weighted and per-pass calibrated instead of from Average or Stepwise. Rung 3 is stage B, unchanged.

When calibration alone falls short, the minimal healing budget sits at rungs 1–2, roughly **10⁶–10⁸ tokens** *[inference]*. The anchors are DeltaLLM's 30–40M tokens and the 3,000 QLoRA updates of the OLMo3-7B FFN-merging run. That budget buys 15–25% fewer unique parameters at a cost of a few points of quality.

A genuine 3× loop over a pretrained LLM, such as A→B→A→B→A→B, is a rung-4 project. RRT's B=3 results sit in an unretrieved appendix, and the retrofit data show that cutting unique layers to about a third is not rescued even by 132 passes.

During stage A, match the teacher's output at every tied position, as RecursiveVLM's per-recursion loss and LARM's checkpoint loss do ([RecursiveVLM](https://arxiv.org/html/2602.09080v1); [LARM](https://github.com/idiap/LARM)).

## A one-day redundancy census decides whether to continue

The first experiment costs no training and measures the one number that bounds every outcome above.

For token models, collect Hessians for every linear map from about 256×2048 calibration tokens, the budget Basis Sharing uses ([config](https://raw.githubusercontent.com/TUDa-HWAI/Basis_Sharing/main/tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml)); for vision and video models, use about 1K representative frames. Then, for every candidate pair and three-layer group in the middle 25–90% of depth, in both contiguous and distance-two (A→B→A→B) patterns and working per head and per FFN neuron group, align with permutations and Procrustes, fit per-pass scales, compute R² in the whitened metric, and record the whitened residual's spectrum (to choose between low-rank and dense residuals) along with each layer's drop-out importance. The output is a predicted bits-saved map from ½·log₂(1/(1−R²)) and a model-level byte saving at int4 fidelity, computed alongside the same saving for the two competing explanations, mixed-precision allocation and plain layer dropping. Hessian collection is ordinary forward passes and the largest assignment problems are about 11K×11K for a 7B-class FFN, so the whole census should fit in GPU-hours on one machine *[inference]*.

These gates are this report's proposals *[inference]*:

| Gate | Measured at | Kill if | Why |
|---|---|---|---|
| G0 | Census | Predicted saving at int4 fidelity under 0.2 bpw (≈5%), or no larger than mixed precision or layer dropping achieve without sharing | Sharing must earn bits that allocation alone cannot |
| G1 | Layer-wise proxy | At 2.8–3.5 bpw, base plus residual does not beat GPTQ/QTIP output error on at least half the shared matrices | The local objective is the method's own claim |
| G2 | End to end, calibration only | Not ≥5% fewer bytes than int4 at ≤ +0.1 WikiText-2 PPL and ≤0.5 points on zero-shot and generation tasks (top-1/mAP for vision); or the proxy improves while perplexity worsens | The 2605.30836 pattern; MMLU alone hides generative loss ([FinerCut](https://arxiv.org/pdf/2405.18218)) |
| G3 | Stage A-lite / stage A | More than 10⁸ tokens needed for ≥20% fewer bytes at int4 quality | Past this point it is a training project; budget it like RRT |
| G4 | Device | Mode B decode >15% slower where RAM is not binding, or Mode A load-time decoding unacceptable on a phone | Ship only the winning mode |

| Stage | Question | Method | Cost | Go condition |
|---|---|---|---|---|
| 0. Census | Is there cross-depth redundancy worth bits? | Alignment, R², spectra, importance, on both LLM and vision teachers | GPU-hours | Passes G0 |
| 1. Proxy bake-off | Does the representation beat direct quantization locally? | Steps 1–8 on a few groups at matched bpw vs GPTQ/QTIP | ~1 day | Passes G1 |
| 2. Calibration-only model | Does it hold end to end? | Full pipeline with sequential re-collection; PPL, zero-shot, generation; ImageNet, detection and per-frame metrics; contiguous vs A→B→A→B ties | 1–3 days | Passes G2 |
| 3. Stage A-lite | Do cheap parameters close the gap? | Block-wise distillation of per-pass parameters and residual factors, 10⁶–10⁷ tokens | Hours | Measurable gain over stage 2 |
| 4. Stage A + B | How far does light healing go? | Unfreeze the base, sum gradients over tied positions, 10⁷–10⁸ tokens; refit the head | Days | Passes G3 |
| 5. Device prototype | Does it ship? | Mode A codec plus a Mode B llama.cpp prototype; measure download, load time, RAM and tok/s on phone, browser and Orin Nano | Days | Passes G4 |
| 6. Optional true loop | Is a real 3× student worth the training? | RRT-style relaxed loop with FiLM, initialized from stage 4, using 10⁹–10¹⁰ tokens of the user's distillation | Weeks | Product decision |

The risks, in the order they are likely to bite *[inference]*, are that the census finds the low R² the literature predicts and the method shrinks to a minor add-on to int4; that newer models prove less tolerant, as Basis Sharing's +2.1 PPL on LLaMA-1-7B becomes +9.4 on Llama-3.1-8B ([LowRankArena](https://github.com/Zishan-Shao/lowrankarena)); that proxy objectives improve while the model worsens; that shared bases amplify quantization error; that Mode B needs per-backend kernels no runtime ships today; that vision redundancy, inferred from representations and training-heavy methods, fails to survive calibration-only fitting; and that anchor numbers still standing as unverified excerpts turn out to be wrong.

## Conclusion

Seen through quantization theory, the looping principle is predictive coding across depth, and its value is set entirely by how predictable one layer is from another. That one framing explains the literature: pretrained layers are nearly unpredictable from each other, so calibration-only looping collapses and training-free sharing earns little; fine-tunes are highly predictable from their base, so base-plus-1-bit-delta coding is near-lossless across models; and layers of a genuinely looped model are predictable from each other, so cross-loop residual coding works for their KV caches. Looping is therefore not a substitute for quantization of an existing model. Redundancy must exist before it can be coded, and for weights that redundancy is created by training (sharing from the start, distillation, or growth by duplication), not uncovered by calibration.

For the user, the practical consequence is to reorder the plan rather than abandon it. Build the census and the closed-form fitter first. For about a day's work they return either a small, shippable download-size codec on top of int4 or a clean kill. Either way they produce a far better stage 0 for the existing distillation pipeline: per-pass norms, FiLM gains and per-pass LoRA start from their least-squares values instead of identity, and the summed gradient that stage A already computes starts from the summed Hessian it approximates. Whether to build a true 3× looped student stays a decision about training budget and runtime support, not about calibration. RAM savings would need GGUF alias tables and per-slot tensors, any speedup would need back-to-back repeat order or early exit, and KV sharing would have to be trained in rather than bolted on.
