# Block-path search and layout for weight-shared / looped networks (fixed layouts vs per-input routing)

Source-type tags used throughout:
- **[primary]**: the authors' own GitHub README, config file or a results image in the official repository, read directly.
- **[search excerpt]**: text of the paper (arXiv/ACL/NVIDIA docs) as quoted by the web-search tool. arxiv.org, openreview.net, alphaxiv, papers.cool and Hugging Face were unreachable from this environment, so paper tables were read only through search excerpts.
- **[secondary]**: third-party summaries (blogs, review sites, social posts).

Notation: A, B, C = distinct parameter sets (blocks). "AABB" = each block repeated back-to-back (Takase's SEQUENCE / MobileLLM "immediate block-wise" / Loopie "layer-loop"). "ABAB" = the whole stack repeated (Takase's CYCLE / MobileLLM "repeat-all-over" / Ouro- and Huginn-style "model-loop"). "ABBA" = mirrored (Takase's CYCLE (REV) / MobileLLM "reverse sharing").

---

## Fixed sharing layouts and how they compare (Takase & Kiyono, MoR, MobileLLM, ALBERT, Subformer, MoEUT, Universal Transformer, ABAB vs AABB)

### Takeaway
Measured differences between ABAB and AABB orders are small: at most about 0.5 points on averaged benchmarks, and the winner changes with depth and scale. Two variables matter more: (a) keeping the first and last layers unshared (Middle-Cycle, Subformer's sandwich, Huginn's prelude/coda) and (b) which sub-block is shared, since sharing the FFN costs far more than sharing attention. Hardware locality can outweigh the small quality gap. MobileLLM shipped AABB because one block fits in SRAM, even though ABAB scored slightly higher.

### Cited Findings
**Takase & Kiyono, "Lessons on Parameter Sharing across Layers in Transformers" (arXiv 2104.06022, Apr 2021; later SustaiNLP 2023)**
- Definitions, from the repo's method figure [primary]: SEQUENCE has consecutive layers share (1,1,2,2,3,3 = AABB). CYCLE repeats the stack (1,2,3,1,2,3 = ABAB). CYCLE (REV) mirrors it (1,2,3,3,2,1 = ABBA). — [takase/share_layer_params](https://github.com/takase/share_layer_params)
- WMT En-De BLEU averaged over newstest2010–2016, from the repo's `result_table.png` [primary] — [result_table.png](https://github.com/takase/share_layer_params/blob/main/result_table.png):
  - Vanilla (M=6 unique, N=6 applied, 61M params, speed ×1.00): avg **26.57**. Admin: 26.48 (×0.97). Universal (M=1, N=6, 63M, ×0.48): **26.90**. Vanilla big (210M, ×0.39): 26.81.
  - M=6, N=12 (61M, ×0.63 speed): SEQUENCE (AABB) **27.13**, CYCLE (ABAB) **27.04**, CYCLE (REV) **27.08**.
  - M=6, N=18 (61M, ×0.47): SEQUENCE **27.02**, CYCLE **27.25**, CYCLE (REV) **27.31**.
  - newstest2014 alone at N=12: SEQUENCE 27.88, CYCLE 27.91, CYCLE (REV) 27.72. At N=18: 27.73 / 28.14 / 28.09.
  - So at 2× reuse AABB is ahead by 0.09 BLEU, and at 3× reuse ABAB is ahead by 0.23. Every shared model with 61M parameters beats the 210M vanilla-big model on average, at roughly the same speed (×0.47 vs ×0.39).
- The repo's worked example trains `--share-type cycle_reverse` (12 layers, 2 parameter sets). Setting the number of shared layers equal to the number of layers "shares parameters of one layer across all layers such as Universal Transformers" [primary]. — [takase/share_layer_params](https://github.com/takase/share_layer_params)
- A search summary describes the paper as a discrete search over the three configurations, not a learned one [search excerpt]. — [ResearchGate listing](https://www.researchgate.net/publication/372919326_Lessons_on_Parameter_Sharing_across_Layers_in_Transformers)

**Mixture-of-Recursions (MoR), Bae et al. (arXiv 2507.10524, Jul 2025; NeurIPS 2025 per the repo)**
- Strategies: Cycle (blocks reused cyclically), Sequence (each block reused consecutively), and Middle-Cycle / Middle-Sequence (first and last layers unique, intermediate weights shared) [search excerpt]. — [arXiv 2507.10524v3](https://arxiv.org/html/2507.10524v3)
- Appendix results: Middle-Cycle has the lowest validation NLL for 360M models at Nr=2 and Nr=3. At 135M with Nr=3 it beats Cycle (**3.1048 vs 3.1154**). Pure Sequence has the worst NLL in all four settings, and "its accuracy gap widens with recursion depth". Adding Middle helps Sequence, but it stays below the Cycle-based variants [search excerpt]. — [arXiv 2507.10524v3](https://arxiv.org/html/2507.10524v3)
- The Middle-Cycle margin widens as model size or depth increases [search excerpt]. — [arXiv 2507.10524v1](https://arxiv.org/html/2507.10524v1)
- The authors' example config is named `smollm-360m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001`, so the reference recipe is Middle-Cycle [primary]. — [raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)

**MobileLLM, Liu et al. (arXiv 2402.14905, Feb 2024; ICML 2024)**
- Average zero-shot accuracy by layer-sharing strategy [search excerpt]:
  - 125M: immediate block-wise (AABB) **45.0**, repeat-all-over (ABAB) **45.2**, reverse (ABBA) **44.8**.
  - 350M: **50.2 / 50.7 / 50.1** in the same order.
  - The paper's text says repeat-all-over performs best. — [arXiv 2402.14905v2](https://arxiv.org/html/2402.14905v2)
- The authors still chose immediate block-wise sharing. Their reasoning: on-device compute SRAM is about 20 MB, which "is usually only sufficient to hold a single transformer block", so running a shared block twice back-to-back avoids moving weights between SRAM and DRAM [search excerpt]. — [arXiv 2402.14905v2](https://arxiv.org/html/2402.14905v2)
- Released models, average over 8 zero-shot tasks [primary]: MobileLLM-125M 46.3 → MobileLLM-LS-125M **47.0** (+0.7). MobileLLM-350M 51.3 → LS-350M **52.1** (+0.8). Parameter count is unchanged and depth doubles. — [facebookresearch/MobileLLM](https://github.com/facebookresearch/MobileLLM)
- A third-party summary wrongly says immediate block-wise performed best, which contradicts the paper's table [search excerpt / secondary]. — [Papers Explained 216](https://ritvik19.medium.com/papers-explained-216-mobilellm-2d7fdd5acd86)

**Loopie, "Loop the Loopies!" (arXiv 2607.16051, Jul 2026)**
- Uses a "layer-loop" schedule in which each attention/MoE layer recurs before passing its output on (L1→L1→L2→L2, i.e. AABB). This contrasts with the "model-loop" (1,2,3,1,2,3) of Ouro and Huginn. Models: 20B total / 2B active and 6B / 0.6B active, both with 2 loop steps. The authors report that it beats compute-matched vanilla baselines, including a vanilla 30B-A3B model [search excerpt]. — [arXiv 2607.16051](https://arxiv.org/html/2607.16051)
- An ablation on 6B-A0.6B removes the layer-loop. A secondary write-up says performance dropped, but the size of the drop was not visible [secondary]. — [Loopie Substack](https://aininza.substack.com/p/loopie-how-looped-transformers-match). The claim that layer-loop "saves activation memory, improves throughput" comes from an alphaXiv post [secondary]. — [alphaXiv on X](https://x.com/askalphaxiv/status/2079283518224691582)

**ALBERT (arXiv 1909.11942; ICLR 2020), all-shared ablation (Table 4)** [search excerpt; one value is from a mirror]
- E=768: not-shared 108M / avg 82.3; shared-attention 83M / 81.6; shared-FFN 57M / 79.5; all-shared 31M / 79.8.
- E=128: not-shared 89M / 81.6 (the mirror also lists it as 81.7); shared-attention 64M / 81.7; shared-FFN 38M / 80.2; all-shared 12M / 80.1.
- The paper says most of the loss comes from sharing the FFN, and that sharing attention alone costs nothing at E=128. — [ALBERT arXiv](https://arxiv.org/pdf/1909.11942); [aiwiki mirror](https://aiwiki.ai/wiki/albert-base-v2_model)

**Subformer (Reid et al., Findings of EMNLP 2021)**
- Sandwich-style sharing: "share all the layers in a block except the first and last". The central shared layers act as a large representation learner and the unshared ends specialize. Generative-Transformer parameters drop by 25–70% "while retaining performance" [primary]. — [machelreid/subformer](https://github.com/machelreid/subformer)

**MoEUT (Csordás et al., arXiv 2405.16039; NeurIPS 2024)**
- `group_size` G is "the number of non-shared layers in the group", and the code defaults to G=2 [primary]. — [RobertCsordas/moeut](https://github.com/RobertCsordas/moeut)
- Paper Fig. 6 (244M models): "A small group size of G=2 works the best" [search excerpt]. — [arXiv 2405.16039](https://arxiv.org/html/2405.16039)

**Sandwich Transformer: sublayer-order search (Press, Smith & Levy, ACL 2020)**
- Reordering self-attention (s) and feed-forward (f) sublayers without changing parameter count improves results. The sandwich pattern puts more s at the bottom and more f at the top (`s.s.sf.f.f` notation). enwik8: 209M params, **0.968** test bpc. text8: 1.076 test bpc [primary]. — [ofirpress/sandwich_transformer](https://github.com/ofirpress/sandwich_transformer)

**Which sub-block to loop (2026)**
- MixerLoop (arXiv 2608.18230, Aug 2026) repeats only the token-mixing component and runs the FFN once. At 15M parameters it beats full-block recurrence on aggregate CORE. At 110M it keeps about 41.5% of the full-recurrence gain while cutting recurrent-backbone projection FLOPs by about 45.9% [search excerpt]. — [arXiv 2608.18230](https://arxiv.org/pdf/2608.18230)
- Looped Diffusion LMs (arXiv 2605.26106, May 2026): looping a small group of early-middle layers captures most of the gains, and looping more layers adds cost without benefit [search excerpt]. — [arXiv 2605.26106](https://arxiv.org/html/2605.26106)

### Inferences
- For a two-block A/B student, the data predict at most a few tenths of a point between A→B→A→B and A→A→B→B. Choose the order on hardware grounds:
  - AABB keeps one block's weights hot in SRAM/L2 across consecutive applications, which is MobileLLM's argument.
  - ABAB is the natural form for a "loop the whole core N times" runtime with a variable N.
- Two levers are better supported than order: unique first and last layers (Middle-Cycle, sandwich, prelude/coda), and sharing attention more aggressively than the FFN (ALBERT).
- The evidence goes both ways. Takase favours AABB at 2× reuse and ABAB/ABBA at 3×. MobileLLM favours ABAB. MoR finds Sequence (AABB) worst. Loopie (2026, MoE, 20B) chose AABB layer-loop. Order is not a first-order effect, and the result may depend on whether the blocks are dense or MoE.

### Gaps
- No source compares exactly A→B→A→B against A→A→B→B for a **distilled** student (teacher-matched internals). All comparisons above are pretrain-from-scratch, except the post-hoc analyses of pretrained LLMs.
- I could not read MoR's full sharing-strategy table (few-shot accuracy for each strategy), Subformer's sandwich vs all-shared ablation numbers, or Loopie's layer-loop vs model-loop numbers, because arXiv was blocked and excerpts were truncated.
- No Universal Transformer (2018) numbers were collected beyond Takase's "Universal" row.

---

## Learned / soft sharing patterns (Savarese & Maire, template banks, NAS over sharing schedules, per-layer coefficients)

### Takeaway
Learned soft sharing does exist. Savarese & Maire (2019) give each layer learned mixing coefficients over a shared template bank, and the trained networks often converge to near-strict recurrence that can be folded into real loops. However, I found no transformer paper that runs a differentiable or NAS search over *which block each layer uses*. In practice, sharing schedules are hand-designed or chosen by small discrete sweeps. In 2026 the "soft" axis moved into MoE: per-layer routers over expert pools shared across adjacent layers (Expert Tying, MoRE).

### Cited Findings
- **Savarese & Maire, "Learning Implicitly Recurrent CNNs Through Parameter Sharing" (ICLR 2019)**:
  - The network keeps a global bank of k templates, and each layer's weights are a learned linear combination of them. Limiting k gives a CNN/RNN hybrid. Trained networks "often end up with near-strict recurrent structure" and can be converted into networks with actual loops "with negligible side effects". Accuracy is competitive with NAS-designed networks [search excerpt]. — [OpenReview](https://openreview.net/forum?id=rJgYxn09Fm); [mlanthology](https://mlanthology.org/iclr/2019/savarese2019iclr-learning/)
  - Implementation: a `TemplateBank` plus `SConv2d` layers, with the bank passed to each layer. The reference model is SWRN-28-10-6 (bank of 6 templates) on CIFAR. The authors warn not to apply weight decay to the coefficients [primary]. — [lolemacs/soft-sharing](https://github.com/lolemacs/soft-sharing)
- **NAS and sharing**: AutoTrans (arXiv 2009.02070) uses a *fixed* cross-layer sharing rule during search to speed it up, rather than searching the sharing pattern. The search tool found no paper that learns a per-layer sharing mask differentiably [search excerpt]. — [AutoTrans](https://arxiv.org/pdf/2009.02070)
- **Expert Tying, "Tying the Loop" (Jaggi, arXiv 2606.16825, Jun 2026)**:
  - Consecutive layers share expert FFN weights in groups of g, while routers, attention and norms stay layer-specific. Tied experts get a learning-rate divisor of √g [primary]. — [epfml/looped-moe](https://github.com/epfml/looped-moe)
  - On OLMoE, Qwen3-MoE and DeepSeekMoE up to 7B, g=4 cuts total parameters by up to **52%** with negligible quality loss and gives up to **23.7%** wall-clock training speed-up [search excerpt]. — [arXiv 2606.16825](https://arxiv.org/abs/2606.16825)
- **MoRE, "Mixture of Reused Experts" (arXiv 2609.18176, Sep 2026; COLM 2026)**:
  - With reuse factor R=2, adjacent layers share a pool of 2N experts and each layer keeps its own router. Learnable depth embeddings let shared experts behave differently at each depth.
  - Tested at 114M–1.15B, with lower perplexity than standard MoE and weight-sharing baselines at matched compute and parameters.
  - The same expert is chosen in both layers of a reuse block about **10.3%** of the time at 1.15B, against about 5.0% at random. Early layers reuse at nearly double the random rate, and reuse falls with depth [search excerpt]. — [arXiv 2609.18176](https://arxiv.org/html/2609.18176)
- **CS-MoE (arXiv 2609.22199)** shares expert pools across layers as a fixed architecture, with no search [search excerpt]. — [arXiv 2609.22199](https://arxiv.org/abs/2609.22199)
- **Dual-Path (arXiv 2605.30202, May 2026)**: a router weighs two sublayers, a shared-parameter sublayer applied K times and a wider sublayer applied once [search excerpt]. — [arXiv 2605.30202](https://arxiv.org/html/2605.30202)
- **Relaxed / Recursive Transformers (arXiv 2410.20672)**: the MoR authors say this earlier work shares parameters so that tokens at different depths can be batched together, but that "two separate training processes for integrating parameter sharing and early-exiting degraded performance" [primary, MoR README]. — [raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)

### Inferences
- A cheap way to *search* the layout of a distilled looped student is the Savarese recipe: give each of the N applied positions coefficients over a bank of K templates (K=2 for A/B), train during the teacher-matching stage, then read the coefficient matrix and snap it to the nearest hard layout (ABAB, AABB, ABBA, ...). This adapts the CNN result and has not been validated on transformers.
- MoRE's 10.3% vs 5.0% reuse rate suggests that per-token routers given a shared pool reuse the same module only modestly above chance. Learned routing does not automatically collapse to a strict recurrence.

### Gaps
- I found no exact CIFAR/ImageNet numbers or coefficient-similarity matrices from Savarese & Maire (paper blocked).
- I found no transformer paper with a differentiable (DARTS-style) or RL search over which shared block each layer uses. The search tool explicitly reported no direct match.
- I did not verify Basis Sharing (shared SVD bases with per-layer coefficients) or hypernetwork-generated layer weights.

---

## Per-input path search over existing layers or shared blocks (CoLa, PoLar, Dr.LLM, FlexiDepth, Router-Tuning, LayerSkip, Layers-as-Painters, MoR, MoM, 2025–2026 routers)

### Takeaway
Searched or oracle per-input paths through a frozen pretrained LLM give very large gains. CoLa/PoLar find shorter correct paths for over 75% of already-correct samples and fix over 60% of wrong ones. Learned routers recover only part of this:
- Dr.LLM: +3.4 points while saving about 5 layers.
- PoLar: consistent gains with fewer layers.
- An independent replication (Sep 2026) found that PoLar's single-shot router's top-1 program collapses back to the standard pass. Only its top-k set helps, and a few generic programs cover most questions.

Per-token routers over shared banks (MoR, MoM, MoRE, TaH) are trained jointly and work in causal decoding, but they need special KV-cache and batching handling.

### Cited Findings
**CoLa: "Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs" (Li, Li & Zhou, UMD; arXiv 2507.07996, Jul 2025). Training-free, per-input oracle search.**
- Each pretrained layer can be skipped, repeated like an RNN, and stacked into a per-sample chain-of-layers. Paths use only the original layers, with no new weights. MCTS starts from the full stack and returns Pareto paths trading accuracy against length. MCTS was preferred over greedy or beam search because "optimal paths are long or non-trivial" [search excerpt]. — [arXiv 2507.07996](https://arxiv.org/abs/2507.07996)
- For over **75%** of correctly predicted samples, MCTS finds a shorter path that stays correct. For over **60%** of incorrectly predicted samples, it finds a path that becomes correct [search excerpt]. — [HF paper page](https://huggingface.co/papers/2507.07996); [paper_notes](https://github.com/AkihikoWatanabe/paper_notes/issues/2610)
- A summary reports Llama-3-8B-Instruct on DART-2 rising from 6.0% to 66.2% with CoLa [secondary]. This is an oracle per-sample search number, not a deployable router. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/skip-a-layer-or-loop-it-test-time-depth-adaptation-of-pretrained-llms/wrk_3dc253e95111a84b5bcc40d831bb4944)

**PoLar: "Skip a Layer or Loop It? Learning Program-of-Layers in LLMs" (same group; arXiv 2606.06574, Jun 2026; ICML 2026 Oral). Frozen LLM, learned single-shot predictor.**
- Pretrained layers are "packed as modules" (contiguous segments), and each segment is skipped, kept or repeated to form a per-input program [primary].
- Offline MCTS discovers valid programs. A lightweight predictor, with beam-decoding helpers, is trained on them and outputs the program for a new input [primary].
- Models: Llama-3.2-3B-Instruct, Qwen1.5-MoE-A2.7B-Chat, Qwen2.5-3B-Instruct, Qwen3-8B. Benchmark: DART-Math difficulties 1–5. The authors claim gains over standard inference and prior dynamic-depth methods, "often while executing fewer layers", and say the gains persist out-of-distribution [primary]. — [tianyi-lab/PoLar](https://github.com/tianyi-lab/PoLar)
- Supervision format: lists of layer indices, where repeated indices mean recurrence (e.g. `[0,1,2,2,3,4,5]`) and omitted indices mean skips. Each sample includes `gt_ans`, `final_valid_transitions` and `final_invalid_transitions` [primary]. — [tianyi-lab/PoLar](https://github.com/tianyi-lab/PoLar)
- The predictor adds "about 0.8% inference overhead" [search excerpt, HF page] and has about 2.1M parameters [secondary]. Recurrence alone beats skipping alone, and combining both is best [search excerpt]. — [HF paper page](https://huggingface.co/papers/2606.06574); [ArXivIQ](https://arxiviq.substack.com/p/skip-a-layer-or-loop-it-learning); [ResearchGate](https://www.researchgate.net/publication/406351133_Skip_a_Layer_or_Loop_It_Learning_Program-of-Layers_in_LLMs)
- PoLar's authors criticise Dr.LLM-style routing because it "interleave[s] local routing decisions during execution, limiting global depth coordination". They argue that predicting the whole program up front supports multi-layer recurrent modules [search excerpt]. — [arXiv 2606.06574](https://arxiv.org/html/2606.06574v1)

**Independent replication: "Programs-of-Layers in LLMs through the Lens of Cortical Areas" (Westerhoff et al., arXiv 2609.31360, Sep 2026)**
- Reproduced:
  - Skipping beats the standard pass, repeating beats skipping, and both together beat either.
  - Easy questions need shorter programs, and hard questions need more repeats.
- Not reproduced: the single-shot router. Its "top-ranked prediction consistently collapsed back to the standard pass", although the top-k predicted programs together gave a real accuracy gain.
- "A small number of generic programs are enough to solve most of the questions" [search excerpt]. — [arXiv 2609.31360](https://arxiv.org/html/2609.31360)
- Qwen3-8B on DART-Math, base vs skip&repeat (search-based) [search excerpt]:
  - DM-1: 48.6% → **88.8%** (+40.2).
  - DM-5: 11.2% → **47.2%**. — [arXiv 2609.31360](https://arxiv.org/html/2609.31360)

**Dr.LLM: Dynamic Layer Routing in LLMs (Heakl et al.; arXiv 2510.12773, Oct 2025; ICLR 2026). Frozen base with trained routers.**
- Each layer has a router (Linear-GELU-Linear, hidden size 128) that chooses skip, execute or repeat-once from windowed mean-pooled hidden states (8 windows). Router parameters total 11M for 3B models, or **0.14%** of weights [primary].
- Training: focal loss with class rebalancing (β=0.999, γ=2) and teacher forcing [primary].
- Supervision: length-aware MCTS with **50 simulations per example**, paths capped at 2L, **4K** MCTS-derived examples. Router training takes 25 epochs and **under 4 hours on one A100-40GB** [primary].
- Results: **+3.4 points** on ARC and DART while saving **about 5 layers** per input. Only a **0.85%** drop on out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval). Beats LayerSkip, FlexiDepth and MindSkip by up to **+7.7 points** [primary].
- Evaluation scripts use `--batch_size 1`. The model class exposes an `is_static_routing` flag [primary]. — [parameterlab/dr-llm](https://github.com/parameterlab/dr-llm)

**FlexiDepth: "Adaptive Layer-skipping in Pre-trained LLMs" (Luo et al., arXiv 2503.23798, Mar 2025). Per-token, per-layer router plus adapter, frozen base.**
- On Llama-3-8B, skipping **8 of 32** layers keeps full (100%) benchmark performance [search excerpt]. A summary of the main table puts retention at about 100.7%, against 43.9% for LayerSkip and 32.0% for ShortGPT at 8 skipped layers [secondary]. — [arXiv 2503.23798](https://arxiv.org/html/2503.23798v3)
- Repetitive or fixed-phrase tokens need fewer layers, and computation-heavy or high-uncertainty tokens need more [search excerpt]. — [arXiv 2503.23798](https://arxiv.org/html/2503.23798v3)
- The repo describes an updated two-stage recipe: tulu-v3 with skip penalty 1e-4, then open-perfectblend with 1e-5 [primary]. — [luoxuan-cs/FlexiDepth](https://github.com/luoxuan-cs/FlexiDepth)

**Router-Tuning / MindSkip (He et al.; arXiv 2410.13184, Oct 2024; EMNLP 2025)**
- Paper headline: **21% speedup with a 0.2% performance drop**. Routers are under 0.01% of parameters, and training takes under 30 minutes on an RTX A6000, "1000 times faster than DLO" [search excerpt]. — [arXiv 2410.13184v6](https://arxiv.org/html/2410.13184v6)
- The README table claims larger savings: LLaMA-2-7B at 50% capacity scores 58.0 vs 58.4 dense with about 48% FLOPs saved and 1.45× speedup. Mistral-7B at 60% capacity scores 63.5 vs 63.8 with 1.35×. Static layer drop at 50% falls to 46.2 [primary]. These numbers do not match the paper's 21% headline and may describe a different setting, so treat them with caution. — [CASE-Lab-UMD/Router-Tuning](https://github.com/CASE-Lab-UMD/Router-Tuning)

**LayerSkip (Elhoushi et al., arXiv 2404.16710; ACL 2024). Early exit plus self-speculative decoding.**
- Speedups of up to **2.16×** on CNN/DM, **1.82×** on coding and **2.0×** on TOPv2 [search excerpt]. — [Meta AI](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding)
- Requires LayerSkip-trained checkpoints. Speedups appear only on generation tasks, not multiple-choice. Tunables are `exit_layer` and `num_speculations` [primary]. — [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

**Transformer Layers as Painters (Sun et al., arXiv 2407.09298; AAAI 2025). Training-free robustness tests.**
- Middle layers can be dropped, reversed or randomly reordered with graceful degradation on Llama2-7B and BERT-Large. Early layers are not robust to this. "Looped Parallel" and "Random Layer Order" degrade least [search excerpt / secondary]. — [arXiv 2407.09298v4](https://arxiv.org/html/2407.09298v4); [Liner review](https://liner.com/review/transformer-layers-as-painters)

**Mixture-of-Recursions: per-token depth over one shared stack (trained from scratch)**
- Routers assign each token a recursion depth. Expert-choice routing takes top-k tokens at each recursion; token-choice assigns a depth up front and needs padding for variable lengths. Per recursion, the implementation gathers the selected tokens, runs the shared block once and scatters the results back [primary].
- Recursion-wise KV caching stores KV only for the tokens active at each recursion. Claimed result: "up to 2× greater inference throughput" than standard transformers at similar accuracy [primary]. — [raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)
- MoR-2 with expert-choice routing at 16.5e18 FLOPs reaches **43.1%** average few-shot accuracy vs **42.3%** for vanilla, with about 50% fewer parameters. MoR trails vanilla at 135M and matches or exceeds it above 360M [search excerpt]. A search summary also cites "up to 2.18×" throughput, which I could not verify. — [arXiv 2507.10524v1](https://arxiv.org/html/2507.10524v1)

**Mixture-of-Modules (MoM; Gong et al., arXiv 2407.06677; EMNLP 2024). The closest match to "router picks among K shared blocks each step".**
- The pool holds N_A attention modules, N_F FFN modules and an identity "skip" module.
- Two GRU-based routers pick K modules per step for H assembly steps. Modules are position-independent, so a token can reuse or jump to any module.
- Can be initialised from a vanilla Transformer in two phases.
- MoM-large is claimed to reduce forward TFLOPs by 16% and memory by 42% at comparable performance [search excerpt / secondary]. — [arXiv 2407.06677](https://arxiv.org/abs/2407.06677); [Moonlight review](https://www.themoonlight.io/en/review/mixture-of-modules-reinventing-transformers-as-dynamic-assemblies-of-modules)

**Think-at-Hard (TaH, arXiv 2511.08577, Nov 2025) and TaH2 (arXiv 2609.35748, Sep 2026). Per-token selective second iteration.**
- TaH: a lightweight decider plus duo-causal attention. It beats fixed two-iteration baselines by **8–11%** while skipping **94%** of second iterations, and beats single-iteration Qwen3 by 4–5%. A LoRA adapter is applied only to tokens marked "hard" in later iterations [primary].
- TaH2: jointly trains backbone and decider with online lookahead labels. It improves the AIME accuracy-compute slope by **53%** over non-looped and peak accuracy by about **3.4 points** at matched test-time compute. It ships with an inference engine adapted from mini-SGLang [primary]. — [thu-nics/TaH](https://github.com/thu-nics/TaH)

**Related**
- KnapSpec (arXiv 2602.20217) frames adaptive layer selection for self-speculative drafting as a knapsack problem [search excerpt, title only]. — [arXiv 2602.20217](https://arxiv.org/pdf/2602.20217)
- MoRE and Expert Tying (see the previous section) are 2026 per-token routers over expert banks shared across adjacent layers.

### Inferences
- The "Rubik's cube" view is well supported for **frozen pretrained, non-looped** LLMs. The space of skip/repeat programs contains far better per-input paths, but oracle search uses the gold answer (PoLar's supervision includes `gt_ans` and valid/invalid paths). Expect a large gap between oracle and routed gains, and the replication confirms this.
- For a looped student distilled from a teacher, the closest trained analogues are MoR (depth per token), TaH (extra iteration plus refine adapter per token) and MoM (choose among a module pool per step). None of them reports learning a non-trivial *order* over two distinct shared blocks. Learned routing chooses "how many" far more than "which".
- Dr.LLM's pooled-window router input and `batch_size 1` evaluation suggest a per-sequence decision rather than a per-generated-token one. Per-sequence routing is the deployable variant (see the deployability section). This is not verified from the paper.

### Gaps
- PoLar's own main-table numbers (accuracy vs layers executed against Dr.LLM, FlexiDepth and MindSkip) were not readable. The out-of-distribution table in the search excerpt was garbled.
- The cost of CoLa's MCTS per sample (simulations, wall-clock) was not found.
- Whether FlexiDepth and Dr.LLM deliver wall-clock speedups in batched causal decoding (as opposed to fewer executed layers) was not established.
- I found no 2025–2026 paper with per-token routing over a bank of K≥2 *distinct shared dense transformer blocks* trained from scratch in a looped model. A search tool explicitly reported finding none. MoM (2024) and MoRE (2026, MoE-level) are the closest.

---

## Phase decomposition analogues (coarse block then refine block; prelude/core/coda; changing token set or resolution between phases)

### Takeaway
The widely deployed "two-phase" pattern is prelude → looped core → coda (Huginn; MoR's Middle-Cycle; Subformer's sandwich). The closer Kociemba analogue, "first phase uses everything, later phases use a restricted or specialised move set", appears in four forms:
- TaH: later iterations add a LoRA and run only on hard tokens.
- TRM/HRM: two update types on fixed schedules.
- MoR: the active token set shrinks per recursion.
- AR-Net / FrameExit: per-frame resolution or early stopping in video.

### Cited Findings
- **Huginn-0125 (Geiping et al., arXiv 2502.05171, Feb 2025)**:
  - Config `nebel-raven-3.5b`: `n_layers_in_prelude=2`, `n_layers_in_recurrent_block=4`, `n_layers_in_coda=2`, `mean_recurrence=32`, `injection_type="linear"`, `sampling_scheme="poisson-lognormal-filling"`, `n_embd=5280`, `block_class_name="SandwichBlock"` [primary]. — [model_registry.py](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_registry.py)
  - Trained on 4096 AMD GPUs (Frontier). vLLM inference plugin available [primary]. — [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
- **Retrofitted recurrence (McLeish et al., arXiv 2511.07384, Nov 2025)** converts pretrained TinyLlama, Llama and OLMo into Huginn-style prelude/recurrent/coda models. Conversion scripts check that hidden states match exactly before training [primary]. — [mcleish7/retrofitting-recurrence](https://github.com/mcleish7/retrofitting-recurrence)
- **MoR Middle-Cycle and Subformer's sandwich** both keep unique first and last layers around a shared middle, and are best or competitive in their papers (see the first section) [primary / search excerpt]. — [MoR repo](https://github.com/raymin0223/mixture_of_recursions); [Subformer repo](https://github.com/machelreid/subformer)
- **TaH**: iteration 1 is the plain backbone. Later iterations activate a LoRA adapter only on tokens the decider marks "hard", so the refine phase uses a different, specialised block [primary]. — [thu-nics/TaH](https://github.com/thu-nics/TaH)
- **TRM: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871, Oct 2025)**:
  - A single tiny network alternates two move types: it updates latent z n times from (x, y, z), then updates answer y from (y, z), for up to K improvement steps.
  - 7M parameters, 45% on ARC-AGI-1 and 8% on ARC-AGI-2 [primary]. — [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- **HRM (arXiv 2506.21734, Jun 2025)**: two interdependent recurrent modules, a slow high-level planner and a fast low-level module. 27M parameters, trained on 1000 samples [primary]. — [sapientinc/HRM](https://github.com/sapientinc/HRM)
- **MoR token-set shrinkage**: expert-choice routing selects top-k tokens at each recursion and runs the shared block only on them, so deeper phases act on fewer tokens [primary]. — [MoR repo](https://github.com/raymin0223/mixture_of_recursions)
- **Video, per-frame**:
  - AR-Net (ECCV 2020): a lightweight policy network picks each frame's input resolution, or skips the frame, on the fly. Trained with Gumbel-Softmax [search excerpt]. — [arXiv 2007.15796](https://arxiv.org/abs/2007.15796)
  - FrameExit (CVPR 2021): a gating policy picks the earliest exit point in time. With EfficientNet-b3 it reports +3.9% mAP on ActivityNet and +2.5% on Mini-Kinetics at 2.3× / 2.5× less compute than AR-Net [secondary]. — [Liner review](https://liner.com/review/frameexit-conditional-early-exiting-for-efficient-video-recognition)

### Inferences
- A literal Kociemba design for an A/B student would be phase 1 = (A→B)×k with the full block set, then phase 2 = B-only (or a B+LoRA "refine" block) on the tokens or frames still unresolved. TaH is the closest validated instance: a different block for later iterations, applied only to hard tokens, with +8–11% at 6% extra second-iteration use.
- For per-frame vision, AR-Net and FrameExit show the deployable form of a phase change: discrete per-frame choices from a small menu (resolution, exit point) driven by a cheap policy network.

### Gaps
- I found no paper that trains a looped model with an explicit "coarse block for the first iterations, then a different refine block" schedule and compares it against ABAB, apart from TaH's LoRA-in-later-iterations.
- No numbers on changing spatial resolution *between loop iterations* of a weight-shared ViT (coarse-to-fine looping) were found in this pass.

---

## Search costs at inference and how they are amortized

### Takeaway
Every per-input search paper runs MCTS **offline** and amortises it in one of three ways:
- Distil it into a router. Dr.LLM: 50 simulations × 4K examples, router trained in under 4 hours on one A100. PoLar: about 2.1M-parameter predictor with about 0.8% inference overhead.
- Use it as an analysis tool. CoLa.
- Reduce it to a fixed layout. Takase-style sweeps, the Sandwich Transformer, and the replication finding that a handful of generic programs covers most inputs.

No paper ran MCTS at deployment.

### Cited Findings
- Dr.LLM: MCTS with `--num_simulations 50` per example and `--budget 2` (path length capped at 2L). Routers are supervised with 4K MCTS paths. Training takes 25 epochs at LR 1e-3 on one A100-40GB in under 4 hours. The base model stays frozen, and only `model.model.routers` (11M params for 3B models) is trained [primary]. — [parameterlab/dr-llm](https://github.com/parameterlab/dr-llm)
- PoLar: "The paper uses MCTS as an offline tool to discover valid execution programs"; the predictor learns from them. The offline MCTS code was released in Sep 2026 [primary]. — [tianyi-lab/PoLar](https://github.com/tianyi-lab/PoLar)
- PoLar predictor overhead is about 0.8% of inference [search excerpt], with about 2.1M parameters [secondary]. — [HF paper page](https://huggingface.co/papers/2606.06574); [ArXivIQ](https://arxiviq.substack.com/p/skip-a-layer-or-loop-it-learning)
- Replication: the router's top-1 program collapses to the default pass, while top-k programs give a real gain. "A small number of generic programs are enough to solve most of the questions" [search excerpt]. — [arXiv 2609.31360](https://arxiv.org/html/2609.31360)
- CoLa: MCTS chosen over greedy or beam search for long, non-trivial paths. Used per test sample as an existence analysis [search excerpt]. — [arXiv 2507.07996](https://arxiv.org/abs/2507.07996)
- Router-Tuning: under 30 minutes of training on an RTX A6000, "1000 times faster than DLO" [search excerpt]. — [arXiv 2410.13184v6](https://arxiv.org/html/2410.13184v6)
- Offline fixed-layout search examples:
  - Takase's 3-way discrete comparison [primary]. — [takase/share_layer_params](https://github.com/takase/share_layer_params)
  - Sandwich Transformer sublayer reordering, which reached 0.968 bpc on enwik8 [primary]. — [ofirpress/sandwich_transformer](https://github.com/ofirpress/sandwich_transformer)

### Inferences
- In PoLar/Dr.LLM, each MCTS simulation is a full forward pass, and for generative math it is plausibly a full decode, on a modified path scored against the gold answer. Search cost therefore scales as simulations × examples × decode length. That is affordable offline at about 4K examples, but is not usable online.
- A practical amortisation for a deployed looped student:
  1. Run offline search on a calibration set.
  2. Cluster the winning programs into K templates (the replication suggests a small K suffices).
  3. Ship K precompiled graphs plus a tiny classifier, or a fixed default if the classifier's top-1 proves unreliable, as the replication found.

### Gaps
- No paper reported total GPU-hours for the MCTS data-generation phase (Dr.LLM, PoLar, CoLa).
- No study compares "router over K clustered programs" with "per-layer router" at matched data.

---

## Deployability: what forms of routing survive static graphs on NPUs, TensorRT and phones

### Takeaway
Data-dependent control flow is the problem. TensorRT and CUDA graphs cannot capture runtime branches, loops or data-dependent shapes efficiently, and edge SRAM favours fixed, cache-local schedules (MobileLLM). Deployable forms, from most to least deployable:
1. A fixed layout (any order, as long as it is static).
2. A per-sequence or per-frame choice among a few precompiled programs, selected by a cheap classifier.
3. A per-token depth with batched scheduling (MoR's expert-choice fixed-k, continuous depth batching), mainly on GPU servers.

Per-token arbitrary paths (MoM, CoLa-style) have no reported edge deployment.

### Cited Findings
- **TensorRT and CUDA graphs**:
  - The NVIDIA TensorRT performance guide lists "loops, conditionals, and layers requiring data-dependent shapes" as operators that CUDA graphs do not support [search excerpt]. — [TensorRT optimization guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
  - The TensorRT for RTX docs say graph capture "cannot represent runtime-dependent branches (for example, if-else nodes)". Data-dependent shapes or on-device control flow need a simplified network, CUDA graphs disabled, or fixed shapes [search excerpt]. — [TensorRT-RTX CUDA graphs](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-cuda-graphs.html)
  - TensorRT 10.x release notes: convolution on implicitly data-dependent-shape tensors "may run significantly slower", and engine building "will likely fail for a bool tensor with an implicitly data-dependent shape" [search excerpt]. — [TensorRT 10.14.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.14.1.html)
- **Compilers**: "Dynamic routing introduces fine-grained, data-dependent task dependencies that are unknown at compile time", which defeats compilers that assume static task graphs (Event Tensor, arXiv 2604.13327) [search excerpt]. — [arXiv 2604.13327](https://arxiv.org/pdf/2604.13327)
- **Edge memory hierarchy**: MobileLLM chose AABB (immediate block-wise) because about 20 MB of SRAM holds about one block, and back-to-back reuse avoids DRAM weight traffic [search excerpt]. — [arXiv 2402.14905v2](https://arxiv.org/html/2402.14905v2)
- **Early-exit and batching problems**: tokens that exit early leave missing KV for deeper layers and sit idle in batches. MoR addresses both with parameter sharing and recursion-wise KV caching [primary].
  - Expert-choice routing "avoid[s] variable-length complexities since all samples have the same length", while token-choice needs padding [primary]. — [raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)
- **Continuous depth batching (Schwethelm et al., arXiv 2608.09444, Aug 2026)**: schedules looped-model inference at the level of single loop iterations. On Ouro 1.4B, the refill variant reaches about **94–99%** of the theoretical speed-up bound offline. Refilling gives little benefit for Huginn 3.5B at low load [search excerpt]. — [arXiv 2608.09444](https://arxiv.org/html/2608.09444v1)
- **Serving status of routed methods**:
  - Dr.LLM's evaluation commands use `--batch_size 1` [primary]. — [parameterlab/dr-llm](https://github.com/parameterlab/dr-llm)
  - LayerSkip gives speedups only for generation, through self-speculative decoding [primary]. — [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
  - Router-Tuning claims skipping "without breaking KV-cache coherence" [primary]. — [CASE-Lab-UMD/Router-Tuning](https://github.com/CASE-Lab-UMD/Router-Tuning)
  - TaH2 ships a mini-SGLang-based server (GPU) [primary]. — [thu-nics/TaH](https://github.com/thu-nics/TaH)
  - Huginn has a vLLM plugin [primary]. — [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
- **Per-frame discrete choice**: AR-Net picks a resolution, or a skip, per frame from a discrete set using a lightweight policy network [search excerpt]. — [arXiv 2007.15796](https://arxiv.org/abs/2007.15796)
- **A few generic programs suffice**: this replication finding supports shipping a small menu of fixed paths [search excerpt]. — [arXiv 2609.31360](https://arxiv.org/html/2609.31360)

### Inferences
- For Jetson/TensorRT, the safe pattern is one engine (or one CUDA-graph capture) per program. A host-side selector picks the engine per sequence (token models) or per frame or clip (vision), which keeps each graph static. Weights are shared across engines only if the runtime allows weight sharing or refit, which is not verified here.
- For phones and browsers, a fixed AABB order is cache-friendly (MobileLLM). An ABAB loop over a 2-block core can still be unrolled into a static graph at a fixed depth. Variable depth then becomes a choice among a few unrolled depths.
- Per-token routing in causal decoding is viable on GPU servers with dedicated schedulers (MoR, continuous depth batching, TaH2's engine). No source shows it running on an NPU.

### Gaps
- I found no vendor documentation (Qualcomm QNN/Hexagon, Apple ANE/Core ML, Android NNAPI, WebGPU/ONNX Runtime Web, Jetson DLA) on conditional or loop ops for dynamic layer routing in this pass.
- No paper measured on-device latency (phone NPU or Jetson) for any per-input layer-routing method (Dr.LLM, PoLar, FlexiDepth, MoR).
- TensorRT engine weight-sharing or refit across several precompiled path variants was not verified.
