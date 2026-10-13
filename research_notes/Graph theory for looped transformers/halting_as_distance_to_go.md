# Halting as distance-to-go: learned remaining-compute predictors and puzzle-solving looped/recurrent models

Scope note (for the report writer): this file covers halting framed as cost-to-go / value / "remaining passes", and recurrent or looped models that solve puzzles by running longer on harder instances. Generic ACT/PonderNet/MoR/Ouro/Huginn exit gates are covered elsewhere and are only mentioned where they contrast with this framing.
Environment caveat: arxiv.org, openreview, arcprize.org, proceedings sites, Semantic Scholar, HF and alphaXiv were all blocked by the proxy (HTTP 403 / DNS failure). Only GitHub was reachable. So:
- **[primary-code]**: read directly from the official GitHub code or README (most reliable).
- **[search excerpt]**: wording or numbers from web-search excerpts of the primary paper (the paper was not opened directly).
- **[secondary]**: blogs, summaries, review sites, replication reports.
Dates are arXiv submission months unless stated.

---

## Q1. Halting as value/Q-learning: HRM (Sapient, Jun 2025), TRM (Samsung SAIL Montreal, Oct 2025) and 2025–2026 follow-ups. How halting targets are defined, and what compute is saved on Sudoku/maze/ARC

### Takeaway
HRM's "Q-learning halting" is a 2-output head on one token position. Its targets are a binary "is the current full answer exactly correct" (halt) plus a bootstrapped one-step-ahead target (continue). TRM keeps only the "is it correct now" BCE. In both official codebases the head is used only during training, to recycle batch slots. Evaluation always runs the maximum of 16 supervision steps. The halting head is a 0/1 "already solved?" classifier, not a remaining-steps predictor. 2026 follow-ups say it halts too early or too late, and they replace it with convergence-based (fixed-point) stopping. Running far past 16 steps substantially raises Sudoku accuracy (59.2%→87.5% at 512 steps for attention-TRM).

### Cited Findings
**HRM halting mechanics (official code)**
- HRM's Q-head is `CastedLinear(hidden_size, 2)` producing (q_halt, q_continue) logits from `z_H[:, 0]`, the first position (a puzzle-embedding/CLS-like slot). It is initialised with weight 0 and bias −5, so sigmoid ≈ 0.007 and the model starts out "never halt". [primary-code] — [sapientinc/HRM hrm_act_v1.py](https://github.com/sapientinc/HRM/blob/main/models/hrm/hrm_act_v1.py)
- In training, halting happens when `q_halt_logits > q_continue_logits` or at `halt_max_steps`. Exploration: with probability `halt_exploration_prob` (0.1) a minimum halt step is drawn uniformly from {2…halt_max_steps}, and halting before it is suppressed. [primary-code] — [hrm_act_v1.py](https://github.com/sapientinc/HRM/blob/main/models/hrm/hrm_act_v1.py), [hrm_v1.yaml](https://github.com/sapientinc/HRM/blob/main/config/arch/hrm_v1.yaml)
- Continue target: the code runs one extra forward pass of the inner model on the new carry. It sets `target_q_continue = sigmoid(next_q_halt if last_step else max(next_q_halt, next_q_continue))`. The code comments: "No replay buffer and target networks for computing target Q-value." [primary-code] — [hrm_act_v1.py](https://github.com/sapientinc/HRM/blob/main/models/hrm/hrm_act_v1.py)
- Halt target: `q_halt_loss = BCE(q_halt_logits, seq_is_correct)`, where `seq_is_correct` means every non-ignored output token equals the label (exact whole-grid match). Total loss = `lm_loss + 0.5*(q_halt_loss + q_continue_loss)`. A logged metric `q_halt_accuracy` = fraction where (q_halt ≥ 0) == seq_is_correct. [primary-code] — [HRM losses.py](https://github.com/sapientinc/HRM/blob/main/models/losses.py)
- **Not used at inference**: the code comment reads "During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes". `halted` is only `is_last_step` outside `self.training`. [primary-code] — [hrm_act_v1.py](https://github.com/sapientinc/HRM/blob/main/models/hrm/hrm_act_v1.py)
- The HRM config sets `halt_max_steps: 16`, `H_cycles: 2`, `L_cycles: 2`, 4+4 layers, hidden 512. A secondary summary says "up to 8 steps", which conflicts with the config. [primary-code] — [hrm_v1.yaml](https://github.com/sapientinc/HRM/blob/main/config/arch/hrm_v1.yaml); [secondary search summary](https://arcprize.org/blog/hrm-analysis)
- HRM paper (Fig. 5, Sudoku-Extreme-Full): ACT "maintains a low and stable number of average compute steps even as M_max increases", with accuracy close to fixed-compute models. The search excerpts gave no exact average-step numbers. HRM claims "inference-time scaling by simply increasing M_max" without retraining. The scaling is strong on Sudoku and small on ARC-AGI. [search excerpt] — [HRM arXiv 2506.21734](https://arxiv.org/abs/2506.21734)

**TRM simplification (official code + paper excerpts)**
- TRM config: `no_ACT_continue: True` ("No continue ACT loss, only use the sigmoid of the halt which makes much more sense"). Halting is `q_halt_logits > 0`, i.e. P(current answer correct) > 0.5. Settings: `halt_max_steps: 16`, `halt_exploration_prob: 0.1`, `H_cycles: 3`, `L_cycles: 6`, `L_layers: 2`. The author comments on the continue loss in code: "This fits Q-learning, but seems totally unecessary". [primary-code] — [TRM trm.yaml](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/config/arch/trm.yaml), [trm.py](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py), [TRM losses.py](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/losses.py)
- TRM keeps HRM's eval behaviour of always using max steps (same code comment). The paper states that the full N_sup = 16 supervision steps run at test time "to maximize downstream performance, while ACT is used only during training". Dropping the continue loss removes the second forward pass during training. [primary-code] + [search excerpt] — [trm.py](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py); [TRM arXiv 2510.04871](https://arxiv.org/abs/2510.04871)
- TRM headline results: 7M params, 45% ARC-AGI-1, 8% ARC-AGI-2. README expectations on Sudoku-Extreme (1k examples ×1000 augmentations): ~87% exact accuracy for the MLP variant and ~75% for attention. Paper figures: 87.4% (MLP) and 74.7% (attention). The 1-step-gradient (HRM-style) ablation gives 56.5%. [primary-code README] + [search excerpt/secondary] — [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels); [TRM arXiv](https://arxiv.org/abs/2510.04871)

**ARC Prize and critique analyses (2025)**
- ARC Prize's HRM ablation repo swaps the hierarchical model for a parameter-matched transformer "but keep[s] the remaining interfaces and outer refinement loop". It also varies H/L cycles (1/1, 2/2, 4/4). [primary-code README] — [arcprize/hierarchical-reasoning-model-analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)
- ARC Prize blog (via search summaries):
  - The hierarchy has minimal impact; a same-size transformer lands within ~5 pp.
  - The outer refinement loop drives performance, "especially at training time".
  - Varying the maximum number of outer steps matters a lot.
  - Reproduction: 32% ARC-AGI-1 semi-private vs 41% claimed public; 2% vs 4% on ARC-AGI-2.
  [secondary] — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis); [Medium summary](https://medium.com/@yongshaoruan/hierarchical-reasoning-model-hrm-what-we-learned-from-arc-prize-analysis-45a778d66f9e)
- Ge, Liao & Poggio, "HRM: Perspectives and Misconceptions" (Oct 2025): the ACT halting mechanism did not meaningfully improve inference results. A simpler, deeper recurrent transformer matched HRM with faster training. [secondary: alphaXiv summary via search] — [arXiv 2510.00355](https://arxiv.org/abs/2510.00355)
- TRM on ARC-AGI-1 (Dec 2025):
  - Pass@1 at recursion step 1 is ~94% of final-step accuracy (38.25% vs 40.50%). Going past training depth does not change accuracy, so "the deployed model behaves more like a single-pass system".
  - A 1000-sample voting pipeline adds ~11 pp.
  - Blank or random puzzle IDs give 0% accuracy.
  [search excerpt] — [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)

**2025–2026 follow-ups on halting/stopping**
- CGAR (Nov 2025; v3 Dec 2025, self-reported):
  - Halting: "100% halting accuracy", 11% fewer reasoning steps; average steps 5.85→5.52 in ablation.
  - Training: 1.71× speedup (10.93→6.38 h) at 86.65%→86.02% Sudoku-Extreme accuracy. The progressive depth curriculum alone gives a 41.4% FLOPs reduction.
  [search excerpt/secondary] — [arXiv 2511.08653](https://arxiv.org/abs/2511.08653v3); [sotaverified](https://sotaverified.org/papers/251108653)
- Ren & Liu, "Reasoning or Guessing?" (Jan 2026):
  - HRM can fail on a Sudoku with one unknown cell, because of a violated fixed-point property.
  - There are multiple latent fixed points, some of them false solutions, and the model "settles on whichever fixed point it reaches first".
  - Escaping false fixed points (augmentation plus input/model bootstrapping) raised Sudoku-Extreme from 54.5% to 96.9%.
  [search excerpt] — [arXiv 2601.10679](https://arxiv.org/abs/2601.10679)
- Equilibrium Reasoners / EqR (Huang, Geng, Kolter; 20 May 2026; ICML 2026):
  - Training is capped at 16 iterations, but the attractor dynamics generalise to >1,024 test iterations.
  - "Simple cases converge within 1 to 5 iteration steps", while hard cases benefit from massive test-time scaling.
  - Sudoku-Extreme accuracy rises from 2.6% (feed-forward) to >99%. The ICML abstract instead says 8%→>99% at 8192× compute; the two versions disagree.
  - Ablation: a hierarchical latent with learned ACT reaches 84.8% on Sudoku vs 73.9% for the single-latent variant.
  [search excerpt] — [arXiv 2605.21488](https://arxiv.org/abs/2605.21488)
- EqR code ranks stochastic restarts (`different_init`) by a convergence score: mean norm of latent deltas over the last `convergence_window` steps (eval configs use window 3, top-k 4; Sudoku eval runs up to `halt_max_steps: 64`). It keeps the most-converged trajectories, which is zero-shot, residual-based selection. [primary-code] — [EqR eval_fn.py](https://github.com/YIFANK/EqR/blob/main/evaluators/eval_fn.py), [EqR eval config](https://github.com/YIFANK/EqR/tree/main/config/eval)
- A search excerpt also claimed that "oracle (ground-truth) halting collapses the hierarchical model from 75.4% to 13.6% on Sudoku, whereas learned ACT improves it to 84.8%" in a 2026 training-dynamics paper. A second search confirmed only the 84.8% (EqR ablation), not 75.4/13.6. [search excerpt, partially unverified] — [arXiv 2605.21488](https://arxiv.org/abs/2605.21488)
- Fixed-Point Reasoners / FPRM (Movahedi et al., Jun 2026):
  - Uses fixed-point convergence as the halting signal, with pre-norm plus learnable residual scaling for stable trajectories, and test-time damping against latent oscillations (patience-based "FPOPT").
  - Notes that TRM does not use its ACT head at inference, and that TRM's ACT "either halts too early, resulting in lower performance, or too late, using excessive compute".
  - Claims to beat TRM at lower cost on Sudoku-Extreme.
  [search excerpt] — [arXiv 2606.18206](https://arxiv.org/abs/2606.18206)
- "Looped Actor" (Sep 2026) adapts FPRM to RL and halts when the predicted action distribution stabilises, rather than on latent residuals. [search excerpt] — [arXiv 2609.37432](https://arxiv.org/html/2609.37432v1)
- "When Recursive Models Finish Computing" (Krishna et al., 22 Sep 2026), on 1,000 hard Sudokus:
  - Running TRM to 512 steps instead of 16 raises cumulative exact-solve accuracy from 59.2%→87.5% (attention) and 74.4%→91.9% (MLP).
  - "Latent-state motion falls sharply after the first exact solution", and completed states are locally contractive along the trajectory.
  [search excerpt] — [arXiv 2609.26487](https://arxiv.org/abs/2609.26487)
- Probabilistic TRM (May 2026): stochastic sampling lifts pass@K on Sudoku-Extreme from 87.3% to 99.06%, and on Maze-Hard from 83.8% to 95.63%. [search excerpt] — [arXiv 2605.19943](https://arxiv.org/abs/2605.19943)

### Inferences
- HRM/TRM "halting" is really a *solved-now* classifier, a one-bit distance-to-solved (d=0 vs d>0). HRM's continue head is the only part that looks at the future, through a one-step TD bootstrap. The user's "predict remaining passes" head would generalise TRM's head from binary to ordinal.
- The compute HRM/TRM save with ACT is **training** compute: halted samples free their batch slot for new data, since halted rows are replaced in `reset_carry`/`current_data`. No inference savings are reported, because the official eval disables halting. Any claim that "HRM/TRM save inference compute via Q-halting" is not supported by the official code.
- The 16→512-step Sudoku gains and EqR's 1–5 steps for easy vs ≫16 for hard show a heavy-tailed per-instance distribution of required depth. That tail is where a remaining-passes predictor pays off. On ARC-AGI-1 the distribution is concentrated at 1 pass (94% of final accuracy at step 1), so a depth head there would mostly predict "exit now".
- The fixed-point critiques (Ren & Liu; FPRM) suggest pairing a learned head with a convergence guard. A confident head on a spurious attractor is a known failure.

### Gaps
- Exact HRM Figure 5 average-step numbers and TRM Table 1 rows other than 87.4/74.7/56.5 could not be verified, because arXiv was blocked. "w/ ACT continue loss = 86.1%" is recalled but unverified.
- ARC Prize blog numbers for ACT vs fixed-step training were not retrievable (arcprize.org blocked). Only secondary summaries were available.
- No source reported calibration curves or depth-prediction accuracy for HRM/TRM Q-heads, beyond the logged `q_halt_accuracy` metric and CGAR's "100% halting accuracy".

---

## Q2. Deep Thinking (Schwarzschild et al. NeurIPS 2021) and "Logical extrapolation without overthinking" (Bansal et al. NeurIPS 2022): solving harder mazes/chess/prefix-sums by iterating longer; how they stop; overthinking

### Takeaway
DT-nets are trained with ≤30 iterations on small instances. At test time they run up to thousands of iterations on larger ones (e.g., 9×9→59×59 mazes, 32-bit→512-bit prefix sums). They have no learned halting. Stopping is either a fixed test budget or a zero-shot "max-confidence" rule: run all iterations and pick the iterate whose summed softmax confidence peaks. Recall (re-injecting the input every iteration) and the progressive loss are what prevent "overthinking", i.e. accuracy decaying with extra iterations. That stability is the precondition for any "run until done" policy.

### Cited Findings
- **Repo defaults**: DT-net configs use `max_iters: 30` in training and test-iteration sweeps of 30–40 by default (configurable). [primary-code] — [deep-thinking config/problem/model](https://github.com/aks2203/deep-thinking/tree/main/config/problem/model)
- **Progressive loss weights**: α (progressive-loss weight) is 0.01 for mazes, 0.5 for chess and 1.0 for prefix sums. [primary-code] — [hyp configs](https://github.com/aks2203/deep-thinking/tree/main/config/problem/hyp)
- **Progressive-loss construction**:
  1. Draw n ~ U{0…max_iters−1} and run n iterations, detaching the state (`interim_thought.detach()`).
  2. Draw k ~ U{1…max_iters−n} and run k more iterations with gradients from that state.
  3. Loss = (1−α)·CE(full max_iters unroll) + α·CE(k-step-from-random-start output).
  The model therefore learns to improve from *any* intermediate state without knowing the absolute iteration index. [primary-code] — [training.py](https://github.com/aks2203/deep-thinking/blob/main/deepthinking/utils/training.py)
- **Stopping rule (`test_max_conf`)**:
  - Runs all `max_iters` and computes per-iteration confidence = sum over pixels/positions of max softmax probability. For mazes it is masked to maze pixels.
  - For each sample it takes the output at the running argmax of confidence (`torch.cummax`).
  - This is zero-shot (no trained halting) and costs the full budget; it selects the iterate rather than stopping early.
  [primary-code] — [testing.py](https://github.com/aks2203/deep-thinking/blob/main/deepthinking/utils/testing.py)
- **Bansal 2022 numbers**:
  - Trained on 32-bit strings with m=30 max iterations; tested on 512-bit prefix sums.
  - DT-Recall solves 90.27% of 512-bit samples with n=0 (plain unrolled loss) and 97.12% with randomly sampled n (progressive loss).
  - Models trained on 9×9 mazes are evaluated on 59×59; "mazes this large cannot be solved without recall".
  - Networks "leap from solving small/simple training problems with tens of iterations to solving large and complex problem instances at test time using thousands of iterations".
  [search excerpt] — [arXiv 2202.05826](https://arxiv.org/abs/2202.05826); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10397305)
- **Overthinking**: the authors define "overthinking" as degradation under many iterations. Recall plus progressive training avoids it, because "the recall architecture keeps an explicit copy of the problem instance in memory". [search excerpt] — [arXiv 2202.05826](https://arxiv.org/abs/2202.05826)
- **MLRC 2022 reproduction**: largely confirms the claims; for mazes only DT-Recall-Prog reaches 100%. Average test time is 0.2 h for prefix sums and 2.25 h for mazes. [secondary] — [ReScience [Re] report](https://openreview.net/pdf/07d5d68b5873d779bd1fd8c95b9767cb57fe0bc4.pdf); [MLRC code](https://github.com/mcleish7/MLRC-deep-thinking)
- **Knutson et al. 2024 (mazes)**: DT models that extrapolate over maze *size* fail to generalise when other difficulty knobs (`deadend_start`, `p`) are varied. "Harder" along one axis does not mean "more iterations fixes it". [search excerpt] — [arXiv 2410.03020](https://arxiv.org/abs/2410.03020)
- **"Learning to Stop Overthinking at Test Time" (Feb 2025)**:
  - Argues early stopping "as soon as possible" limits RNN extrapolation.
  - Trains an auxiliary self-supervised task (rotation prediction) and halts the main task at the iteration where self-supervised performance peaks. This is a learned proxy for the accuracy trend over iterations, with no main-task labels needed at test time.
  - Uses a Conv-LiGRU model, evaluated on CIFAR-C.
  [search excerpt] — [arXiv 2502.10954](https://arxiv.org/abs/2502.10954)
- **Path independence (Anil et al., NeurIPS 2022)**:
  - Equilibrium models that converge to the same state regardless of initialisation ("path independence") show strong upward generalisation to harder instances "which need more inference iterations".
  - Interventions promoting path independence help; penalising it hurts.
  - Per-example path independence on OOD inputs "strongly correlates with accuracy", so it works as a per-example diagnostic.
  [search excerpt] — [arXiv 2211.09961](https://arxiv.org/abs/2211.09961)

### Inferences
- The progressive loss (random detached start, random number of further steps) matches what a looped student needs before any exit head makes sense. Each pass must be a valid improvement step from any intermediate state, so outputs are usable at every depth and stay stable past the needed depth.
- The DT "max-confidence" selector is a cheap baseline label/oracle generator. Run P passes on the calibration set, and the argmax-confidence pass versus the first-correct pass gives a direct measurement of how well confidence tracks required depth.
- Path independence (two different initial states converging to the same output) can serve as a zero-shot correctness check alongside a learned head. It costs 2× compute, so it suits calibration or label checking rather than on-device runtime.

### Gaps
- The primary text of Bansal 2022 was not reachable. I could not confirm whether the paper itself (as opposed to the repo) proposes a halting criterion, nor get exact 59×59 maze accuracies or iteration counts at which mazes are solved.
- No source found for a learned "iterations remaining" head on DT-nets.

---

## Q3. Learned "remaining steps"/difficulty predictors for early exit: exit-depth regression, oracle-depth labels, confidence vs learned depth, calibration; exit heads trained on labels from running the model at several depths

### Takeaway
There is a clear line of work that builds **oracle-depth labels by running the model at every depth** and trains a small head to imitate them:
- Depth-Adaptive Transformer (2020): correctness and likelihood oracles, geometric vs multinomial exit distributions.
- "Learning to Stop While Learning to Predict" (2020): oracle model, then an imitated stopping policy.
- AdaInfer (2024/IJCAI 2025): per-layer correct/incorrect labels with an SVM/CRF on logit features.
- Think-at-Hard (TaH, Nov 2025 / ICML 2026): per-token "hard" labels from a reference model's first-pass mismatch, then an MLP decider with BCE on `label > depth`.
- TaH2 (2026): online labels for "does another iteration improve next-token CE".
The **geometric/hazard ("continue at depth d iff label > d") formulation** recurs and works better than a single multinomial "which depth" classifier.

### Cited Findings
**Depth-Adaptive Transformer (Elbayad, Gu, Grave, Auli; ICLR 2020)**
- The exit distribution per token is parametric, q_t(n), fitted with cross-entropy to an oracle distribution q*_t.
- There are two granularities, sequence-specific depth and token-specific depth, and two classifier types:
  - multinomial: predict the exit block directly;
  - "geometric-like": a binary continue/exit decision after every block.
- Oracles are likelihood-based (LL) and correctness-based (C).
[search excerpt] — [arXiv 1910.10073](https://arxiv.org/abs/1910.10073); [author slides](https://elbayadm.github.io/assets/talks/anytime/decore_04102019.pdf)
- Results:
  - Matches a well-tuned 6-block Transformer on IWSLT De-En "while using less than a quarter of the decoder layers".
  - Best token-level model (Tok-C, geometric-like): 34.73 BLEU at an average exit of 1.42 blocks, ≈76% fewer decoding blocks.
  - The correctness oracle beats the likelihood oracle at token level for geometric classifiers; at sequence level the likelihood oracle is better.
  - Geometric-like token classifiers give a better speed-accuracy trade-off than multinomial ones.
  [search excerpt] — [arXiv 1910.10073](https://arxiv.org/abs/1910.10073); [Meta AI page](https://ai.meta.com/research/publications/depth-adaptive-transformer/)

**Learning to Stop While Learning to Predict (Chen, Dai, Li, Gao, Song; ICML 2020)**
- A steerable architecture: a feed-forward deep model plus a *variational stopping policy* that "sequentially determine[s] the optimal number of layers for each input instance".
- Training splits into an **oracle model learning stage** and an **imitation stage**. The paper motivates this by noting that classical iterative algorithms have stopping criteria while unrolled deep models use a fixed depth.
- Tasks: sparse recovery (unrolled ISTA-style), few-shot meta-learning, vision.
[search excerpt] — [arXiv 2006.05082](https://arxiv.org/abs/2006.05082); [Google Research page](https://research.google/pubs/learning-to-stop-while-learning-to-predict/)

**AdaInfer (Fan et al.; arXiv Mar 2024, IJCAI 2025)**
- Uses logit features ("gap" = top1−top2 probability, "top prob"), which rise and stabilise with depth. A classical SVM/CRF predicts the stop layer.
- Labels come from whether each layer's output gives the correct answer, so an L-layer model yields L training pairs per example. The label rule comes from a lecture slide, not the paper text.
- Result: 17.8% average layer pruning and up to 43% on sentiment, with <1% performance drop, on Llama2/OPT.
[search excerpt/secondary] — [arXiv 2403.02181](https://arxiv.org/abs/2403.02181); [IJCAI 2025](https://www.ijcai.org/proceedings/2025/566)

**Generic early-exit oracle definition**
- An "oracle" exit is the first internal classifier that gets the label right. It is the ceiling for how early a model could exit, and it is the standard per-sample label for training exit predictors. [search excerpt] — [Early Exiting with Ensemble Internal Classifiers, arXiv 2105.13792](https://arxiv.org/abs/2105.13792); survey: [arXiv 2501.07670](https://arxiv.org/abs/2501.07670)

**CALM (Schuster et al., NeurIPS 2022) confidence measures**
- Three measures compared: softmax response, "state propagation" (cosine distance between consecutive layers' hidden states, a zero-shot convergence signal) and a trained early-exit classifier.
- Softmax response is "statistically strong and simple"; the other two are cheaper in FLOPs.
[search excerpt] — [Google Research blog](https://research.google/blog/accelerating-text-generation-with-confident-adaptive-language-modeling-calm/); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract.html)

**Think-at-Hard (TaH; Fu et al.; arXiv 2511.08577, Nov 2025; ICML 2026), the closest match to the user's design**
- Labels (v1): a reference model (Qwen3-1.7B in the recipes) labels each token `mismatch=1` iff its greedy prediction ≠ the gold next token. These "hard" tokens get extra latent iterations. [primary-code] — [TaH label.py](https://github.com/thu-nics/TaH/blob/main/script/tah/preparation/label.py); [TaH README](https://github.com/thu-nics/TaH)
- Two-stage training:
  1. Train the backbone/LoRA with the oracle `IterLabelDecider`, which continues iff `iter_count_labels` say so.
  2. Train an `MLPIterDecider` with `IterDeciderLoss`, a per-iteration BCE whose target at depth d is `(iter_count_labels > d)`, with an optional `pos_weight` "to handle imbalance".
  [primary-code] — [loss.py](https://github.com/thu-nics/TaH/blob/main/tah/model/loss.py), [README](https://github.com/thu-nics/TaH)
- Decider input: stacked hidden states from selected layers (default layers 16, 20, 24, 28) concatenated with projected top-k logits, fed into a small residual MLP (hidden dims 256-512-256). Serving threshold: `--tah_iter_threshold 0.5`. [primary-code] — [iter_decider.py](https://github.com/thu-nics/TaH/blob/main/tah/model/iter_decider.py)
- Results:
  - "Latent overthinking": tokens correct after one pass are sometimes revised into errors.
  - An oracle that skips iterations on correct tokens improves accuracy by up to 7.3%.
  - TaH gains 8.1–11.3% over always-iterate baselines while exempting 94% of tokens from the second iteration (arXiv v1). The ICML version reports 3.8–4.4% at 93% skipped.
  [search excerpt] — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62678); [arXiv html v1](https://arxiv.org/html/2511.08577v1); [PMLR v306](https://proceedings.mlr.press/v306/fu26t.html)
- **TaH2** (newer, same repo):
  - "jointly trains the backbone and an iteration decider with lookahead depth supervision, using online labels that indicate whether another iteration improves prediction".
  - "Posterior labels are generated online from next-token cross-entropy improvements" measured in a side pass, with cost-sensitive BCE weighted by "boundary distances" and an `abs_improv_coverage` (default 0.99) threshold. No offline labelling is needed.
  - Reported: 53% better accuracy-compute slope than the non-looped baseline and about +3.4 points peak accuracy on AIME at matched test-time compute (repo claim).
  [primary-code] — [TaH README](https://github.com/thu-nics/TaH), [tah2/model/iter_label.py](https://github.com/thu-nics/TaH/blob/main/tah2/model/iter_label.py)

### Inferences
- Label definitions that recur across these works:
  - (a) **first-correct depth**: the first depth whose output equals gold or teacher (Depth-Adaptive "C" oracle, early-exit oracle, AdaInfer, TaH v1);
  - (b) **likelihood/improvement-based**: the depth maximising likelihood, or "does one more pass still reduce loss" (Depth-Adaptive "LL", TaH2);
  - (c) **one-bit "solved now"** (HRM/TRM).
  For a distilled looped student, (b) measured against the *teacher* (KL to teacher logits or teacher features) avoids label noise from gold labels that the student can never reach.
- The "continue at depth d iff label > d" BCE (TaH, Depth-Adaptive geometric-like) is exactly a discrete-time **survival/hazard model of remaining passes**. Summing the predicted continue probabilities gives an expected remaining-pass count. This is the natural way to realise "predict how many more passes" while keeping per-pass decisions monotone.
- The finding that geometric/hazard heads beat multinomial "pick-the-depth" heads (Depth-Adaptive) argues against a one-shot regression of total depth from pass 1. Re-predicting after each pass, with the information gained, works better.

### Gaps
- No direct accuracy, MAE or calibration numbers for predicted exit depth were found for Depth-Adaptive Transformer, AdaInfer or TaH. Only downstream BLEU/accuracy versus compute was found.
- The exact Depth-Adaptive LL-oracle formula (its depth-penalty regulariser) and the confidence-thresholding baseline settings were not retrieved.
- HashEE, RAEE and BERxiT were not verified in this session; the search did not surface usable excerpts.

---

## Q4. Cost-to-go learning from search/RL that transfers: approximate value iteration (DeepCubeA), learned admissible heuristics, learned stopping in iterative solvers/DEQs

### Takeaway
DeepCubeA learns cost-to-go by deep approximate value iteration on states generated by reverse scrambles from the goal, then uses it in batched weighted A*. Q* search predicts all children's costs in one forward pass. 2026 work trains *admissible* (under-estimating) neural heuristics with asymmetric losses and post-hoc calibration offsets. The transferable pieces are:
- labels generated backwards from "solved" (cheap, exact distance labels);
- Bellman-consistent bootstrapping (HRM's continue target is a one-step instance of this);
- asymmetric loss plus a calibration offset to make the predicted distance err on the safe side.

### Cited Findings
- **DeepCubeA (Agostinelli et al., Nat. Mach. Intell. 2019)**:
  - Trains a DNN cost-to-go function with approximate value iteration on states "obtained by starting from the goal state and randomly taking moves in reverse", because random forward play rarely reaches the goal.
  - The learned cost-to-go is the heuristic for weighted A* with f = λg + h and batched expansion of the lowest-cost nodes for GPU-parallel evaluation.
  [search excerpt] — [DeepCubeA paper PDF](https://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf); [code](https://github.com/forestagostinelli/DeepCubeA)
- **DeepCubeA results**: shortest path found 60.3% of the time for the Rubik's cube and >90% for the 15-puzzle, 24-puzzle and Lights Out. A later table from the same group lists the 15-puzzle at 99.4% optimal (avg 52.03 moves). "Solved 100% of test configurations" is only confirmed by a secondary article. [search excerpt]/[secondary] — [DeepCubeA paper](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf); [HPCwire](https://www.hpcwire.com/?p=66874)
- **DAVI loss**: squared error between the network's cost estimate for a state and a one-step backed-up target over successors. The target is 0 for terminal states, and the target network is periodically refreshed. Convergence guarantees exist only for tabular value iteration, not for the neural approximation. [search excerpt of Agostinelli lecture slides] — [DRL and HeurSearch slides](https://cse.sc.edu/~foresta/assets/files/DRL_and_HeurSearch.pdf)
- **Q* search ("A* Search Without Expansions", Agostinelli et al.)**: a DQN outputs costs for all actions/children in one forward pass, generating one node per iteration. It is up to 129× faster and generates up to 1,288× fewer nodes than A*. [search excerpt] — [arXiv 2102.04518](https://arxiv.org/abs/2102.04518)
- **Learned empirically admissible heuristics (arXiv 2606.04860, Jun 2026)**:
  - Plain MSE training "regularly yields overestimations".
  - The method combines an under-estimating "Admissible Bellman Operator", an asymmetric loss and a post-hoc calibration safety offset.
  - Zero observed admissibility violations on evaluation sets for Lights Out, the 8-puzzle and the 2×2 cube, with fewer node expansions than analytical baselines. Admissibility is empirical, not proven.
  [search excerpt] — [arXiv 2606.04860](https://arxiv.org/abs/2606.04860)
- **Related 2026 heuristic work**: "Learning Admissible Heuristics for A*: Theory and Practice" (ICLR 2026) uses a "Cross-Entropy Admissibility" loss with near-perfect admissibility on Rubik's cube. "Trajectory-Aware Heuristic Learning for Combinatorial Search" (ICML 2026) addresses uncertain cost-to-go labels. Abstract-level only. [search excerpt] — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009069); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63536)
- **Self-supervised cube solving**: "Self-Supervision is All You Need for Solving Rubik's Cube" (Takano) trains on random scrambles; it surfaced in search but was not examined, as other researchers cover cube theory. — [arXiv 2106.03157](https://arxiv.org/abs/2106.03157)
- **Learned stopping in iterative algorithms**: Chen et al. 2020 jointly learn an unrolled model and a variational stopping policy (oracle stage, then imitation stage), applied to sparse recovery among other tasks (see Q3). [search excerpt] — [arXiv 2006.05082](https://arxiv.org/abs/2006.05082)
- **Convergence-based stopping in implicit/fixed-point models**:
  - Path-independent equilibrium models generalise upward and use more iterations on harder inputs (Anil et al. 2022).
  - FPRM (2026) uses fixed-point convergence as the halting signal, with damping against oscillation.
  - EqR (2026) learns task-conditioned attractors; easy instances converge in 1–5 steps.
  [search excerpt] — [arXiv 2211.09961](https://arxiv.org/abs/2211.09961); [arXiv 2606.18206](https://arxiv.org/abs/2606.18206); [arXiv 2605.21488](https://arxiv.org/abs/2605.21488)
- **Early-exit KV systems**: EE-LLM-style inference reproduces CALM and handles KV for skipped layers by filling the KV cache of the remaining layers before an iteration ends, reporting up to 1.25× speedup over vLLM running all layers. Relevant to Q6. [search excerpt] — [arXiv 2407.20272](https://arxiv.org/abs/2407.20272)

### Inferences
- **Transfer recipe, DeepCubeA style**: remaining passes r satisfies a Bellman identity, r(h_t) = 0 if the output at pass t is already "good enough", else 1 + r(h_{t+1}). HRM's continue-head target is this identity in probability form. With only ~3–6 passes (A→B→A→B→A→B), Monte-Carlo labels from running every depth are cheap and exact, so TD bootstrapping (and its instability; HRM uses no target network) is unnecessary. TD only becomes attractive if training depth ≫ deploy depth.
- **Admissibility flips sign for exits**: in A*, *under*-estimating cost-to-go is safe, because optimality is preserved. For an exit head, *over*-estimating remaining passes is the safe error, because it wastes compute but keeps quality. The asymmetric-loss plus calibration-offset machinery transfers directly with the asymmetry reversed. Calibrate the offset on held-out data to hit a target "premature-exit rate", e.g. <1% of frames or tokens.
- **Q* analogue**: a head that outputs remaining-pass estimates for several candidate continuations in one forward pass (e.g., "next block = A vs B") is a Q-function over block choices. This is where the remaining-steps head meets per-input block-path search, which is another researcher's slice.

### Gaps
- No primary DeepCubeA numbers on accuracy of the cost-to-go estimate itself (e.g., MAE vs true distance) were retrieved.
- No source found that applies DAVI-style bootstrapped remaining-steps targets to neural *layer/pass* counts (as opposed to puzzle moves), apart from HRM's one-step continue target.

---

## Q5. Looped/recurrent transformers solving combinatorial puzzles and graph problems: how needed iterations scale with instance difficulty; length/depth generalisation

### Takeaway
Constructive results (Giannou 2023; de Luca & Fountoulakis 2024) tie loop count to algorithm steps: one instruction or one algorithm step per pass, so required passes grow with instance size, and halting comes from the program or algorithm itself. Learned looped models (Fan et al. 2024/25) need the ground-truth step count at training time and use either the true count or a confidence rule at inference. Puzzle solvers (RRN, DT-nets, TRM, EqR) show harder instances keep improving with more iterations well past the training budget, with per-instance needs ranging from 1–5 steps to hundreds.

### Cited Findings
- **Looped Transformers as Programmable Computers (Giannou et al., ICML 2023)**:
  - A hand-set looped transformer emulates a SUBLEQ one-instruction computer: program counter, conditional branches, Turing complete.
  - An author's blog says one forward pass executes one instruction and you loop until the program halts, adding that "the authors did not solve the halting problem".
  - A "13-layer" figure appears only in a secondary summary.
  [search excerpt]/[secondary] — [PMLR v202](https://proceedings.mlr.press/v202/giannou23a.html); [Papailiopoulos blog](https://dimitrisp.substack.com/p/can-you-train-a-computer); [DeepAI](https://api.deepai.org/publication/looped-transformers-as-programmable-computers)
- **Simulation of Graph Algorithms with Looped Transformers (Back de Luca & Fountoulakis, ICML 2024)**:
  - A looped transformer with extra graph-interacting attention heads simulates Dijkstra, BFS, DFS and Kosaraju's SCC.
  - Parameter count does not grow with graph size, but "a limit to simulation in our solution due to finite precision" exists.
  - It is Turing complete with constant width.
  [search excerpt] — [PMLR v235](https://proceedings.mlr.press/v235/back-de-luca24a.html); [arXiv 2402.01107](https://arxiv.org/abs/2402.01107)
- **Looped Transformers for Length Generalization (Fan, Du, Ramchandran, Lee; ICLR 2025)**:
  - Defines n-RASP-L tasks whose required loop count depends on length: copy = n shift iterations, parity = n shift+XOR iterations, addition = n+1 iterations.
  - Training uses only input-output pairs plus the number of steps.
  - At inference the model either stops by "predefined stopping criteria" or at the ground-truth step count. A secondary summary calls the criterion "Maximum Confidence".
  - Extra iterations tend to leave copy/addition outputs stable (secondary).
  - A 2026 follow-up notes that Fan et al. "assume an oracle number of training iterations based on sample complexity".
  [search excerpt]/[secondary] — [arXiv 2409.15647](https://arxiv.org/abs/2409.15647); [Lacuna summary](https://lacuna.tiptreesystems.com/work/looped-transformers-for-length-generalization/wrk_a672571cdfd705f2c0a6ca12953e5a88); [Loop, Think & Generalize, arXiv 2604.07822 (Apr 2026)](https://arxiv.org/html/2604.07822v1)
- **Recurrent Relational Networks (Palm et al., NeurIPS 2018)**: Sudoku as a graph with message passing trained for 32 steps. The abstract figure is 96.6% on the hardest 17-givens puzzles; an earlier draft said 94.1%. The paper describes Sudoku as "requiring upwards of 64 steps of relational reasoning". A benchmark page says harder puzzles benefit from steps beyond 32. [search excerpt]/[secondary] — [arXiv 1711.08028](https://arxiv.org/abs/1711.08028)
- **DT-nets**: tens of iterations in training, thousands at test, for larger mazes and prefix sums (see Q2). [search excerpt] — [arXiv 2202.05826](https://arxiv.org/abs/2202.05826)
- **TRM on Sudoku**: 16 vs 512 steps gives 59.2→87.5% (attention) and 74.4→91.9% (MLP). Latent motion collapses after the first exact solution. [search excerpt] — [arXiv 2609.26487](https://arxiv.org/abs/2609.26487)
- **EqR**: trained ≤16 iterations, generalises past 1,024; easy cases converge in 1–5 steps; Sudoku-Extreme and Maze-Unique. A secondary summary reports >93% on uniquely solvable mazes vs 55% feed-forward (unverified). [search excerpt] — [arXiv 2605.21488](https://arxiv.org/abs/2605.21488); [EqR repo](https://github.com/YIFANK/EqR)
- **TRM maze config**: Maze-Hard 30×30 uses `H_cycles=3`, `L_cycles=4`, while Sudoku uses `L_cycles=6`, so per-step depth is tuned per task rather than learned. [primary-code] — [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

### Inferences
- Across these results, required passes scale with **algorithmic depth** (path length, number of propagation rounds, number of constraint-propagation sweeps), not with input size per se. Knutson et al. show other difficulty axes may not be fixable by iterating at all. For per-frame vision models the analogue is "how many refinement passes until features match the teacher", which likely correlates with scene clutter or motion rather than resolution. That is a hypothesis, not sourced.
- Wherever a model generalises to more iterations at test time (DT-nets with recall, TRM, EqR), training used randomised or progressive depth, or deep supervision from intermediate states. A looped student whose exit head will choose variable depth should be trained with randomised pass counts, or have every pass supervised (stage A matching the teacher at each pass), so outputs at early exits are valid.

### Gaps
- Exact iteration-vs-graph-size bounds in de Luca & Fountoulakis, and the exact inference stopping criterion in Fan et al., could not be read from primary text.
- No source found on looped transformers solving Rubik's cube or permutation puzzles with learned pass counts; DeepCubeA-family solvers use search, not looping.

---

## Q6. Practical recipe and pitfalls for a stage-B exit head (label noise, class imbalance, monotonicity, KV-cache interaction, head cost)

### Takeaway
The evidence supports this recipe:
- Build labels by running the frozen looped student at every pass count against the teacher.
- Define "solved" as *stays good from this pass on*.
- Train a per-pass hazard head (BCE on `remaining > 0`, or cumulative `remaining > k`) on a dedicated token or pooled features, with positive-class weighting.
- Calibrate the threshold on held-out data with a deliberately conservative bias.
- Keep a zero-shot convergence or confidence guard.
Main pitfalls:
- spurious fixed points and confident wrong answers;
- "overthinking" flips, where more passes make the answer worse;
- heavy label imbalance (93–94% of tokens need no extra pass in TaH; ~94% of ARC-AGI-1 accuracy comes at step 1);
- batching and KV-cache bookkeeping when tokens exit at different depths.

### Cited Findings
- **Head design in the puzzle models**: HRM/TRM use a single linear 2-logit head on one reserved position (`z_H[:,0]`), with bias init −5 so it does not halt early in training and a 10% exploration floor on minimum steps. [primary-code] — [hrm_act_v1.py](https://github.com/sapientinc/HRM/blob/main/models/hrm/hrm_act_v1.py); [trm.py](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py)
- **Head design in TaH**: an MLP over hidden states from 4 layers plus top-k logits, with `pos_weight` for imbalance. TaH2 uses cost-sensitive BCE weighted by distance to the decision boundary. [primary-code] — [TaH iter_decider.py](https://github.com/thu-nics/TaH/blob/main/tah/model/iter_decider.py); [loss.py](https://github.com/thu-nics/TaH/blob/main/tah/model/loss.py); [tah2 iter_label.py](https://github.com/thu-nics/TaH/blob/main/tah2/model/iter_label.py)
- **Imbalance**: TaH skips the second iteration for 93–94% of tokens. On ARC-AGI-1, TRM reaches ~94% of final accuracy at step 1. [search excerpt] — [ICML 2026 TaH](https://icml.cc/virtual/2026/poster/62678); [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)
- **Non-monotonicity and overthinking**:
  - Extra passes can make correct tokens wrong; oracle skipping recovers up to 7.3% accuracy (TaH).
  - DT-nets without recall degrade with more iterations (Bansal 2022).
  - HRM can fail on a one-blank Sudoku by converging to a false fixed point (Ren & Liu).
  [search excerpt] — [TaH](https://icml.cc/virtual/2026/poster/62678); [arXiv 2202.05826](https://arxiv.org/abs/2202.05826); [arXiv 2601.10679](https://arxiv.org/abs/2601.10679)
- **Mis-timed halting**: TRM's learned ACT halts either too early (accuracy loss) or too late (wasted compute), per FPRM, which switches to fixed-point convergence with damping. [search excerpt] — [arXiv 2606.18206](https://arxiv.org/abs/2606.18206)
- **Training-time coupling**: a learned ACT head was part of what made a hierarchical latent model work (84.8% Sudoku in the EqR ablation). A claim that ground-truth/oracle halting collapses training (75.4→13.6%) was seen only in one search excerpt and is unverified. [search excerpt] — [arXiv 2605.21488](https://arxiv.org/abs/2605.21488)
- **Batching**: official HRM/TRM disable halting at eval "to guarantee the same halting steps inside a batch". EE-LLM processes a batch until all sequences pass the confidence threshold. [primary-code] — [trm.py](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py); [search excerpt] — [arXiv 2407.20272](https://arxiv.org/abs/2407.20272)
- **KV cache**:
  - CALM names "attending back to missing hidden representations" as a core challenge of token-level exits. When a token exits early, later tokens attend to the hidden state of its last computed layer, which is propagated rather than recomputed.
  - EE-LLM instead fills the KV cache for the remaining layers before the iteration ends, reporting up to 1.25× over vLLM.
  [search excerpt] — [CALM blog](https://research.google/blog/accelerating-text-generation-with-confident-adaptive-language-modeling-calm/); [arXiv 2407.20272](https://arxiv.org/abs/2407.20272)
- **Zero-shot guards available**:
  - softmax confidence (CALM; DT `max_conf`);
  - hidden-state change (CALM "state propagation"; EqR latent-delta norm; FPRM fixed-point residual);
  - auxiliary self-supervised proxy ("Learning to Stop Overthinking");
  - path-independence agreement (Anil et al.).
  [primary-code]/[search excerpt] — [deep-thinking testing.py](https://github.com/aks2203/deep-thinking/blob/main/deepthinking/utils/testing.py); [EqR eval_fn.py](https://github.com/YIFANK/EqR/blob/main/evaluators/eval_fn.py); [arXiv 2502.10954](https://arxiv.org/abs/2502.10954); [arXiv 2211.09961](https://arxiv.org/abs/2211.09961)
- **Hazard vs multinomial**: geometric/hazard heads beat multinomial depth classifiers at token level (Depth-Adaptive Transformer). [search excerpt] — [arXiv 1910.10073](https://arxiv.org/abs/1910.10073)

### Inferences
These are recommendations synthesised from the findings above; none comes from a single source.

1. **Labels (stage B, frozen student)**: for each calibration input, run passes p = 1…P (P = 6 for A→B→A→B→A→B) and score each pass against the teacher:
   - tokens: argmax agreement with the teacher's token, plus KL(teacher‖student_p) < τ;
   - vision/video: per-frame task metric or feature cosine to the teacher above ε.
   Define d* = the smallest p such that all p' ≥ p are good. This "stays-solved" definition absorbs overthinking flips. Inputs never solved by pass P get d* = the pass after which loss stops improving (TaH2-style "does another pass help"), not P. Otherwise the head learns "always run to max" on hopeless inputs, which wastes compute for no gain.
2. **Targets**: at pass t, train BCE on y_t = 1[d* > t] (TaH/Depth-Adaptive hazard form). Optionally add cumulative outputs 1[d* > t+k] for k = 1…K, so a single read after pass 1 also yields a remaining-pass estimate for scheduling or power planning. Cumulative logits enforce monotone, ordinal predictions (P(r>k) non-increasing in k) if parameterised as ordered thresholds.
3. **Imbalance and asymmetric cost**:
   - use `pos_weight` or cost-sensitive weights (TaH, TaH2);
   - pick the threshold on held-out data for a target premature-exit rate;
   - add a conservative offset, mirroring the admissible-heuristic calibration offset but biased toward running extra passes.
4. **Head placement and cost**: a linear or tiny-MLP head on a reserved token (HRM/TRM) or on pooled features plus top-k logits (TaH). It costs far less than one block. For vision, read it from a pooled CLS/register token. For video, feed the previous frame's d* or exit pass as an extra input feature, since temporal coherence is a cheap prior (hypothesis, unsourced). Initialise the bias toward "continue" (HRM/TRM −5) so stage-B training does not collapse to early exit.
5. **Guards**: exit only if the head says stop **and** a cheap convergence check passes (‖h_t − h_{t−1}‖ small, or argmax unchanged between passes). Cap passes at P. This hedges against the confident-but-wrong attractor failure (Ren & Liu) and the "halts too early" failure (FPRM).
6. **KV caching for token models**: when a token exits at pass p < P, deeper passes of later tokens need its K/V at those passes. Options are CALM-style reuse of the last computed state (copy the pass-p state into the slots for passes p+1…P) or computing only the K/V projections for the remaining passes (EE-LLM-style). With weight-shared blocks, the copied state is at least in-distribution for the shared projections. Sequence-level or per-frame exits avoid this issue entirely and suit batch-1 on-device inference (phone/browser/Jetson).
7. **Training interaction**: if stage B also fine-tunes the body, follow deep-supervision practice and supervise every pass so early-pass outputs remain valid exits. Do not let the head's decisions gate which passes receive gradient, given the HRM/EqR evidence that halting couples to training dynamics.

### Gaps
- No published study found that reports *calibration* (reliability diagrams/ECE) of a learned remaining-pass head in a looped transformer, or MAE of predicted vs oracle depth.
- No source found on exit heads for per-frame video looped models specifically, including use of temporal priors from previous frames.
- Head FLOP or latency overhead numbers on edge hardware (Jetson, mobile GPU, WebGPU) for any of these heads were not found.
- KV handling for weight-shared looped decoders with per-token exits beyond CALM/EE-LLM, e.g. Mixture-of-Recursions KV sharing, is covered by another researcher and was not re-examined here.
