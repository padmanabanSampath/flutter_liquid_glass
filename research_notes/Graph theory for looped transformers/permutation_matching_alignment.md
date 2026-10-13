# Permutation symmetry, graph matching and multi-way alignment for tying transformer layers into looped blocks

Scope: the matching and synchronization machinery for aligning 2 or N weight sets, framed for building a looped student (for example A-B-A-B-A-B) whose shared blocks start from an aligned average of teacher layers. Collected 2026-10-13.

Source-type tags used on every number:
- **[primary]**: the authors' own repo README or code, or a library's own docstring, read directly.
- **[search excerpt]**: the paper's own text (abstract, tables or body) as returned by the web-search tool. arxiv.org, openreview.net, neurips.cc, proceedings sites, ar5iv, alphaxiv, semanticscholar, researchgate and every other paper mirror were blocked by the proxy (DNS or CONNECT 403), so no paper PDF could be opened directly. Only GitHub and raw.githubusercontent.com worked.
- **[secondary]**: a third-party review or aggregator (Liner, Moonlight, Pith, Emergent Mind, aggregator sites).
- **[measured here]**: a benchmark I ran in this environment (4-core CPU, 15 GB RAM, Python, numpy 2.5.4, scipy 1.18.1). The scripts are in the session scratchpad. These are synthetic timings and simulations, not neural-network results.

---

## Q1. What symmetry groups does a transformer block have, and which can be used when tying layers inside one model?

### Takeaway
For generic weights, a multi-head attention block's exact symmetry ("gauge") group is ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h. That is: permute the h heads, apply any invertible map to each head's Q/K pair (W_Q→W_Q·A, W_K→W_K·A^{-T}), and apply any invertible map to each head's V/O pair. RoPE shrinks the Q/K part to the commutant of the rotary rotations, roughly one complex scalar (a 2×2 rotation-scaling) per frequency pair. MLPs add S_{d_ff} on hidden units. The residual stream has one global orthogonal symmetry under RMSNorm, the one SliceGPT and QuaRot use. Inside one model the residual basis is already shared by every layer, so layer-to-layer alignment for tying uses only the per-layer internal symmetries: FFN hidden units, head order, per-head QK and VO maps, and folded norm gains.

### Cited Findings
**Attention gauge group (maximal characterization)**
- Wang & Wang, "Complete Characterization of Gauge Symmetries in Transformer Architectures" (NeurReps 2025 proceedings track, listed at NeurIPS 2025). Their stated result is that the gauge group of canonical Transformers (GPT-2, BERT, LLaMA, Qwen family) equals exactly `G_max = ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h`, under generic-rank assumptions — [primary: repo README](https://github.com/kellywang2030/transformer-gauge-symmetry); [search excerpt: NeurIPS listing](https://neurips.cc/virtual/2025/loc/san-diego/136893)
- With RoPE the group becomes `((C_RoPE)^h × (GL(d_v))^h) ⋊ S_h`, where C_RoPE is the commutant of the position-dependent rotations. For standard RoPE this is "typically a product of copies of GL(1, ℂ)". Maximality is proved partly by showing that query-key and value-output transformations factorize independently — [search excerpt](https://neurips.cc/virtual/2025/loc/san-diego/136893)
- The characterization implies more than 1.1 million redundant dimensions in a 110M-parameter Transformer-Base model — [search excerpt](https://neurips.cc/virtual/2025/loc/san-diego/136893)
- Empirical validation on GPT-2 124M/355M/774M/1.5B (all d_k = d_v = 64):
  - Valid gauge transforms preserve outputs to about 1e-6 relative error (10–21 ε_mach in FP32). Invalid transforms break outputs by O(1)–O(10³).
  - Generation exactness under canonicalization is 100%.
  - The genericity assumptions hold on pretrained weights: W_Q, W_K, W_V have full column rank (σ_min ≥ 0.12), the per-head bilinear forms W_Q^(i)W_K^(i)T are linearly independent (rank = h at every layer), and W_O has full row rank (σ_min ≥ 0.12).
  - [primary: README](https://github.com/kellywang2030/transformer-gauge-symmetry)
- Independent confirmation of the RoPE restriction: query-key invariance requires the head transformation to commute with the RoPE rotation — [search excerpt, arXiv 2607.17146](https://arxiv.org/abs/2607.17146). Another paper also identifies the invertible Q/K reparameterizations that preserve rotary attention with the rotation commutant, which it calls a classical result — [search excerpt, arXiv 2608.10288](https://arxiv.org/abs/2608.10288)
- A 2026 preprint argues that GQA reduces the number of unidentified parameters and that RoPE+GQA compounds the reduction, with a factor-of-8 example for Llama-2-70B. This is a single unreviewed claim — [search excerpt, arXiv 2609.01231](https://arxiv.org/abs/2609.01231)
- Transformer-NFN (Tran et al., arXiv 2410.04209, Oct 2024) derives "the maximal symmetry group of the weights in a multi-head attention module" and a necessary-and-sufficient condition for two parameter sets to define the same function. It builds an NFN equivariant to that group and releases a dataset of more than 125,000 Transformer checkpoints. One secondary summary describes the group as head permutations plus invertible linear maps in the query/key/value spaces — [search excerpt](https://arxiv.org/abs/2410.04209); [secondary](https://www.emergentmind.com/topics/equivariant-attention-weights)

**Beyond permutations: rotations and invertible maps used for alignment**
- Zhang, Zheng, Chen & Li, "Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion" (ICML 2025 Spotlight, PMLR 267:77090–77106):
  - It introduces rotation symmetry, which rotates parameter matrices in self-attention and is continuous rather than discrete, "significantly expanding the equivalence set".
  - It gives a "theoretically optimal parameter matching algorithm" as a plug-in for fusion, which "substantially improves model fusion" on pretrained NLP and vision transformers. No numbers were retrievable.
  - Code builds on dataless-model-merging, with DeBERTa NER and emotion experiments.
  - [search excerpt](https://icml.cc/virtual/2025/poster/43634); [primary: repo](https://github.com/zhengzaiyi/RotationSymmetry)
- Theus et al., "Generalized Linear Mode Connectivity for Transformers" (arXiv 2506.22712, Jun 2025):
  - It uses four symmetry classes: permutations, semi-permutations, orthogonal transformations and general invertible maps. A secondary source gives the nesting permutation ⊂ semi-permutation ⊂ orthogonal ⊂ invertible.
  - It finds low- or zero-barrier linear paths between independently trained ViTs and GPT-2 models, and it can align Transformers of different widths.
  - The repo offers orthogonal-alignment, strict-permutation and different-embedding-dimension merging modes.
  - [search excerpt](https://arxiv.org/abs/2506.22712); [primary: repo](https://github.com/alexandertheus/Generalized-LMC-for-Transformers)

**Residual stream: one global symmetry, shared by every layer**
- SliceGPT's "computational invariance": an orthogonal transform applied to one block's output can be undone in the next, because RMSNorm commutes with orthogonal Q. Due to residual connections the transform must be carried through the embedding and every later layer up to the LM head — [search excerpt, arXiv 2401.15024](https://arxiv.org/abs/2401.15024)
- LayerNorm models are first converted to RMSNorm form: the LayerNorm scales are absorbed into neighbouring weights and the mean subtraction is pushed into the output weights. SliceGPT computes a different PCA rotation per block, which adds extra transforms on the residual path — [secondary: Moonlight review](https://themoonlight.io/review/slicegpt-compress-large-language-models-by-deleting-rows-and-columns)
- QuaRot applies randomized Hadamard rotations using the same invariance. With 4-bit weights, activations and KV cache, Llama-2-70B loses at most 0.29 WikiText-2 perplexity — [search excerpt, arXiv 2404.00456](https://arxiv.org/abs/2404.00456). A 2026 paper describes QuaRot as one global orthogonal matrix in the residual stream, applied after RMSNorm gains are folded into the consuming weights — [search excerpt, arXiv 2608.25188](https://arxiv.org/abs/2608.25188)
- In permutation-only merging of independently trained BERTs (MultiBERTs), there is "only one permutation pair for the entire residual stream". With only the residual components aligned, the identity permutation gave the lowest loss barrier (4.95) and learned alignments gave 7.34–9.38 — [search excerpt, Verma & Elbayad, arXiv 2403.00986](https://arxiv.org/abs/2403.00986); authors per [primary: repo](https://github.com/nverma1/merging-text-transformers)
- Head-structured units need group-aware matching. MuDSC adds "group matching" so that matching works for ViT heads and GroupNorm — [primary: README](https://github.com/zju-vipa/training_free_model_merging)

**Cross-layer evidence that head order matters inside one model**
- LiSA (cross-layer attention sharing for pretrained LLMs, TACL vol. 14): "reusing weight matrices without first rearranging the heads performs poorly", and shallow layers are sensitive to small deviations in attention weights. It aligns heads between adjacent layers with tiny FFNs — [search excerpt, arXiv 2408.01890](https://arxiv.org/abs/2408.01890)

### Inferences
- **For tying layers L_i, L_j of one teacher, the free alignment variables per layer are:**
  1. A permutation of FFN hidden units. For SwiGLU, the gate and up rows and the down columns are permuted jointly. For ReLU, positive per-unit rescaling is also exact, but GELU/SiLU allow permutations only. That last point is standard algebra, not from a fetched source.
  2. A permutation of heads. Under GQA, query heads can only be permuted within their KV group, together with permuting whole groups. This is my inference; the 2609.01231 preprint only says GQA shrinks the redundancy.
  3. Per-head V/O maps in GL(d_v). Orthogonal Procrustes on V/O is allowed.
  4. Per-head Q/K maps. Without RoPE these are GL(d_k). With RoPE they are restricted to block-diagonal 2×2 rotation-scalings, one per frequency pair (GL(1,ℂ)^{d_k/2}), so a full Q/K Procrustes rotation is not function-preserving in RoPE models.
- **The residual-stream rotation is not a per-layer degree of freedom.** It is shared by all layers of a model, as in SliceGPT and QuaRot. When tying layers inside one model it is already aligned "for free", which is the within-model analogue of Verma & Elbayad's finding that identity beat learned residual permutations. A per-layer residual rotation would need extra matrices on the residual path, as SliceGPT adds, and a looped block that runs several times can absorb only one shared basis.
- **Fold RMSNorm gains into the following linear weights before averaging.** Pre-norm gains then stop being separate parameters that get averaged, which addresses the norm-averaging failure seen in Relaxed Recursive Transformers (see Q6).

### Gaps
- Could not open the gauge-symmetry paper itself, so it is unknown whether it treats GQA, SwiGLU or QK-norm explicitly. The README lists only GPT-2 validation, which has no RoPE and no GQA.
- No numbers retrieved for the ICML 2025 rotation-symmetry fusion gains or for Generalized-LMC barrier values (arXiv blocked).

---

## Q2. Two-model / two-layer alignment: algorithms, costs, and measured quality of averaging after vs before alignment

### Takeaway
Pairwise alignment is a solved, cheap problem. Each permutation is one linear assignment problem (LAP) over a similarity matrix, from weights (weight matching) or from activation correlations (activation matching). Coordinate descent over layers handles the coupling between layers, and a dense 11008×11008 LAP takes about 2–26 s on a 4-core CPU depending on how structured the similarity matrix is. Alignment turns near-random or collapsed averages into usable models. Soft or richer alignments (OT, CCA, orthogonal, learned) and post-merge renormalization (REPAIR) close much of the remaining gap.

### Cited Findings
**Git Re-Basin (Ainsworth, Hayase, Srinivasa; ICLR 2023)**
- Three methods:
  - Activation matching.
  - Weight matching, which is data-free and "orders of magnitude faster", running "in mere seconds for all models tested".
  - A straight-through estimator (STE), which is best but "very computationally expensive".
  - Weight matching and activation matching perform similarly.
  - [search excerpt](https://personalrobotics.cs.washington.edu/publications/ainsworth2023gitrebasin.pdf); [search excerpt](https://arxiv.org/abs/2209.04836)
- It was the first demonstration of zero-barrier linear mode connectivity between independently trained ResNets on CIFAR-10 and CIFAR-100. Barriers after weight matching shrink as width grows — [primary: README abstract](https://github.com/samuela/git-re-basin); [search excerpt](https://personalrobotics.cs.washington.edu/publications/ainsworth2023gitrebasin.pdf)
- The weight-matching algorithm in the official code works as follows:
  - For each permutation group, visited in random order each iteration, it builds A = Σ W_a W_b^T over all weights touching that permutation, with the other permutations held fixed.
  - It solves `linear_sum_assignment(A, maximize=True)`.
  - It repeats until no group improves, with `max_iter=100`.
  - [primary: code](https://github.com/samuela/git-re-basin/blob/main/src/weight_matching.py)
- Joint permutation matching across coupled layers is NP-hard. The proof reduces from the quadratic assignment problem — [search excerpt via Crisostomi thesis search](https://arxiv.org/abs/2605.01580). FedMA likewise notes that permutations of consecutive layers are coupled, giving "an NP-hard combinatorial optimization problem", and solves them recursively layer by layer — [search excerpt, arXiv 2002.06440](https://arxiv.org/abs/2002.06440)
- A replication project reported long ResNet run times and a residual barrier on MNIST MLPs — [secondary](https://github.com/stanislavfort/dissect-git-re-basin)

**REPAIR (Jordan et al., ICLR 2023): renormalize after aligning**
- Interpolating two aligned, independently trained ResNet50s on ImageNet gives less than 1% midpoint accuracy. REPAIR on conv outputs raises this to 53.2%, and full REPAIR (also on residual block outputs) to 56.5%.
- The barrier falls from 76% to 20%, a 74% relative reduction. For ResNet18 on CIFAR-10 the reduction is 90%.
- [search excerpt](https://arxiv.org/abs/2211.08403)

**OTFusion (Singh & Jaggi, NeurIPS 2020) and transformer OT fusion (Imfeld et al., ICLR 2024)**
- OTFusion uses layer-wise optimal transport (soft) alignment of neurons, then averages. With 4, 6 and 8 VGG11 models on CIFAR-100, one-shot OT averaging beat the best fine-tuned individual model by about 1.4%, 1.7% and 2%. Vanilla averaging "degrades to near-random performance and fails to recover as the number of models increases" — [search excerpt from author feedback](https://proceedings.neurips.cc/paper_files/paper/2020/file/fb2697869f56484404c8ceee2985b01d-AuthorFeedback.pdf); [search excerpt](https://arxiv.org/abs/1910.05653)
- A later Wasserstein-barycenter reformulation derives the alignment cost from a barycenter objective and solves it by iterative minimization — [search excerpt, arXiv 2210.06671](https://arxiv.org/abs/2210.06671)
- Imfeld et al. extend OT fusion to multi-head attention, LayerNorm and residual connections:
  - Evaluated on ViT (CIFAR-10/100, Tiny ImageNet, ImageNet-1K) and BERT.
  - It "consistently outperforms vanilla fusion" and, after "a surprisingly short finetuning", beats the individual parents.
  - Soft alignment is key; hard alignment underperforms.
  - It supports heterogeneous (different-width) fusion for compression.
  - [primary: README](https://github.com/graldij/transformer-fusion); [search excerpt](https://arxiv.org/abs/2310.05719)

**ZipIt! (Stoica et al.; ICLR 2024 per most listings, one summary says 2023)**
- It "zips" features both across and within each model, so redundant features of the same model can merge. "Partial zipping" leaves the last layers unmerged.
- It reports a 20–60% improvement over prior merging. One setting reaches 83.8% joint accuracy with the last stage unzipped.
- [search excerpt](https://arxiv.org/abs/2305.03053); [secondary](https://liner.com/review/zipit-merging-models-from-different-tasks-without-training)

**MuDSC (Xu et al., CVPR 2024)**
- It matches on a linear combination of weight-space and activation-space unit similarity, because the two can disagree. Group matching handles ViT and GroupNorm — [primary: README](https://github.com/zju-vipa/training_free_model_merging)
- It reportedly beats alignment-based and zip-based baselines on average across 12 Taskonomy tasks — [secondary](https://www.themoonlight.io/review/training-free-pretrained-model-merging)

**CCA Merge (Horoi et al., ICML 2024)**
- It replaces permutations with CCA linear maps. In one table excerpt (80/20 data split), CCA Merge reaches 61.41% against 11.40% for direct averaging, 65.66% for the base-model average and 77.84% for an ensemble. The other rows were garbled — [search excerpt](https://arxiv.org/abs/2407.05385); [venue](https://proceedings.mlr.press/v235/horoi24a.html)

**Learned and amortized alignment**
- Deep-Align (Navon et al., ICML 2024) on CIFAR-10 CNN barriers:
  - Deep-Align 0.23, activation matching 0.23, Sinkhorn 0.31, weight matching 0.66. Deep-Align used to initialize Sinkhorn reaches 0.08.
  - Reported runtimes are 0.20 for Deep-Align, 6.38 for activation matching and 37.74 for Sinkhorn (units unconfirmed).
  - Sinkhorn is better than Deep-Align on STL10 CNN and VGG16.
  - [secondary](https://liner.com/review/equivariant-deep-weight-space-alignment); [primary: repo](https://github.com/AvivNavon/deep-align)
- Dual Learned Matching (Li & Shen, MBZUAI, arXiv 2606.23607, Jun 2026):
  - Both models learn functionality-preserving transforms toward a shared path.
  - It reports "near-zero loss barriers on WikiText" for medium LMs (one version says 160M), above 69% ImageNet top-1 along the ViT-L path, and small barriers for billion-parameter LLMs.
  - Code is "coming soon".
  - [search excerpt](https://arxiv.org/abs/2606.23607); [primary: repo](https://github.com/VILA-Lab/Dual-Learned-Matching)

**Within-model FFN alignment (Verma, Murray & Duh, TMLR 2026)**
- One FF sublayer in a group is the anchor. The other sublayers' neurons are permuted to it using a cross-correlation matrix of activations and the Jonker–Volgenant LAP solver, then averaged and tied — [search excerpt](https://arxiv.org/abs/2501.06126)
- Correlations are computed from `--max-toks 10000` tokens — [primary: README](https://github.com/nverma1/merging-ffs-compression)

**LAP cost**
- scipy's `linear_sum_assignment` is "a modified Jonker-Volgenant algorithm with no initialization" (Crouse 2016) — [primary: scipy 1.18.1 docstring](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
- **[measured here]** dense LAP, maximize, single call, 4-core CPU:

  | n | random uniform matrix | correlation matrix (matched-unit corr 0.6) | computing that correlation matrix from 4096 tokens |
  |---|---|---|---|
  | 768 | 0.04 s | 0.01 s | 0.02 s |
  | 3072 | 1.1 s | 0.15 s | 0.57 s |
  | 4096 | 2.8 s | 0.39 s | 1.3 s |
  | 8192 | 14.3 s | 0.91 s | 5.0 s |
  | 11008 | 26.1 s | 1.7 s | 9.0 s |

  The hidden permutation was recovered exactly (accuracy 1.000) at every size in the correlated case. Scaling from 4096 to 11008 on the random matrix is about 9.3× for a 2.7× larger n, roughly n^2.3 in this range, far below the O(n³) worst case.

### Inferences
- **Cost is not the bottleneck for tying layers.** For a Llama-7B-class FFN (d_ff = 11008), one pairwise alignment costs seconds for the LAP plus seconds for the correlation matmul. Tying 12 layers to an anchor is about 11 LAPs, under 5 minutes on a laptop CPU, and far less on GPU for the correlation part. Weight-matching coordinate descent multiplies this by (number of permutation groups × iterations).
- **Activation-based (correlation) matching is the natural choice for within-model tying.** All layers see the same input distribution and Verma et al. use it. Weight matching is data-free but within one model the "inputs" to different layers differ, so W_a W_b^T similarity is less meaningful for different depths. This is my inference; I found no head-to-head within-model comparison.
- **Soft or richer alignment matters most for transformers.** Imfeld et al. found soft beats hard. Rotation and orthogonal symmetry expand the equivalence set. For an averaged init that will be distilled anyway, hard permutations plus REPAIR-style renormalization are probably sufficient. If one-shot quality matters, per-head orthogonal V/O and RoPE-restricted Q/K alignment are the next lever.

### Gaps
- Exact Git Re-Basin wall-clock numbers and width-vs-barrier tables, Imfeld fusion accuracies, and ZipIt/MuDSC per-model tables were not retrievable (arXiv/OpenReview blocked).
- My LAP timings are for synthetic matrices. Real FFN correlation matrices are lower-contrast, and LAP time tends to rise on less structured, near-random instances (26 s random vs 1.7 s structured at n = 11008).

---

## Q3. Multi-way (N-layer) alignment: synchronization, cycle consistency, universes, and why chaining fails

### Takeaway
Tying k ≥ 3 layers into one block needs k mutually consistent permutations, P_1..P_k into a shared "universe" basis, so that P_i P_j^{-1} is the map from layer j to layer i for every pair. There are three practical routes:
1. **Star alignment:** align every layer to one anchor (Verma et al.). This needs k−1 LAPs, is cycle-consistent by construction, but is biased to the anchor.
2. **Universe or barycenter alignment:** iteratively align every layer to their running average (OTFusion barycenter, generalized-Procrustes style, FedMA/PFNM global model, C2M3's Frank-Wolfe universe).
3. **Permutation synchronization:** spectral or projected power iteration over all pairwise matches (Pachauri et al. 2013).

Pairwise chaining (L1→L2→L3…) compounds per-hop errors, and permutation methods are measurably cycle-inconsistent: more than 50% of neurons disagree between direct and via-third-model matches.

### Cited Findings
**Synchronization theory**
- Pachauri, Kondor & Singh, "Solving the multi-way matching problem by permutation synchronization" (NIPS 2013, pp. 1860–1868):
  - It matches all sets jointly "in one shot" instead of pairwise in series, by relaxing to an eigenvector decomposition. It is "computationally efficient and much more stable to noise than previous methods".
  - It assumes full permutations, meaning every feature appears in every set.
  - [search excerpt](https://neurips.cc/virtual/2013/poster/3941); [search excerpt](https://www.semanticscholar.org/paper/Solving-the-multi-way-matching-problem-by-Pachauri-Kondor/ab63735bb09ab96916830a8fb563049bc10c420d)
- Multi-way matching for n ≥ 3 is hard: even unweighted 3-dimensional matching is NP-hard. Tang & Jebara (AISTATS 2017) propose coordinate updates initialized from a maximum spanning tree of pairwise alignments — [search excerpt](https://proceedings.mlr.press/v54/tang17a.html)
- Partial-permutation and projected-power-iteration variants exist, including "Recovery Theory for Projected Power Iterations in Permutation Synchronization" (arXiv 2609.09502, Sep 2026) and "Fast, Accurate and Memory-Efficient Partial Permutation Synchronization" (arXiv 2203.16505) — [search excerpt](https://arxiv.org/abs/2609.09502); [search excerpt](https://arxiv.org/abs/2203.16505)

**C2M3: cycle-consistent multi-model merging (Crisostomi et al., NeurIPS 2024)**
- It is data-free weight matching that optimizes permutations of all layers jointly, not layer-locally, using Frank-Wolfe. Each Frank-Wolfe step solves an LAP, then a line search.
- For n ≥ 2 models, every model is mapped into a universe space. The pairwise map is factorized as P_{A←B} = P^A (P^B)^T, so every cycle composes to the identity by construction.
- Pairwise methods like Git Re-Basin "do not guarantee cycle consistency". Fixing one model as the universe makes the merge depend on an arbitrary reference.
- Coupled with activation renormalization (REPAIR) it gives the best results. It "outperforms recent layer-local approaches".
- [search excerpt](https://arxiv.org/abs/2405.17897); [search excerpt: NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/3268f1e2474ef9d1af7f034401197a7f-Paper-Conference.pdf); [primary: repo](https://github.com/crisostomi/cycle-consistent-model-merging)
- The only numeric fragment retrieved is ResNet20 CIFAR-100 at 16× width, test accuracy 0.49 for one configuration. Context is unclear — [search excerpt](https://arxiv.org/abs/2405.17897)
- Crisostomi's PhD thesis ("Model Merging: Foundations and Algorithms", Sapienza, arXiv 2605.01580, May 2026) restates C2M3 as aligning networks "into a shared, reference-free parameter space". It lists QAP-style joint optimization of all permutations, coupling adjacent layers, as a future direction beyond the layer-by-layer Frank-Wolfe relaxation — [search excerpt](https://arxiv.org/abs/2605.01580)

**Measured cycle inconsistency and multi-model failure of pairwise methods**
- CCA Merge: for permutation-based methods, "over 50% of the neurons from B don't get matched to the same neuron from C when aligned directly or indirectly through A". CCA Merge produces fewer such mismatches and "works significantly better than past methods" when more than 2 models are merged — [search excerpt](https://arxiv.org/abs/2407.05385)
- Sharma et al. (ECML PKDD 2024) separate weak linear connectivity mod P (a per-pair permutation exists; this is all the evidence supports) from strong (one permutation per network connects it to all others). They give first evidence that strong connectivity may hold, because three-way interpolation barriers fall with width — [search excerpt](https://arxiv.org/abs/2404.06498)
- Ito et al., "Linear Mode Connectivity between Multiple Models modulo Permutation Symmetries" (ICML 2025): two-model permutation search "can fail to transfer multiple models into the same convex low-loss basin". They propose a multi-model STE permutation search — [search excerpt](https://proceedings.mlr.press/v267/ito25a.html)

**Universe or global-model approaches from federated learning**
- PFNM (ICML 2019) matches every client's neurons to a Bayesian-nonparametric global model, using the Hungarian algorithm on a cost matrix. Global width adapts to heterogeneity. It works mainly for fully connected nets — [secondary](https://liner.com/review/bayesian-nonparametric-federated-learning-neural-networks)
- FedMA (Wang et al., ICLR 2020) matches and averages layer by layer into a global model, reusing each layer's permutation for the next, and works on CNNs and LSTMs — [search excerpt](https://arxiv.org/abs/2002.06440)

**Star alignment used for within-model FFN tying**
- Verma et al. designate one FF sublayer per group as the anchor and compute the permutation for each (anchor, other) pair — [search excerpt](https://arxiv.org/abs/2501.06126). The code exposes `--ref-type {first,middle,last}` as an anchor ablation (paper Table 4) — [primary: README](https://github.com/nverma1/merging-ffs-compression)
- Groups of 4 of 12 (ViT, OPUS-MT) and 12 of 36 (GPT-2) sublayers were merged into one — [secondary](https://pith.science/paper/2501.06126)

**[measured here] Synthetic multi-way simulation**
- Setup: k = 6 "layers", n = 1024 units each, T = 1024 tokens. Each layer is a hidden permutation of a shared latent unit set plus independent noise, giving true matched-unit correlation ρ. Methods compared:
  - chain (0→1→2…, composed)
  - star (anchor = layer 0)
  - align-to-mean (5 iterations of re-matching every layer to the running mean of aligned activations, starting from star)
  - spectral permutation synchronization (top-n eigenvectors of the 6144×6144 block matrix of pairwise LAP permutations, rounded with an LAP)
- Results: fraction of layer-0 units mapped to the correct unit in layers 1..5, averaged. All methods ran on the same data (seed fixed).

  | matched-unit corr ρ | single-pair LAP acc | chain (0→1→…→5) | star (anchor 0) | align-to-mean, leave-one-out | spectral sync |
  |---|---|---|---|---|---|
  | 0.09 | 0.348 | 0.109 (hop 5: 0.005) | 0.361 | **0.643** | 0.362 |
  | 0.11 | 0.648 | 0.341 (hop 5: 0.136) | 0.655 | **0.995** | 0.873 |
  | 0.14 | 0.981 | 0.917 (hop 5: 0.872) | 0.968 | **1.000** | **1.000** |

  - Runtimes: chain and star 0.1–0.3 s; align-to-mean 1.2–2.9 s; spectral sync 26–30 s on an idle CPU (78–80 s on a first run that shared the CPU).
  - Pitfall: re-matching each layer to a mean that includes that same layer left the solution stuck exactly at the star result (identical accuracies). The layer's own contribution (1/k of the mean) correlates far more with its current assignment than the true signal does. Matching to the mean of the other layers fixed this.
  - Caveat: my spectral sync used only the binary pairwise LAP permutations and discarded similarity magnitudes. Align-to-mean used raw activations, so the comparison favours align-to-mean. Toy Gaussian data, not network activations.

### Inferences
- **Chaining is the wrong default for tying 3–12 layers.** If each hop is right with probability p, the composed map 1→k is right only about p^{k−1} of the time. In my simulation p = 0.648 gave 0.136 after 5 hops, close to 0.648^5 ≈ 0.11. Chaining also never revisits early mistakes.
- **For tying k layers to one block:**
  1. Use star alignment with a central anchor (middle of the group).
  2. Run 1–5 rounds of leave-one-out align-to-mean (a universe or barycenter). In the simulation this turned a 65%-accurate pairwise regime into 99.5% joint accuracy, because k−1 noisy views average out the noise.
  3. C2M3 provides the cycle-consistent, joint-over-layers Frank-Wolfe version.
  4. The gain is largest exactly where per-pair similarity is weak, which is the case for distant layers inside one model.
- **What scales to 3–12 layers:**
  - Star and align-to-mean cost about k LAPs per round and scale trivially, about 30 s per round at d_ff = 11008 for 12 layers on CPU.
  - C2M3 Frank-Wolfe costs (k models × permutation groups) LAPs per Frank-Wolfe iteration and was demonstrated on small vision nets.
  - Spectral synchronization needs eigenvectors of a (k·n)×(k·n) matrix. For k = 12 and n = 11008 that is 132k×132k, impossible densely (about 140 GB in fp64). It would need iterative or projected-power methods, and in my run it cost 26–30 s (idle CPU) even at k·n = 6144, about 10–20× the cost of align-to-mean for equal or worse accuracy.
- **Use the right "universe" dimension.** For FFNs the universe can be the d_ff hidden space. For heads it is S_h on head order, then per-head maps.

### Gaps
- No published experiment applies C2M3, permutation synchronization or barycenter alignment to layers inside one transformer. All published within-model tying (Verma et al.) uses star alignment.
- No numbers retrieved comparing anchor = first/middle/last (Verma Table 4) or C2M3 vs pairwise accuracy tables.

---

## Q4. Graph formulations for choosing WHICH layers (or heads) to tie

### Takeaway
Layer selection is a graph problem on a layer-similarity graph:
- **Pairs:** max-weight matching, solved exactly by Edmonds' Blossom (GeoPair, Sep 2026).
- **Contiguous windows:** sliding-window or segment search (Verma et al., LaCo).
- **Head level:** greedy best-match pairing by cosine similarity (DirectShare), or learned adjacent-layer head alignment (LiSA).

Evidence consistently says adjacent or middle layers are the most similar and safest to tie. First and last layers and shallow attention are fragile. Choosing the groups well matters as much as aligning within them.

### Cited Findings
**GeoPair (Mohammad, Ali & Lefkimmiatis; MWS AI / ITMO; arXiv 2609.25963, 22 Sep 2026)**
- Method:
  - Cross-layer grouping is cast as weighted maximum matching on a complete graph of candidate weight matrices, "solved optimally via Edmonds' Blossom algorithm using a shape-agnostic column-space alignment metric" (scale-invariant distance), replacing fixed adjacency heuristics.
  - Paired layers share a dictionary but keep their own whitening transforms. The dictionary update is a generalized Sylvester equation with closed-form solution.
  - Structured sparsity comes from Hard Thresholding Pursuit with conjugate gradient.
  - [search excerpt](https://arxiv.org/abs/2609.25963)
- Results:
  - The full pipeline "recovers over 90% of the uncompressed baseline accuracy while maintaining competitive perplexity" — [search excerpt](https://arxiv.org/abs/2609.25963)
  - At compression ratio 0.2, average accuracy is 54.09 on Llama-3.2-1B and 68.17 on Llama-3-8B, against Basis Sharing at 33.11 and 53.47 — [secondary aggregator, not checked against the paper's tables](https://arxiv.org/html/2609.25963)
  - Basis Sharing at 1B has WikiText perplexity 928.07 at CR 0.2, against 11.60 uncompressed — [search excerpt](https://arxiv.org/html/2609.25963)

**Window and threshold heuristics**
- Verma et al. (Algorithm 1) slide a window over adjacent FF sublayers, merge each candidate window, and keep the best on validation data. OLMo3 windows are selected by WikiText-103 perplexity — [primary: README](https://github.com/nverma1/merging-ffs-compression). The authors observe that some groups of FF sublayers have high activation similarity, which they suggest explains why they merge well — [search excerpt](https://arxiv.org/abs/2501.06126)
- LaCo (Findings of EMNLP 2024) collapses rear layers into a prior layer and accepts a merge only if a representation-similarity score clears a threshold T. Iterative search beats fixed rules. It keeps over 80% average task performance at 25–30% pruning (Llama2, Baichuan2) — [search excerpt](https://arxiv.org/abs/2402.11187)
- MKA (Liu et al., EMNLP 2024) maps layer activations to manifolds via a diffusion kernel and pairs layers by an (NPIB/IB) information-bottleneck similarity. On Llama3-8B it reaches 43.75% compression with a 2.82% MMLU drop — [search excerpt](https://arxiv.org/abs/2406.16330); [EMNLP](https://preview.aclanthology.org/setup/2024.emnlp-main.987)
- FuseGPT fuses less-important blocks into neighbouring blocks with learnable low-rank fusion and light local distillation. It claims up to 27% lower perplexity than LaCo/MKA-style baselines and 52.1% total compression with 4-bit GPTQ — [search excerpt](https://arxiv.org/abs/2411.14507)
- Training-free heterogeneous merging (Xu et al., arXiv 2501.00061) handles depth mismatch by grouping adjacent layers with similar representations into segments, because one-to-one layer matching fails across depths. "Elastic neuron zipping" handles width — [search excerpt](https://arxiv.org/abs/2501.00061). A GNN cross-architecture paper uses CKA layer alignment with dynamic programming — [search excerpt](https://arxiv.org/abs/2602.19332)

**Head-level graphs**
- DirectShare (Cao, Yang & Zhao, Findings of EMNLP 2024) traverses every head pair, scores it by cosine similarity, takes each head's best match, then keeps the top-N pairs by the sharing ratio α. This is greedy matching that can pair heads across layers. PostShare first post-trains with a weight-similarity regularizer, then shares — [search excerpt](https://arxiv.org/abs/2402.11819)
- LiSA shares attention across about 53–84% of layers with 6× compression of Q/K. Maximum throughput gains are 19.5%, 32.3% and 40.1% for LLaMA3-8B, LLaMA2-7B and LLaMA2-13B — [search excerpt](https://arxiv.org/abs/2408.01890)

**Looped-model initialization selection (Relaxed Recursive Transformers, Bae et al., ICLR 2025)**
- For non-relaxed recursive models, "Stepwise" (take intermediate layers at fixed intervals) gave the lowest loss and the best few-shot accuracy across Gemma, TinyLlama and Pythia — [secondary](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora)
- The "Average" baseline performed poorly, especially on Gemma, which the authors attribute to possible instability from averaging normalization-layer weights. Gemma did better when both the first and last layers were preserved — [search excerpt](https://arxiv.org/abs/2410.20672)
- For relaxed models (layer-wise LoRA, SVD-initialized, rank 512 in Figure 4), Average init worked best. The default number of looping blocks is B = 2 — [search excerpt](https://arxiv.org/abs/2410.20672)

### Inferences
- **Concrete graph recipe for choosing which teacher layers feed each shared block:**
  1. Build a complete layer graph with edge weight = aligned similarity. That is the similarity after the per-pair LAP, for example the mean matched-unit activation correlation or GeoPair's column-space metric. Unaligned CKA is invariant to permutation, so it also works as a cheap proxy.
  2. For 2 blocks (A, B) sharing L teacher layers, the A-B-A-B pattern is a constrained 2-coloring or partition: even-index layers to A, odd to B. Contiguous groups are a 1-D segmentation solvable by dynamic programming. Free assignment is a k-way partition (spectral clustering or max-k-cut).
  3. Blossom matching gives the optimal pairing for 2-layer groups only. Groups of 3 or more need partitioning or clustering heuristics.
- **The A-B-A-B schedule constrains the graph.** It fixes which teacher depths map to each block (interleaved), so the useful graph question becomes "is the interleaved partition much worse than the best contiguous partition?" That can be checked by computing the partition's total within-group aligned similarity.
- **Keep first and last layers untied, or give them their own block.** RRT's Gemma result, LiSA's shallow-layer sensitivity and the general adjacent-layer similarity findings all point this way.

### Gaps
- GeoPair's own tables, edge-weight definition details, and whether its matching ever pairs non-adjacent layers in practice were not retrievable.
- No paper found that uses spectral clustering or k-way partitioning over a layer-similarity graph specifically to pick tied groups in an LLM. Only matching (GeoPair), greedy matching (DirectShare), windows (Verma) and thresholds (LaCo) were found.
- Exact RRT Stepwise vs Average few-shot numbers were not retrievable.

---

## Q5. Quadratic assignment / joint attention alignment, and Procrustes across layers

### Takeaway
Attention cannot be aligned matrix by matrix. The function depends on the composed per-head operators M_QK = W_Q W_K^T (logits) and M_VO = W_V W_O (output), which are invariant to the per-head GL maps. So attention alignment is a head-level assignment over composed operators, followed by per-head refactorization. SLOA (NeurIPS 2026) does this sequentially, under the partially merged model's activations, to stop error accumulation. Exact joint permutation matching across coupled layers is a QAP, and is NP-hard. Practice uses layer-by-layer LAPs, Frank-Wolfe, or sequential local alignment. Orthogonal (Procrustes-type) alignment is valid on the residual stream (one global rotation) and on V/O per head, but is restricted under RoPE on Q/K.

### Cited Findings
- SLOA, "Training-Free Transformer Merging via Sequential Local Operator Alignment" (Maurya, Lin, Jegelka, Stich, Mulayoff; arXiv 2610.06415, Oct 2026; NeurIPS 2026):
  - Merging Q, K, V, O independently is fragile because "attention logits are determined by the query-key operator, while the remaining computation depends on the value-output operator".
  - Merging earlier components shifts downstream activations.
  - The method uses calibration data to estimate each component's local behaviour, aligns operators sequentially under the intermediate activations of the partially merged model, then factorizes merged operators back into valid transformer parameters.
  - "This sequential step reduces error accumulation across layers."
  - No numbers were retrievable.
  - [search excerpt](https://arxiv.org/abs/2610.06415); [search excerpt](https://neurips.cc/virtual/2026/163034)
- The gauge-group result that Q-K and V-O transformations factorize independently, plus the verified linear independence of the per-head bilinear forms W_Q^(i)W_K^(i)T (rank = h), means these composed per-head operators are well-defined, identifiable matching features — [primary](https://github.com/kellywang2030/transformer-gauge-symmetry); [search excerpt](https://neurips.cc/virtual/2025/loc/san-diego/136893)
- Permutation-based model merging is NP-hard by reduction from QAP. Crisostomi's 2026 thesis lists a QAP formulation "jointly optimizing all permutations while coupling adjacent layers" as future work, noting current methods use Frank-Wolfe relaxations layer by layer — [search excerpt](https://arxiv.org/abs/2605.01580)
- Verma & Elbayad handle multi-head attention and residual inputs/outputs with dedicated interventions so that permutations stay in the same functional equivalence class — [search excerpt](https://arxiv.org/abs/2403.00986). Their repo includes separate attention-merging-algorithm and residual-merging-algorithm experiment scripts — [primary](https://github.com/nverma1/merging-text-transformers)
- Imfeld et al. treat the attention "transportation map flow graph" explicitly and find soft (OT) alignment necessary — [primary: README](https://github.com/graldij/transformer-fusion)
- Orthogonal alignment within one model:
  - SliceGPT computes per-block PCA rotations of the residual stream [search excerpt](https://arxiv.org/abs/2401.15024); QuaRot uses one global randomized-Hadamard rotation [search excerpt](https://arxiv.org/abs/2404.00456).
  - GeoPair keeps per-layer whitening with a shared dictionary, solved by a generalized Sylvester equation. It argues that a single global whitening (as in the predecessor approach) distorts layer-specific geometry — [search excerpt](https://arxiv.org/abs/2609.25963)
- Rotation symmetry in attention (ICML 2025) extends matching from permutations to continuous rotations of attention parameters with a "theoretically optimal" matching step — [search excerpt](https://icml.cc/virtual/2025/poster/43634)

### Inferences
- **Recommended attention alignment for tying layer j onto anchor i:**
  1. Build an h×h head-similarity matrix. Options are ⟨M_QK^(a), M_QK^(b)⟩_F + ⟨M_VO^(a), M_VO^(b)⟩_F (data-free), or attention-pattern and head-output correlations on calibration tokens. Solve the head permutation with one LAP, with h = 32 so it is trivial.
  2. For each matched head pair, align V/O with orthogonal Procrustes in d_v.
  3. Align Q/K only within the RoPE commutant (per-frequency 2×2 rotation-scaling), or skip it.
  4. Average the composed operators, then refactorize (SVD to rank d_k / d_v), as SLOA does.
  - Averaging raw W_Q, W_K separately after only a head permutation leaves the per-head GL gauge unaligned. Averaging the composed operators avoids that.
- **Sequential, activation-conditioned alignment is especially apt for a looped student.** In SLOA's terms, iteration t of block A sees activations produced by the already-tied earlier iterations, so aligning teacher layers in depth order under the student's own partial activations directly targets the "downstream components receive different activations" failure. This is inference; SLOA merges separate models, not layers of one model.
- **Full QAP is unnecessary at the sizes involved.** Head-level problems are h ≤ 64. FFN problems decouple per layer once the residual basis is fixed, which it is within one model. So within-model tying is mostly a set of independent LAPs, not a coupled QAP.

### Gaps
- No SLOA numbers, and no statement of whether SLOA uses head permutations, OT or least-squares operator fitting (full text blocked).
- No published use of Procrustes alignment between different layers of one transformer for weight tying was found. SliceGPT and QuaRot rotate the shared residual basis for compression or quantization, not for tying.

---

## Q6. Measured payoff: does alignment before averaging improve a tied/looped model's starting point and reduce recovery or distillation budget?

### Takeaway
Direct evidence inside one model comes from Verma, Murray & Duh (TMLR 2026). Permutation-aligned, averaged and tied FFNs consistently beat unaligned ("vanilla") tying, the gap widens as more sublayers are tied, and the method matches or beats layer dropping. With modest recovery fine-tuning it keeps about 99% of ViT accuracy at 21% fewer parameters. Across models, alignment is the difference between near-random and usable averages (REPAIR, OTFusion, CCA). In Relaxed Recursive Transformers, naive (unaligned) averaging was the weakest looped-model init. No published study directly measures aligned-average init vs Stepwise init for a looped student under a fixed distillation budget; that is the key open experiment.

### Cited Findings
**Within one model (layer tying)**
- Verma, Murray & Duh, "Merging Feed-Forward Sublayers for Compressed Transformers" (TMLR 2026; arXiv v1 Jan 2025, v2 Mar 2025; per search, published in TMLR in July 2026):
  - Merging more than a third of FF sublayers keeps performance "comparable to their original counterparts". Over 21% of a ViT's total parameters can be removed "while maintaining 99% of its original performance" — [search excerpt](https://arxiv.org/abs/2501.06126)
  - Per a review: 4 of 12 ViT FF sublayers merged plus fine-tune gives more than 21% params removed at about 99% ImageNet accuracy. 12 of 36 GPT-2 sublayers raises perplexity by about 1 point. 4 of 12 OPUS-MT sublayers costs about 2 BLEU. The permutation-aligned merge beats the no-permutation version and matches or beats layer dropping. The review also flags uneven baselines on 2 of 3 tasks — [secondary](https://pith.science/paper/2501.06126)
  - "Permutation-based compression is consistently better compared to no-permute vanilla baselines", and the gap grows as more sublayers are removed — [search excerpt](https://arxiv.org/html/2501.06126)
  - FF sublayers are about two-thirds of non-embedding parameters — [search excerpt](https://arxiv.org/abs/2501.06126)
  - Recovery budgets: ViT recovery fine-tune 50,000 updates; OLMo3-7B QLoRA 3,000 updates (rank 8, α 16, all-linear) on SamSum, NarrativeQA and HotPotQA. Also covers GPT-2-large, OLMo-1B and OPUS-MT, LLM.int8() compatibility, and throughput (Table 7) — [primary: README](https://github.com/nverma1/merging-ffs-compression)
- Relaxed Recursive Transformers (ICLR 2025):
  - Averaging-based looped init "performed poorly, particularly on Gemma", possibly from averaging normalization weights. Stepwise is best without relaxation; Average is best with layer-wise LoRA relaxation.
  - A recursive Gemma 1B uptrained from Gemma 2B gains up to 13.5 absolute few-shot points (22% error reduction) over a non-recursive 1B trained from scratch.
  - With knowledge distillation on 60B tokens it is "on par with" full-size Gemma trained on 3T tokens. With 15B tokens of uptraining it beats full-size TinyLlama and Pythia.
  - [search excerpt](https://arxiv.org/abs/2410.20672); [secondary](https://liner.com/review/relaxed-recursive-transformers-effective-parameter-sharing-with-layerwise-lora)
- LiSA: sharing attention across layers without head rearrangement "performs poorly" — [search excerpt](https://arxiv.org/abs/2408.01890)
- GeoPair: optimal (Blossom) grouping plus geometry-preserving sharing beats Basis Sharing by large margins at CR 0.2 (54.09 vs 33.11 on Llama-3.2-1B; 68.17 vs 53.47 on Llama-3-8B) — [secondary, unverified against the paper](https://arxiv.org/html/2609.25963)

**Across models (the averaging step itself)**
- REPAIR raises aligned ResNet50 midpoints from below 1% to 56.5% ImageNet accuracy (barrier 76% to 20%). On ResNet18 CIFAR-10 the barrier falls 90% — [search excerpt](https://arxiv.org/abs/2211.08403)
- OTFusion: aligned one-shot averages of 4, 6 and 8 VGG11s beat the best fine-tuned individual model by about 1.4%, 1.7% and 2%, while vanilla averages approach random — [search excerpt](https://proceedings.neurips.cc/paper_files/paper/2020/file/fb2697869f56484404c8ceee2985b01d-AuthorFeedback.pdf)
- Transformer OT fusion beats vanilla fusion and, after short fine-tuning, beats the parents — [primary](https://github.com/graldij/transformer-fusion)
- CCA Merge reaches 61.41% against 11.40% for direct averaging in one split — [search excerpt](https://arxiv.org/abs/2407.05385)
- Deep-Align barriers on CIFAR-10 CNNs: weight matching 0.66, Deep-Align or activation matching 0.23, Deep-Align plus Sinkhorn 0.08 — [secondary](https://liner.com/review/equivariant-deep-weight-space-alignment)
- Dual Learned Matching: near-zero WikiText barriers for medium LMs; ViT-L above 69% top-1 along the path — [search excerpt](https://arxiv.org/abs/2606.23607)
- Git Re-Basin: zero-barrier LMC for wide ResNets on CIFAR-10/100 — [primary](https://github.com/samuela/git-re-basin)

### Inferences
- **Expected effect for a looped-student stage-A init.** Aligning the teacher layers assigned to each shared block (FFN hidden permutation, head permutation, per-head V/O Procrustes, RoPE-restricted Q/K, RMSNorm gains folded) before averaging should give a strictly better starting loss than raw averaging. Verma et al. and RRT's poor raw-average result both point this way. It plausibly beats or matches Stepwise (copying one layer), because it keeps information from every tied layer.
- **The budget saving is not quantified anywhere.** The closest data points are Verma's recovery budgets (50k updates for ViT; 3k QLoRA updates for OLMo3-7B) and RRT's 15B/60B-token uptraining with KD.
- **Minimal ablation worth running:**
  1. Four arms: Stepwise, raw Average, aligned Average (star to the middle layer), and aligned Average with align-to-mean refinement plus REPAIR-style renormalization.
  2. For each, record the starting KD loss, then the KD tokens needed to reach a fixed loss.
  3. Report per-block mean matched correlation as a predictor.
- **Expect diminishing alignment payoff as the tied set grows,** since the averaged block must serve more diverse depths. Verma's tie-a-third result and GeoPair's pairwise grouping suggest groups of 2–4 adjacent or similar layers per shared block are the well-trodden regime. A 2-block A-B looped model of a 12–36-layer teacher ties 6–18 layers per block, well beyond published within-model alignment evidence. For that regime, per-iteration LoRA or relaxation (RRT) or per-iteration norms likely remain necessary.

### Gaps
- No published measurement of starting loss or distillation-token savings from aligned vs unaligned averaging when initializing a looped (recursive) transformer. RRT's Average baseline was not permutation-aligned.
- Verma et al.'s exact vanilla-vs-permuted tables (per k, per model) were not retrievable; only the qualitative "consistently better, gap grows" and review-level numbers.
- No evidence was found on aligning attention (heads, QK, VO) for within-model tying. Verma et al. tie FFNs only.
