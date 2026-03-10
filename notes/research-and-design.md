# Research & Design Notes

## Project Goal

Generate contamination-resistant tabular ML benchmarks as a drop-in
replacement for TabArena.  Datasets are procedurally generated from causal
DAGs with cryptographic seed derivation (HMAC-SHA256), so they are
deterministic given the secret but unpredictable without it.

---

## Related Work & References

### Directly Comparable

| Paper / Project | Venue | Relevance |
|-----------------|-------|-----------|
| **TabArena** (Erickson et al.) | AutoGluon, 2025 | The benchmark we replace. Uses 51 curated real-world datasets; susceptible to LLM training contamination. |
| **CausalProfiler** (arXiv 2511.22842) | 2025 | Closest prior work. Procedural benchmark generator for causal ML with "coverage guarantees" — any SCM in the defined Space of Interest has non-zero probability of being generated. We borrowed their idea of parametric scenario sampling (`sample_scenario()`). |
| **CauTabBench** (github.com/TURuibo/CauTabBench) | 2025 | Generates tabular data from causal mechanisms (linear-Gaussian, sigmoid-Gaussian, neural network). Different goal (evaluating synthesis quality, not benchmarking predictors). |

### Empirical Evidence for Realism Gaps

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| **TabReD** (Kotelnikov et al., arXiv 2406.19380) | ICLR 2025 Spotlight | Academic benchmarks diverge from real-world data on: (1) temporal drift, (2) correlated/redundant features, (3) class imbalance, (4) missing values. Rankings change significantly when these are present. |
| **CLIMB benchmark** | 2025 | 73 datasets with varying class imbalance. Showed ensemble methods and tuned boosted trees dominate on imbalanced data. |
| **Label noise study** (Springer MACH 2024) | 2024 | Label noise degrades synthetic tabular data quality non-linearly. |

### Synthetic Data Generation Methods (not competitors, but context)

| Method | Venue | Notes |
|--------|-------|-------|
| **TabDiff** | ICLR 2025 | SOTA diffusion model for mixed-type tabular data. 22.5% improvement on column correlation estimation. |
| **CTGAN / TVAE** | NeurIPS 2019 | Standard GAN/VAE baselines for tabular data. Common comparison points. |
| **GReaT** (tabularis-ai) | 2023 | LLM fine-tuning approach; 140K+ downloads. Converts rows to text. |
| **SPADA** (arXiv 2507.19334) | 2025 | LLM-induced sparse dependency graphs. 9,500x faster than LLM baselines. |
| **StructSynth** (arXiv 2508.02601) | 2025 | LLM + DAG structure discovery for low-data regimes. Conceptually similar to our approach but in reverse (they discover the DAG, we define it). |
| **GPT-4o zero-shot** (arXiv 2502.14523) | 2025 | GPT-4o generates high-fidelity tabular data via prompting, outperforming CTGAN. Strengthens our contamination motivation. |
| **DAGAF** (Springer, 2025) | 2025 | DAG-based generation with ANM, LiNGAM, Post-Nonlinear models. Inspired our richer sampled mechanisms (sigmoid, piecewise-linear, spline). |
| **CausalMix** (arXiv 2603.03587) | Mar 2026 | Controllable causal mechanisms via mixture-of-Gaussians. Parameterizes confounding strength and effect heterogeneity independently. Inspired separating difficulty axes. |

### Benchmark Meta-Evaluation

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| **BetterBench** (Stanford) | NeurIPS 2024 Spotlight | 46 best-practice criteria for AI benchmarks. Key: discriminability, statistical significance, ranking stability. |
| **BENCHMARK²** | 2025 | Three quantitative metrics: Cross-Benchmark Ranking Consistency (CBRC), Discriminability Score (DS), Capability Alignment Deviation (CAD). |
| **PSN-IRT** (arXiv 2505.15055) | AAAI 2026 Oral | Item Response Theory for ML benchmarks. Estimates per-task difficulty (b) and discrimination (a) parameters. |
| **IRT-on-Bench** (github.com/lamalab-org/irt-on-bench) | ICLR 2025 | Python package for fitting IRT models to benchmark data. |
| **Benchmark Agreement Testing (BAT)** | 2025 | Standardizes how new benchmarks are validated against established ones. |

### Surveys

| Paper | arXiv ID | Scope |
|-------|----------|-------|
| Shi et al. (Apr 2025) | 2504.16506 | Unified taxonomy: traditional, diffusion, LLM-based methods |
| Lee et al. (Jul 2025) | 2507.11590 | Taxonomy by practical generation objectives; benchmark framework |
| Sheffield systematic review (2025) | 2504.18544 | Evaluation challenges: metric consensus, reproducibility |
| KDD '25 (Meta + Warwick) | — | Privacy attacks and defenses for synthetic tabular data |

---

## Design Decisions

### Why not use TabStruct?

TabStruct (ICLR 2026 Oral) evaluates how well *generators* preserve causal
structure of *existing* data. Our goal is orthogonal — we *generate* new
benchmark tasks, not evaluate generators. Their "global utility" metric is
interesting but designed for a different workflow.

### Why parametric scenario sampling?

Five hand-picked scenarios provide limited coverage. CausalProfiler showed
that sampling from a continuous space gives coverage guarantees (any valid
configuration has non-zero probability). Our `sample_scenario()` draws all
parameters from defined ranges, letting `generate_sampled_datasets()` create
arbitrarily many diverse tasks per round.

### Why separate difficulty axes?

The original design bundled all difficulty knobs into three presets
(easy/medium/hard). CausalMix showed that independently varying noise,
nonlinearity, interaction probability, confounder strength, etc. produces
more diverse and diagnostic benchmarks. The `sample_scenario()` function
samples each axis independently.

### Class imbalance via sigmoid bias

Real-world binary problems are often 90/10 or worse (CLIMB benchmark).
We shift the sigmoid by `logit(imbalance_ratio)`:

    probs = expit(latent + logit(target_ratio))

This preserves the relative ordering of latent values while centering
the class distribution around the desired ratio.

### Correlated root features via multivariate Gaussian

Real datasets have correlated features even without direct causal links
(e.g. height/weight). We sample root node latents jointly from a
multivariate Gaussian with a randomly generated correlation matrix,
then transform each marginal to its target distribution via quantile
mapping (the same rank-based transform already used in
`_transform_to_distribution`).

### Missing value mechanisms

Standard taxonomy from Rubin (1976), used across all modern synthetic
data papers:

- **MCAR**: each entry independently dropped with probability `rate`
- **MAR**: missingness in column A depends on column B's values
- **MNAR**: missingness depends on the value itself (high values more
  likely missing)

Injected as a post-processing step after DAG sampling, before splits.
Target column is never masked.

### Richer sampled mechanisms

The original four forms (linear, quadratic, threshold, interaction)
miss common real-world patterns. A fixed public menu is also easier
for benchmark participants to adapt to over time. From DAGAF and the
causal DAG literature, we expanded the mechanism family and moved edge
specification to structured mechanism dicts rather than a single enum.

Current mechanism families include:

- **sigmoid**: saturating relationships (biological/medical data)
- **tanh**: smooth bounded responses with signed saturation
- **piecewise_linear**: threshold effects with slopes, not just steps
- **sinusoidal**: periodic/seasonal effects
- **spline**: smooth local nonlinearities without committing to a single analytic form

### Heteroscedastic residual noise

Real datasets often violate constant-variance assumptions. We now sample
per-node noise models, not just scalar noise scales. Non-root nodes can
use **heteroscedastic** residuals where the local noise scale depends on a
selected parent feature via a smooth gating function.

This broadens the benchmark support beyond "signal shape only" and makes it
harder to overfit to homoscedastic residual structure.

---

## Meta-Evaluation Framework

Built to answer: "are our benchmarks good benchmarks?"

### Implemented Metrics

1. **Discriminability Score (DS)** — mean pairwise score gap per task.
   Flags tasks where DS < threshold (too easy or too noisy).

2. **Ranking Concordance** — Kendall's tau and Spearman's rho between
   our model rankings and a reference benchmark (e.g. TabArena).
   Target: tau > 0.7 for strong agreement.

3. **Task Diversity** — inter-task Spearman correlation of model scores.
   Good benchmarks have moderate correlations (0.3–0.7). Flags pairs
   with |rho| > 0.9 as redundant.

### Not Yet Implemented

4. **Per-task IRT** — fit a 2-parameter Item Response Theory model
   (difficulty + discrimination) to the task × model score matrix.
   Requires 8+ models to fit reliably. Can use `irt-on-bench` package.

### Validation Workflow

Every generation improvement should be validated via:

    1. Generate round with current settings → compute meta-eval baseline
    2. Add improvement (e.g. class imbalance)
    3. Generate round with new settings → compute meta-eval
    4. Compare: did discriminability go up? Did concordance hold?

---

## What Was Implemented

### Phase 0: Meta-Evaluation
- `tabular_bank/evaluation/meta_eval.py` — discriminability, concordance, diversity
- `tabular_bank/leaderboard.py` — added `get_task_scores()` for score matrix extraction
- `examples/scripts/run_meta_eval.py` — example script

### Phase 1: Generation Improvements
- **Class imbalance** — `imbalance_ratio` in scenarios, sigmoid bias in sampler
- **Noise features** — `noise_feature_ratio` in scenarios, three noise types in sampler
- **Parametric scenarios** — `sample_scenario()`, `SCENARIO_SPACE`, `generate_sampled_datasets()`
- **Correlated roots** — `root_correlation_strength` in difficulty presets, multivariate Gaussian sampling
- **Missing values** — `tabular_bank/generation/missing.py` with MCAR/MAR/MNAR
- **Structured mechanisms** — edge behavior now uses sampled mechanism dicts, with tanh and spline support in dag_builder + sampler
- **Heteroscedastic noise** — per-node noise models can depend on a driver parent instead of using constant variance everywhere

### Runner Fix
- `_encode_features()` now handles NaN in both categorical and numeric columns (median imputation for numerics, -1 for categoricals)

### Follow-on Hardening
- **Simple feature naming restored** — informative and noise features now use readable dataset-local labels like `f_0`, `f_1`, ...
- **Round metadata made authoritative** — cache loading now follows `scenario_ids`, avoiding stale extra datasets when `n_scenarios` changes
- **Sampler fixes** — confounders now affect root nodes, and autocorrelation is applied before marginal distribution transforms so supports stay valid

---

## TODOs

### High Priority
- [ ] Run full meta-evaluation with 8+ diverse models to establish baseline discriminability and diversity numbers
- [ ] Compare model rankings against TabArena's published leaderboard for concordance validation
- [ ] Validate that class imbalance ratios in generated data match target ratios across seeds (statistical test)

### Medium Priority
- [ ] Implement per-task IRT analysis (stretch metric in meta-eval) — consider `irt-on-bench` package
- [ ] Add distribution/concept drift as a generation feature (TabReD identified this as a major gap)
- [ ] Explore time-based train/test splits (TabReD showed these significantly change method rankings)

### Lower Priority
- [ ] Add feature engineering artifacts (derived ratios, aggregates, binned versions) for extra realism
- [ ] Expose meta-eval metrics in the CLI (`tabular-bank info --meta-eval`)
- [ ] Write integration test that runs `generate_sampled_datasets()` end-to-end with meta-eval
- [ ] Add regime-switch / mixture mechanisms beyond single-edge smooth families
- [ ] Benchmark generation speed for large sampled rounds (50+ scenarios)

### Research Questions
- [ ] What is the minimum number of scenarios needed for stable model rankings? (bootstrap analysis)
- [ ] Do our synthetic benchmarks recover known results (e.g. GBDTs > linear models on nonlinear data)?
- [ ] How sensitive are rankings to the `root_correlation_strength` parameter?
- [ ] Is there a principled way to calibrate difficulty parameters so that generated tasks span a target discriminability range?
