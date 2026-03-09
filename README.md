# synthetic-tab

A contamination-proof tabular ML benchmark — drop-in replacement for [TabArena](https://github.com/autogluon/tabarena) with procedurally generated synthetic datasets.

## Why synthetic-tab?

TabArena is the leading benchmark for tabular ML models, but it uses real-world datasets that may be contaminated in LLM/foundation model training data. `synthetic-tab` solves this by generating datasets **procedurally from a secret seed** — the repo contains only the generation engine and minimal scenario templates. No dataset-specific information is ever committed.

### Anti-Contamination Architecture

- **Procedural everything**: Feature names, DAG topology, distributions, functional forms, coefficients, noise — all generated from the seed
- **Cryptographic seed derivation**: HMAC-SHA256 ensures datasets are unpredictable without the master secret
- **Rotating benchmark rounds**: Each round uses a fresh seed; past rounds' seeds are published after expiry
- **Auditable fairness**: All generation code is public — anyone can verify the engine is unbiased

## Installation

```bash
pip install synthetic-tab

# With TabArena integration for official benchmarking
pip install "synthetic-tab[benchmark]"
```

## Quick Start

### Generate Datasets

```bash
# Via CLI
synthetic-tab generate --round round-001 --secret "your-secret"

# Via Python
from synthetic_tab.generation.generate import generate_all
generate_all(master_secret="your-secret", round_id="round-001")
```

### Run a Benchmark

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from synthetic_tab.context import SyntheticTabContext
from synthetic_tab.runner import run_benchmark
from synthetic_tab.leaderboard import generate_leaderboard, format_leaderboard

# Models to benchmark
models = {
    "GBM": GradientBoostingClassifier(n_estimators=100),
    "RF": RandomForestClassifier(n_estimators=100),
}

# Run benchmark
result = run_benchmark(
    models=models,
    round_id="round-001",
    master_secret="your-secret",
)

# Generate leaderboard
leaderboard = generate_leaderboard(result)
print(format_leaderboard(leaderboard))
```

### Inspect Datasets

```bash
synthetic-tab info --round round-001
```

## Architecture

```
Secret + Round ID
       │
       ▼
  HMAC-SHA256  ──► Round Seed
       │
       ├──► Feature Seed ──► Feature Generator ──► Names, Types, Distributions
       ├──► DAG Seed     ──► DAG Builder       ──► Causal Graph, Functional Forms
       ├──► Data Seed    ──► Sampler            ──► Tabular Data (DataFrame)
       └──► Split Seed   ──► Split Generator    ──► Cross-Validation Folds
```

## Scenario Templates

The repo ships 5 minimal scenario templates (no dataset-specific information):

| # | Domain | Problem Type | Features | Difficulty |
|---|--------|-------------|----------|-----------|
| 1 | Commercial | Binary | 8-15 | Medium |
| 2 | Healthcare | Multiclass (4) | 10-18 | Hard |
| 3 | Real Estate | Regression | 10-16 | Medium |
| 4 | Financial | Binary | 7-13 | Easy |
| 5 | HR | Binary | 8-14 | Hard |

## TabArena Compatibility

`synthetic-tab` is designed as a drop-in replacement for TabArena. Generated datasets can be converted to TabArena's `UserTask` format for use with TabArena's full evaluation pipeline (8-fold bagging, standardized HPO, ELO leaderboards).

```python
ctx = SyntheticTabContext(round_id="round-001", master_secret="your-secret")
tabarena_tasks = ctx.get_tabarena_tasks()  # Requires tabarena package
```

## License

Apache-2.0
