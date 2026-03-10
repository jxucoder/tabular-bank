"""Example: Generate synthetic benchmark datasets.

Run:
    python examples/scripts/generate_datasets.py

Or via CLI:
    tabular-bank generate --round round-001 --secret "demo-secret"
"""

import _bootstrap  # noqa: F401
from tabular_bank.generation.generate import generate_all

# Generate a sampled benchmark round
paths = generate_all(
    master_secret="demo-secret-do-not-use-in-production",
    round_id="round-001",
    n_scenarios=5,
    cache_dir="/tmp/tabular_bank_demo",
    force=True,
)

print(f"Generated {len(paths)} datasets:")
for p in paths:
    print(f"  {p}")

# Show dataset info
from tabular_bank.context import TabularBankContext

ctx = TabularBankContext(
    round_id="round-001",
    cache_dir="/tmp/tabular_bank_demo",
    auto_generate=False,
)

print("\nDataset metadata:")
print(ctx.get_metadata().to_string(index=False))
