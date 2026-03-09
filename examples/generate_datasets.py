"""Example: Generate synthetic benchmark datasets.

Run:
    python examples/generate_datasets.py

Or via CLI:
    synthetic-tab generate --round round-001 --secret "demo-secret"
"""

from synthetic_tab.generation.generate import generate_all

# Generate all 5 datasets for a benchmark round
paths = generate_all(
    master_secret="demo-secret-do-not-use-in-production",
    round_id="round-001",
    cache_dir="/tmp/synthetic_tab_demo",
    force=True,
)

print(f"Generated {len(paths)} datasets:")
for p in paths:
    print(f"  {p}")

# Show dataset info
from synthetic_tab.context import SyntheticTabContext

ctx = SyntheticTabContext(
    round_id="round-001",
    cache_dir="/tmp/synthetic_tab_demo",
    auto_generate=False,
)

print("\nDataset metadata:")
print(ctx.get_metadata().to_string(index=False))
