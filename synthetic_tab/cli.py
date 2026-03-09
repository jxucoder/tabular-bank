"""CLI entry point for synthetic-tab.

Usage:
    synthetic-tab generate --round round-001
    synthetic-tab generate --round round-001 --secret "my-secret"
    synthetic-tab generate --round round-001 --cache-dir /data/benchmark
    synthetic-tab generate --round round-001 --force
    synthetic-tab info --round round-001
"""

from __future__ import annotations

import argparse
import logging
import sys

from synthetic_tab.generation.seed import get_default_cache_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="synthetic-tab",
        description="Contamination-proof tabular ML benchmark",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate benchmark datasets")
    gen_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    gen_parser.add_argument("--secret", default=None, help="Master secret (or set SYNTHETIC_TAB_SECRET)")
    gen_parser.add_argument("--cache-dir", default=None, help="Cache directory for generated data")
    gen_parser.add_argument("--force", action="store_true", help="Force regeneration of existing datasets")
    gen_parser.add_argument("--scenario", type=int, default=None, help="Generate only this scenario index")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about generated datasets")
    info_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    info_parser.add_argument("--cache-dir", default=None, help="Cache directory")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_generate(args) -> None:
    from synthetic_tab.generation.generate import generate_all, generate_one

    if args.scenario is not None:
        path = generate_one(
            scenario_index=args.scenario,
            master_secret=args.secret,
            round_id=args.round,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        print(f"Generated dataset: {path}")
    else:
        paths = generate_all(
            master_secret=args.secret,
            round_id=args.round,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        print(f"Generated {len(paths)} datasets:")
        for p in paths:
            print(f"  {p}")


def _cmd_info(args) -> None:
    import json
    from pathlib import Path

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()
    round_dir = cache_dir / args.round

    if not round_dir.exists():
        print(f"No datasets found for round '{args.round}' in {cache_dir}")
        print("Run 'synthetic-tab generate' first.")
        sys.exit(1)

    meta_file = round_dir / "round_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            round_meta = json.load(f)
        print(f"Round: {round_meta['round_id']}")
        print(f"Datasets: {round_meta['n_datasets']}")
        print()

    for ds_dir in sorted(round_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        meta_path = ds_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  {meta['dataset_id']}:")
        print(f"    Domain:       {meta['domain']}")
        print(f"    Problem type: {meta['problem_type']}")
        print(f"    Samples:      {meta['n_samples']}")
        print(f"    Features:     {meta['n_features']} ({meta['n_continuous']} continuous, {meta['n_categorical']} categorical)")
        print(f"    Target:       {meta['target_name']}")
        if "n_classes" in meta:
            print(f"    Classes:      {meta['n_classes']}")
        print()


if __name__ == "__main__":
    main()
