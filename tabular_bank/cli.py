"""CLI entry point for tabular-bank.

Usage:
    tabular-bank generate --round round-001
    tabular-bank generate --round round-001 --secret "my-secret"
    tabular-bank generate --round round-001 --cache-dir /data/benchmark
    tabular-bank generate --round round-001 --force
    tabular-bank info --round round-001
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tabular_bank.baselines import run_official_baselines
from tabular_bank.board import build_board_site
from tabular_bank.generation.seed import get_default_cache_dir
from tabular_bank.rounds import write_validation_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tabular-bank",
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
    gen_parser.add_argument(
        "--secret",
        default=None,
        help="Master secret (or set TABULAR_BANK_SECRET)",
    )
    gen_parser.add_argument("--cache-dir", default=None, help="Cache directory for generated data")
    gen_parser.add_argument("--force", action="store_true", help="Force regeneration of existing datasets")
    gen_parser.add_argument("--scenario", type=int, default=None, help="Generate only this scenario index")
    gen_parser.add_argument(
        "--n-scenarios",
        type=int,
        default=10,
        help="Number of sampled scenarios to generate for the round",
    )
    gen_parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help=(
            "Override a scenario space parameter.  Ranges use comma-separated "
            "values, e.g. --set n_samples_range=5000,20000. "
            "Repeatable: --set n_features_range=10,50 --set missing_rate_range=0,0"
        ),
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about generated datasets")
    info_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    info_parser.add_argument("--cache-dir", default=None, help="Cache directory")

    validate_parser = subparsers.add_parser("validate", help="Validate a generated round")
    validate_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    validate_parser.add_argument("--cache-dir", default=None, help="Cache directory")

    baseline_parser = subparsers.add_parser("run-baselines", help="Run official baseline models")
    baseline_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    baseline_parser.add_argument("--cache-dir", default=None, help="Cache directory")
    baseline_parser.add_argument(
        "--repeat",
        dest="repeats",
        action="append",
        type=int,
        help="Repeat index to evaluate. Repeat to provide multiple values.",
    )
    baseline_parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        type=int,
        help="Fold index to evaluate. Repeat to provide multiple values.",
    )
    baseline_parser.add_argument(
        "--track",
        dest="tracks",
        action="append",
        choices=["classical", "foundation"],
        help="Restrict the run to one or more method tracks.",
    )

    board_parser = subparsers.add_parser("build-board", help="Build a static leaderboard site")
    board_parser.add_argument("--round", default="round-001", help="Benchmark round ID")
    board_parser.add_argument("--cache-dir", default=None, help="Cache directory")
    board_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the static board site should be written",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "run-baselines":
        _cmd_run_baselines(args)
    elif args.command == "build-board":
        _cmd_build_board(args)
    else:
        parser.print_help()
        sys.exit(1)


def _parse_overrides(raw: list[str] | None) -> dict | None:
    """Parse ``--set key=value`` pairs into a scenario_space dict."""
    if not raw:
        return None
    space: dict = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid --set format (expected KEY=VALUE): {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        # Try to interpret as a numeric tuple/range (e.g. "1000,15000")
        if "," in value:
            parts = [p.strip() for p in value.split(",")]
            try:
                space[key] = tuple(
                    int(p) if p.lstrip("-").isdigit() else float(p)
                    for p in parts
                )
                continue
            except ValueError:
                pass
            space[key] = parts
        else:
            # Single value — try numeric, fall back to string
            try:
                space[key] = int(value) if value.lstrip("-").isdigit() else float(value)
            except ValueError:
                space[key] = value
    return space


def _cmd_generate(args) -> None:
    from tabular_bank.generation.generate import generate_all, generate_one

    if args.scenario is not None and args.scenario < 0:
        raise SystemExit("--scenario must be non-negative")
    if args.n_scenarios <= 0:
        raise SystemExit("--n-scenarios must be positive")

    scenario_space = _parse_overrides(args.overrides)

    if args.scenario is not None:
        path = generate_one(
            scenario_index=args.scenario,
            master_secret=args.secret,
            round_id=args.round,
            cache_dir=args.cache_dir,
            force=args.force,
            scenario_space=scenario_space,
        )
        print(f"Generated dataset: {path}")
    else:
        paths = generate_all(
            master_secret=args.secret,
            round_id=args.round,
            n_scenarios=args.n_scenarios,
            cache_dir=args.cache_dir,
            force=args.force,
            scenario_space=scenario_space,
        )
        print(f"Generated {len(paths)} datasets:")
        for p in paths:
            print(f"  {p}")


def _cmd_info(args) -> None:
    import json

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()
    round_dir = cache_dir / args.round

    if not round_dir.exists():
        print(f"No datasets found for round '{args.round}' in {cache_dir}")
        print("Run 'tabular-bank generate' first.")
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


def _cmd_validate(args) -> None:
    path = write_validation_report(
        round_id=args.round,
        cache_dir=args.cache_dir,
    )
    print(f"Wrote validation report: {path}")


def _cmd_run_baselines(args) -> None:
    if args.repeats and any(repeat < 0 for repeat in args.repeats):
        raise SystemExit("--repeat values must be non-negative")
    if args.folds and any(fold < 0 for fold in args.folds):
        raise SystemExit("--fold values must be non-negative")

    path = run_official_baselines(
        round_id=args.round,
        cache_dir=args.cache_dir,
        repeats=args.repeats,
        folds=args.folds,
        tracks=set(args.tracks) if args.tracks else None,
    )
    print(f"Wrote baseline run manifest: {path}")


def _cmd_build_board(args) -> None:
    path = build_board_site(
        round_id=args.round,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    print(f"Built static board site: {path}")


if __name__ == "__main__":
    main()
