"""Coverage-aware leaderboard artifacts and static site generation."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd

from tabular_bank.baselines import BASELINE_RUN_DIR
from tabular_bank.generation.seed import get_default_cache_dir
from tabular_bank.leaderboard import generate_leaderboard_from_dataframe
from tabular_bank.rounds import (
    ROUND_MANIFEST_NAME,
    VALIDATION_REPORT_NAME,
    get_round_dir,
    write_validation_report,
)


def build_board_site(
    round_id: str,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Build a static leaderboard site for an official round."""
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = get_round_dir(round_id, cache)
    baseline_dir = round_dir / BASELINE_RUN_DIR
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"No official baseline artifacts found for round '{round_id}' in {baseline_dir}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    round_output = output / "rounds" / round_id
    round_output.mkdir(parents=True, exist_ok=True)

    with open(round_dir / ROUND_MANIFEST_NAME) as f:
        manifest = json.load(f)
    validation_path = round_dir / VALIDATION_REPORT_NAME
    if not validation_path.exists():
        validation_path = write_validation_report(round_id, cache_dir=cache)
    with open(validation_path) as f:
        validation = json.load(f)
    with open(baseline_dir / "methods.json") as f:
        methods = json.load(f)
    results = pd.read_csv(baseline_dir / "results.csv")

    artifacts = build_board_artifacts(manifest=manifest, validation=validation, methods=methods, results=results)

    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = data_dir / "rounds.json"
    rounds_index = _load_rounds_index(rounds_path)
    rounds_index = _upsert_round_summary(rounds_index, artifacts["round_summary"])
    with open(rounds_path, "w") as f:
        json.dump(rounds_index, f, indent=2)
    with open(round_output / "leaderboard_overall.json", "w") as f:
        json.dump(artifacts["overall"], f, indent=2)
    with open(round_output / "leaderboard_by_track.json", "w") as f:
        json.dump(artifacts["tracks"], f, indent=2)
    with open(round_output / "round_summary.json", "w") as f:
        json.dump(artifacts["round_summary"], f, indent=2)

    _write_index_html(output / "index.html", rounds_index)
    _write_round_html(round_output / "index.html", artifacts)
    _write_track_html(round_output / "classical.html", artifacts["round_summary"], "classical", artifacts["tracks"]["classical"])
    _write_track_html(round_output / "foundation.html", artifacts["round_summary"], "foundation", artifacts["tracks"]["foundation"])

    styles_src = Path(__file__).resolve().parent.parent / "website" / "styles.css"
    if styles_src.exists():
        shutil.copyfile(styles_src, output / "styles.css")

    return output


def build_board_artifacts(
    manifest: dict,
    validation: dict,
    methods: list[dict],
    results: pd.DataFrame,
) -> dict:
    """Build JSON-serializable leaderboard artifacts."""
    total_tasks = int(manifest["n_scenarios"])
    methods_df = pd.DataFrame(methods)
    if methods_df.empty:
        methods_df = pd.DataFrame(columns=["method_name", "track", "status", "coverage_ratio", "coverage_status"])

    overall_methods = methods_df[
        (methods_df["status"] == "ok") &
        (methods_df["coverage_status"] == "full")
    ]["method_name"].tolist()
    overall_results = results[results["model"].isin(overall_methods)] if not results.empty else results
    overall_board = _merge_leaderboard_metadata(
        generate_leaderboard_from_dataframe(overall_results),
        methods_df,
    )

    track_boards: dict[str, dict] = {}
    for track in ("classical", "foundation"):
        track_methods = methods_df[
            (methods_df["track"] == track) &
            (methods_df["status"] == "ok") &
            (methods_df["coverage_ratio"] > 0)
        ]["method_name"].tolist()
        track_results = results[results["model"].isin(track_methods)] if not results.empty else results
        track_board = _merge_leaderboard_metadata(
            generate_leaderboard_from_dataframe(track_results),
            methods_df[methods_df["track"] == track],
        )
        track_boards[track] = {
            "track": track,
            "entries": track_board.to_dict(orient="records"),
        }

    return {
        "overall": {
            "entries": overall_board.to_dict(orient="records"),
        },
        "tracks": track_boards,
        "round_summary": {
            "round_id": manifest["round_id"],
            "n_scenarios": total_tasks,
            "problem_type_counts": manifest["problem_type_counts"],
            "validation_status": validation["status"],
            "validation_errors": validation["errors"],
            "validation_warnings": validation["warnings"],
            "overall_method_count": len(overall_board),
            "available_tracks": sorted(
                track for track, payload in track_boards.items() if payload["entries"]
            ),
        },
    }


def _merge_leaderboard_metadata(board: pd.DataFrame, methods_df: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame(columns=["rank", "model", "elo", "avg_rank", "win_rate", "mean_score",
                                     "n_tasks", "track", "coverage_ratio", "coverage_status", "status"])

    merged = board.reset_index().merge(
        methods_df.rename(columns={"method_name": "model"}),
        on="model",
        how="left",
    )
    return merged


def _write_index_html(path: Path, rounds_index: list[dict]) -> None:
    round_links = "".join(
        (
            "<li>"
            f"<a href=\"./rounds/{html.escape(str(summary['round_id']), quote=True)}/index.html\">"
            f"{html.escape(str(summary['round_id']))}</a> "
            f"({html.escape(str(summary['n_scenarios']))} scenarios, "
            f"validation={html.escape(str(summary['validation_status']))})"
            "</li>"
        )
        for summary in rounds_index
    )
    path.write_text(
        _html_page(
            title="tabular-bank leaderboard",
            stylesheet_href="./styles.css",
            body=(
                "<h1>tabular-bank leaderboard</h1>"
                "<p>Official benchmark rounds published from static artifacts.</p>"
                f"<ul>{round_links}</ul>"
            ),
        ),
        encoding="utf-8",
    )


def _write_round_html(path: Path, artifacts: dict) -> None:
    summary = artifacts["round_summary"]
    overall_rows = _table_rows(artifacts["overall"]["entries"])
    esc_round_id = html.escape(str(summary['round_id']))
    esc_n_scenarios = html.escape(str(summary['n_scenarios']))
    esc_validation = html.escape(str(summary['validation_status']))
    esc_tracks = html.escape(', '.join(summary['available_tracks'])) or 'none'
    path.write_text(
        _html_page(
            title=f"tabular-bank {esc_round_id}",
            stylesheet_href="../../styles.css",
            body=(
                f"<h1>Round {esc_round_id}</h1>"
                f"<p>Scenarios: {esc_n_scenarios} | Validation: {esc_validation}</p>"
                f"<p>Tracks: {esc_tracks}</p>"
                "<p>"
                "<a href=\"./classical.html\">Classical track</a> | "
                "<a href=\"./foundation.html\">Foundation track</a>"
                "</p>"
                "<h2>Overall leaderboard</h2>"
                f"{overall_rows}"
            ),
        ),
        encoding="utf-8",
    )


def _write_track_html(path: Path, summary: dict, track: str, payload: dict) -> None:
    esc_round_id = html.escape(str(summary['round_id']))
    esc_track = html.escape(track)
    path.write_text(
        _html_page(
            title=f"{esc_round_id} {esc_track}",
            stylesheet_href="../../styles.css",
            body=(
                f"<h1>{html.escape(track.title())} track</h1>"
                f"<p><a href=\"./index.html\">Back to round {esc_round_id}</a></p>"
                f"{_table_rows(payload['entries'])}"
            ),
        ),
        encoding="utf-8",
    )


def _table_rows(entries: list[dict]) -> str:
    if not entries:
        return "<p>No leaderboard entries available.</p>"

    headers = ["rank", "model", "elo", "avg_rank", "win_rate", "mean_score", "n_tasks", "coverage_ratio", "coverage_status"]
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in entries:
        body.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers
            ) + "</tr>"
        )
    return (
        "<table>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _html_page(title: str, body: str, stylesheet_href: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <link rel="stylesheet" href="{stylesheet_href}">
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border-bottom: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
      th {{ background: #f5f5f5; }}
      a {{ color: #0f5fcc; }}
    </style>
  </head>
  <body>{body}</body>
</html>
"""


def _load_rounds_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _upsert_round_summary(rounds_index: list[dict], round_summary: dict) -> list[dict]:
    by_id = {
        str(item.get("round_id")): item
        for item in rounds_index
        if item.get("round_id") is not None
    }
    by_id[str(round_summary["round_id"])] = round_summary
    return [by_id[round_id] for round_id in sorted(by_id)]
