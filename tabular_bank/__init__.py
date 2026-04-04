"""tabular-bank: A contamination-proof tabular ML benchmark.

Drop-in replacement for TabArena with procedurally generated synthetic datasets.
All dataset generation is controlled by a secret seed — the repo contains only
the generation engine and minimal scenario templates, no dataset-specific information.
"""

__version__ = "0.1.0"


def _scenario_sort_key(scenario_id: str) -> tuple[str, int, str]:
    """Sort sampled scenario identifiers numerically when possible."""
    prefix, sep, suffix = scenario_id.rpartition("_")
    if sep and suffix.isdigit():
        return (prefix, int(suffix), scenario_id)
    return (scenario_id, -1, scenario_id)
