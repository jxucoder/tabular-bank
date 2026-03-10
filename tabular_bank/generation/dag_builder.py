"""Procedural DAG construction from seed.

Builds a random Directed Acyclic Graph (DAG) that defines causal relationships
between features. The DAG topology, sampled mechanisms, and coefficients are all
determined by the seed — nothing is hardcoded.

Realism mechanisms:
- Sparsity bias: parent counts follow a geometric distribution, so most nodes
  have 1-2 parents (matching empirical real-world graph statistics).
- Confounders: latent hidden variables influence multiple observed features,
  creating spurious correlations not explained by the visible graph.
- Graph validation: after construction, the DAG's statistics are checked
  against empirical ranges from real-world tabular datasets.
- Temporal autocorrelation: root nodes can carry an AR(1) signal, injected
  during sampling.
- Heteroscedastic noise: non-root nodes can use sampled noise models whose
  variance depends on one of their parents.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tabular_bank.templates.scenarios import get_difficulty_preset


@dataclass
class Edge:
    """A directed edge in the causal DAG with a sampled mechanism.

    ``form`` and related scalar fields are preserved as a backward-compatible
    legacy interface. Internally, all edge behavior is driven by
    ``mechanism``.
    """

    parent: str
    child: str
    form: str = "linear"
    coefficient: float = 1.0
    threshold: float = 0.0
    slope_left: float = 0.0
    slope_right: float = 1.0
    frequency: float = 1.0
    interaction_parent: str | None = None
    mechanism: dict[str, Any] | None = None
    is_confounder: bool = False

    def __post_init__(self):
        if self.mechanism is None:
            self.mechanism = _legacy_mechanism_from_edge(self)
        else:
            self.mechanism = _normalize_mechanism(self.mechanism)

        # Keep legacy aliases in sync for tests and downstream callers.
        self.form = str(self.mechanism["type"])
        self.threshold = float(self.mechanism.get("threshold", 0.0))
        self.slope_left = float(self.mechanism.get("slope_left", 0.0))
        self.slope_right = float(self.mechanism.get("slope_right", 1.0))
        self.frequency = float(self.mechanism.get("frequency", 1.0))
        self.interaction_parent = self.mechanism.get("interaction_parent")


@dataclass
class DAGSpec:
    """Complete specification of a procedurally generated DAG."""

    nodes: list[str]                     # All node names in topological order
    target: str                          # Target node name
    root_nodes: list[str]                # Nodes with no parents (exogenous)
    edges: list[Edge]                    # All directed edges
    noise_scales: dict[str, float]       # Per-node baseline noise magnitude
    noise_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Latent confounders: name -> list of observed nodes they influence
    confounders: dict[str, list[str]] = field(default_factory=dict)
    # AR(1) autocorrelation coefficients for root nodes (0 = no autocorr)
    autocorr: dict[str, float] = field(default_factory=dict)
    node_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.node_index = {name: i for i, name in enumerate(self.nodes)}

    def get_parents(self, node: str) -> list[Edge]:
        """Get all incoming edges for a node."""
        return [e for e in self.edges if e.child == node]


# Supported mechanism families. ``FUNCTIONAL_FORMS`` is kept as an alias for
# compatibility with older code and documentation.
MECHANISM_TYPES = [
    "linear",
    "quadratic",
    "threshold",
    "sigmoid",
    "tanh",
    "piecewise_linear",
    "sinusoidal",
    "spline",
    "interaction",
]
FUNCTIONAL_FORMS = MECHANISM_TYPES

# Empirical graph statistics from real-world tabular datasets.
# Source: survey of UCI/OpenML datasets with 10-100 features.
_REAL_WORLD_STATS = {
    "mean_in_degree": (1.0, 3.5),   # average number of parents per non-root node
    "root_fraction": (0.1, 0.65),    # fraction of nodes with no parents
    "max_in_degree": (2, 8),         # max parents any single node has
}


def build_dag(
    rng: np.random.Generator,
    features: list[dict],
    target: dict,
    template: dict,
) -> DAGSpec:
    """Construct a random DAG from generated features.

    The algorithm:
    1. Create a random topological ordering of features + target
    2. For each non-root node, sample parent count from a geometric
       distribution (sparsity bias: most nodes get 1-2 parents)
    3. For each edge, randomly pick a mechanism family and coefficient
    4. Inject latent confounders that create hidden common causes
    5. Optionally add AR(1) autocorrelation to root nodes
    6. Sample node noise models, including heteroscedastic residuals
    7. Validate graph statistics against empirical real-world ranges
    8. The target node is always last in the ordering
    """
    difficulty = get_difficulty_preset(template["difficulty"])

    # Build the node list: features first (random order), then target
    feature_names = [f["name"] for f in features]
    rng.shuffle(feature_names)
    all_nodes = list(feature_names) + [target["name"]]

    max_parents = difficulty["max_parents"]
    edge_density = difficulty["edge_density"]
    nonlinear_prob = difficulty["nonlinear_prob"]
    interaction_prob = difficulty["interaction_prob"]

    edges: list[Edge] = []
    root_nodes: list[str] = []
    noise_scales: dict[str, float] = {}
    noise_models: dict[str, dict[str, Any]] = {}
    autocorr: dict[str, float] = {}

    for i, node in enumerate(all_nodes):
        if i == 0:
            # First node is always a root
            root_nodes.append(node)
            base_noise = float(rng.uniform(0.1, 1.0))
            noise_scales[node] = base_noise
            noise_models[node] = {"type": "homoscedastic", "scale": base_noise}
            autocorr[node] = _sample_autocorr(rng, difficulty)
            continue

        # Determine number of parents for this node
        available_parents = all_nodes[:i]

        if node == target["name"]:
            # Target should depend on multiple features, but respect max_parents cap
            min_parents = min(2, len(available_parents))
            upper = min(len(available_parents) + 1, max_parents + 1)
            upper = max(upper, min_parents + 1)
            n_parents = int(rng.integers(min_parents, upper))
        else:
            # For feature nodes: use edge density to decide if this node has parents
            if rng.random() > edge_density and i > 1:
                # Make it a root node (sparsity: some features are exogenous)
                root_nodes.append(node)
                base_noise = float(rng.uniform(0.1, 1.0))
                noise_scales[node] = base_noise
                noise_models[node] = {"type": "homoscedastic", "scale": base_noise}
                autocorr[node] = _sample_autocorr(rng, difficulty)
                continue
            # Sparsity bias: geometric distribution means most nodes get 1-2 parents
            n_parents = _sample_n_parents(rng, max_parents)

        # Select parents randomly
        n_parents = min(n_parents, len(available_parents))
        parent_indices = rng.choice(len(available_parents), size=n_parents, replace=False)
        parents = [available_parents[j] for j in parent_indices]

        # Create edges from each parent
        for parent in parents:
            mechanism = _sample_mechanism(rng, nonlinear_prob, interaction_prob)
            # Scale coefficient to keep values in reasonable range
            coefficient = float(rng.normal(0, 1)) * 0.5

            edge = Edge(
                parent=parent,
                child=node,
                coefficient=coefficient,
                mechanism=mechanism,
            )

            if edge.form == "interaction":
                # Pick a second parent for the interaction
                other_parents = [p for p in parents if p != parent]
                if other_parents:
                    other_parent = str(rng.choice(other_parents))
                    edge.interaction_parent = other_parent
                    edge.mechanism["interaction_parent"] = other_parent
                else:
                    # Fall back to linear if no other parent available
                    edge.form = "linear"
                    edge.mechanism = {"type": "linear"}

            edges.append(edge)

        # Noise scale depends on difficulty
        base_noise = difficulty["noise_scale"]
        sampled_noise = float(rng.uniform(base_noise * 0.5, base_noise * 1.5))
        noise_scales[node] = sampled_noise
        noise_models[node] = _sample_noise_model(rng, sampled_noise, parents, difficulty)

    # Inject latent confounders
    confounders = _inject_confounders(rng, all_nodes, edges, difficulty)

    dag = DAGSpec(
        nodes=all_nodes,
        target=target["name"],
        root_nodes=root_nodes,
        edges=edges,
        noise_scales=noise_scales,
        noise_models=noise_models,
        confounders=confounders,
        autocorr=autocorr,
    )

    # Validate graph statistics against real-world empirical ranges
    _validate_dag_stats(dag)

    return dag


def _sample_n_parents(rng: np.random.Generator, max_parents: int) -> int:
    """Sample parent count with sparsity bias (geometric distribution).

    Real-world causal graphs are sparse: most nodes have 1-2 parents.
    A geometric distribution naturally captures this — it's heavy at low
    counts and has a long tail, clipped at max_parents.
    """
    # Geometric with p=0.5 gives E[X]=2, strongly favouring 1-2 parents
    count = int(rng.geometric(p=0.5))
    return max(1, min(count, max_parents))


def _sample_autocorr(rng: np.random.Generator, difficulty: dict) -> float:
    """Sample an AR(1) autocorrelation coefficient for a root node.

    Most root nodes are i.i.d. (coeff=0). A fraction carry temporal
    structure, with strength scaled by difficulty.
    """
    temporal_prob = difficulty.get("temporal_prob", 0.2)
    if rng.random() < temporal_prob:
        # AR(1) coefficient in (0, 0.9) — stationary
        max_rho = difficulty.get("max_autocorr", 0.7)
        return float(rng.uniform(0.2, max_rho))
    return 0.0


def _inject_confounders(
    rng: np.random.Generator,
    all_nodes: list[str],
    edges: list[Edge],
    difficulty: dict,
) -> dict[str, list[str]]:
    """Inject latent hidden common causes into the DAG.

    A confounder is a variable that causally influences two or more observed
    features but is itself unobserved. This creates spurious correlations
    between its children that cannot be explained by the visible graph alone —
    a key property of real-world messy data.

    Confounder effects are stored separately and applied during sampling.
    """
    n_confounders = difficulty.get("n_confounders", 2)
    confounder_strength = difficulty.get("confounder_strength", 0.4)
    confounders: dict[str, list[str]] = {}

    # Need at least 2 observed nodes to create a confounder
    if len(all_nodes) < 2 or n_confounders == 0:
        return confounders

    for k in range(n_confounders):
        name = f"_latent_{k}"
        # Each confounder influences 2-4 randomly chosen observed nodes
        n_children = int(rng.integers(2, min(5, len(all_nodes) + 1)))
        child_indices = rng.choice(len(all_nodes), size=n_children, replace=False)
        children = [all_nodes[j] for j in child_indices]
        confounders[name] = children

        # Add edges from this confounder into the edge list
        for child in children:
            coeff = float(rng.normal(0, confounder_strength))
            edges.append(Edge(
                parent=name,
                child=child,
                coefficient=coeff,
                mechanism={"type": "linear"},
                is_confounder=True,
            ))

    return confounders


def _validate_dag_stats(dag: DAGSpec) -> None:
    """Warn if DAG statistics fall outside empirical real-world ranges.

    Ranges are derived from a survey of UCI/OpenML tabular datasets.
    This is a soft check — it warns but does not fail.
    """
    observed_nodes = [n for n in dag.nodes if n != dag.target]
    n_observed = len(observed_nodes)
    if n_observed == 0:
        return

    # Compute in-degrees (excluding confounder edges for the structural check)
    in_degrees = {n: 0 for n in dag.nodes}
    for edge in dag.edges:
        if not edge.is_confounder and edge.child in in_degrees:
            in_degrees[edge.child] += 1

    non_root_degrees = [
        d for n, d in in_degrees.items()
        if n not in dag.root_nodes and n != dag.target
    ]

    if not non_root_degrees:
        return

    mean_in_degree = np.mean(non_root_degrees)
    root_fraction = len(dag.root_nodes) / n_observed
    max_in_degree = max(in_degrees.values())

    lo, hi = _REAL_WORLD_STATS["mean_in_degree"]
    if not (lo <= mean_in_degree <= hi):
        warnings.warn(
            f"DAG mean in-degree {mean_in_degree:.2f} outside empirical range "
            f"[{lo}, {hi}]. Graph may be too dense or too sparse.",
            stacklevel=3,
        )

    lo, hi = _REAL_WORLD_STATS["root_fraction"]
    if not (lo <= root_fraction <= hi):
        warnings.warn(
            f"DAG root fraction {root_fraction:.2f} outside empirical range "
            f"[{lo}, {hi}].",
            stacklevel=3,
        )

    lo, hi = _REAL_WORLD_STATS["max_in_degree"]
    if not (lo <= max_in_degree <= hi):
        warnings.warn(
            f"DAG max in-degree {max_in_degree} outside empirical range "
            f"[{lo}, {hi}].",
            stacklevel=3,
        )


_NONLINEAR_MECHANISM_TYPES = [
    "quadratic",
    "threshold",
    "sigmoid",
    "tanh",
    "piecewise_linear",
    "sinusoidal",
    "spline",
]


def _sample_mechanism(
    rng: np.random.Generator,
    nonlinear_prob: float,
    interaction_prob: float,
) -> dict[str, Any]:
    """Randomly sample a structured mechanism specification for an edge."""
    r = rng.random()
    if r < interaction_prob:
        return {"type": "interaction"}
    elif r < interaction_prob + nonlinear_prob:
        mechanism_type = str(rng.choice(_NONLINEAR_MECHANISM_TYPES))
        return _sample_mechanism_params(rng, mechanism_type)
    else:
        return {"type": "linear"}


def _sample_mechanism_params(
    rng: np.random.Generator,
    mechanism_type: str,
) -> dict[str, Any]:
    """Sample the parameters for a specific mechanism family."""
    if mechanism_type == "linear":
        return {"type": "linear"}
    if mechanism_type == "quadratic":
        return {
            "type": "quadratic",
            "center": float(rng.normal(0, 0.5)),
        }
    if mechanism_type == "threshold":
        threshold = float(rng.normal(0, 1))
        low = float(rng.uniform(-0.5, 0.25))
        high = float(rng.uniform(0.75, 1.5))
        return {
            "type": "threshold",
            "threshold": threshold,
            "low_value": low,
            "high_value": high,
        }
    if mechanism_type == "sigmoid":
        return {
            "type": "sigmoid",
            "slope": float(rng.uniform(0.6, 2.5)),
            "offset": float(rng.normal(0, 0.75)),
        }
    if mechanism_type == "tanh":
        return {
            "type": "tanh",
            "slope": float(rng.uniform(0.6, 2.5)),
            "offset": float(rng.normal(0, 0.75)),
        }
    if mechanism_type == "piecewise_linear":
        return {
            "type": "piecewise_linear",
            "threshold": float(rng.normal(0, 1)),
            "slope_left": float(rng.uniform(0, 0.6)),
            "slope_right": float(rng.uniform(0.5, 2.0)),
        }
    if mechanism_type == "sinusoidal":
        return {
            "type": "sinusoidal",
            "frequency": float(rng.uniform(0.5, 3.0)),
            "phase": float(rng.uniform(-np.pi, np.pi)),
        }
    if mechanism_type == "spline":
        n_knots = int(rng.integers(4, 7))
        knots = np.linspace(-2.5, 2.5, n_knots)
        values = rng.normal(0, 0.9, size=n_knots)
        values -= np.mean(values)
        return {
            "type": "spline",
            "knots": knots.tolist(),
            "values": values.tolist(),
        }
    if mechanism_type == "interaction":
        return {"type": "interaction"}
    raise ValueError(f"Unknown mechanism type: {mechanism_type}")


def _sample_noise_model(
    rng: np.random.Generator,
    base_scale: float,
    parents: list[str],
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    """Sample a node noise model from the scenario difficulty settings."""
    heteroscedastic_prob = float(difficulty.get("heteroscedastic_prob", 0.0))
    if not parents or rng.random() >= heteroscedastic_prob:
        return {"type": "homoscedastic", "scale": base_scale}

    low_multiplier = float(rng.uniform(0.35, 1.0))
    high_multiplier = float(rng.uniform(1.0, 2.6))
    if rng.random() < 0.5:
        low_multiplier, high_multiplier = high_multiplier, low_multiplier

    return {
        "type": "heteroscedastic",
        "base_scale": base_scale,
        "driver": str(rng.choice(parents)),
        "low_multiplier": low_multiplier,
        "high_multiplier": high_multiplier,
    }


def _legacy_mechanism_from_edge(edge: Edge) -> dict[str, Any]:
    """Convert legacy scalar edge fields into a structured mechanism dict."""
    mechanism: dict[str, Any] = {"type": edge.form}
    if edge.form in ("threshold", "piecewise_linear"):
        mechanism["threshold"] = float(edge.threshold)
    if edge.form == "piecewise_linear":
        mechanism["slope_left"] = float(edge.slope_left)
        mechanism["slope_right"] = float(edge.slope_right)
    if edge.form == "sinusoidal":
        mechanism["frequency"] = float(edge.frequency)
    if edge.form == "interaction" and edge.interaction_parent is not None:
        mechanism["interaction_parent"] = edge.interaction_parent
    return _normalize_mechanism(mechanism)


def _normalize_mechanism(mechanism: dict[str, Any]) -> dict[str, Any]:
    """Normalize a mechanism dict so all expected keys are present."""
    normalized = dict(mechanism)
    mechanism_type = str(normalized.get("type", "linear"))
    normalized["type"] = mechanism_type

    if mechanism_type == "quadratic":
        normalized.setdefault("center", 0.0)
    elif mechanism_type == "threshold":
        normalized.setdefault("threshold", 0.0)
        normalized.setdefault("low_value", 0.0)
        normalized.setdefault("high_value", 1.0)
    elif mechanism_type in ("sigmoid", "tanh"):
        normalized.setdefault("slope", 1.0)
        normalized.setdefault("offset", 0.0)
    elif mechanism_type == "piecewise_linear":
        normalized.setdefault("threshold", 0.0)
        normalized.setdefault("slope_left", 0.0)
        normalized.setdefault("slope_right", 1.0)
    elif mechanism_type == "sinusoidal":
        normalized.setdefault("frequency", 1.0)
        normalized.setdefault("phase", 0.0)
    elif mechanism_type == "spline":
        knots = normalized.get("knots") or [-2.5, -0.5, 0.5, 2.5]
        values = normalized.get("values") or [-1.0, -0.2, 0.2, 1.0]
        if len(knots) != len(values):
            raise ValueError("Spline mechanism requires equally sized knots and values")
        normalized["knots"] = [float(k) for k in knots]
        normalized["values"] = [float(v) for v in values]
    elif mechanism_type == "interaction":
        normalized.setdefault("interaction_parent", None)
    elif mechanism_type != "linear":
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")

    return normalized
