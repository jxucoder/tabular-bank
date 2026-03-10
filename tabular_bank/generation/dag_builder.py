"""Procedural DAG construction from seed.

Builds a random Directed Acyclic Graph (DAG) that defines causal relationships
between features. The DAG topology, functional forms, and coefficients are all
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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from tabular_bank.templates.scenarios import get_difficulty_preset


@dataclass
class Edge:
    """A directed edge in the causal DAG with a functional form."""

    parent: str
    child: str
    form: str           # "linear", "quadratic", "threshold", "interaction",
                        # "sigmoid", "piecewise_linear", "sinusoidal"
    coefficient: float
    # For threshold / piecewise_linear form
    threshold: float = 0.0
    # For piecewise_linear: slope on each side of the threshold
    slope_left: float = 0.0
    slope_right: float = 1.0
    # For sinusoidal: frequency
    frequency: float = 1.0
    # For interaction form: the second parent
    interaction_parent: str | None = None
    # Whether this edge originates from a latent confounder
    is_confounder: bool = False


@dataclass
class DAGSpec:
    """Complete specification of a procedurally generated DAG."""

    nodes: list[str]                     # All node names in topological order
    target: str                          # Target node name
    root_nodes: list[str]                # Nodes with no parents (exogenous)
    edges: list[Edge]                    # All directed edges
    noise_scales: dict[str, float]       # Per-node noise magnitude
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


# Functional form options
FUNCTIONAL_FORMS = ["linear", "quadratic", "threshold", "sigmoid", "piecewise_linear", "sinusoidal"]

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
    3. For each edge, randomly pick a functional form and coefficient
    4. Inject latent confounders that create hidden common causes
    5. Optionally add AR(1) autocorrelation to root nodes
    6. Validate graph statistics against empirical real-world ranges
    7. The target node is always last in the ordering
    """
    difficulty = get_difficulty_preset(template["difficulty"])

    # Build the node list: features first (random order), then target
    feature_names = [f["name"] for f in features]
    rng.shuffle(feature_names)
    all_nodes = list(feature_names) + [target["name"]]

    n_features = len(feature_names)
    max_parents = difficulty["max_parents"]
    edge_density = difficulty["edge_density"]
    nonlinear_prob = difficulty["nonlinear_prob"]
    interaction_prob = difficulty["interaction_prob"]

    edges: list[Edge] = []
    root_nodes: list[str] = []
    noise_scales: dict[str, float] = {}
    autocorr: dict[str, float] = {}

    for i, node in enumerate(all_nodes):
        if i == 0:
            # First node is always a root
            root_nodes.append(node)
            noise_scales[node] = float(rng.uniform(0.1, 1.0))
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
                noise_scales[node] = float(rng.uniform(0.1, 1.0))
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
            form = _select_functional_form(rng, nonlinear_prob, interaction_prob)
            # Scale coefficient to keep values in reasonable range
            coefficient = float(rng.normal(0, 1)) * 0.5

            edge = Edge(
                parent=parent,
                child=node,
                form=form,
                coefficient=coefficient,
            )

            if form in ("threshold", "piecewise_linear"):
                edge.threshold = float(rng.normal(0, 1))

            if form == "piecewise_linear":
                edge.slope_left = float(rng.uniform(0, 0.5))
                edge.slope_right = float(rng.uniform(0.5, 2.0))

            if form == "sinusoidal":
                edge.frequency = float(rng.uniform(0.5, 3.0))

            if form == "interaction":
                # Pick a second parent for the interaction
                other_parents = [p for p in parents if p != parent]
                if other_parents:
                    edge.interaction_parent = str(rng.choice(other_parents))
                else:
                    # Fall back to linear if no other parent available
                    edge.form = "linear"

            edges.append(edge)

        # Noise scale depends on difficulty
        base_noise = difficulty["noise_scale"]
        noise_scales[node] = float(rng.uniform(base_noise * 0.5, base_noise * 1.5))

    # Inject latent confounders
    confounders = _inject_confounders(rng, all_nodes, edges, difficulty)

    dag = DAGSpec(
        nodes=all_nodes,
        target=target["name"],
        root_nodes=root_nodes,
        edges=edges,
        noise_scales=noise_scales,
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
                form="linear",
                coefficient=coeff,
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


_NONLINEAR_FORMS = ["quadratic", "threshold", "sigmoid", "piecewise_linear", "sinusoidal"]


def _select_functional_form(
    rng: np.random.Generator,
    nonlinear_prob: float,
    interaction_prob: float,
) -> str:
    """Randomly select a functional form for an edge."""
    r = rng.random()
    if r < interaction_prob:
        return "interaction"
    elif r < interaction_prob + nonlinear_prob:
        return str(rng.choice(_NONLINEAR_FORMS))
    else:
        return "linear"
