"""Procedural DAG construction from seed.

Builds a random Directed Acyclic Graph (DAG) that defines causal relationships
between features. The DAG topology, functional forms, and coefficients are all
determined by the seed — nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synthetic_tab.templates.scenarios import get_difficulty_preset


@dataclass
class Edge:
    """A directed edge in the causal DAG with a functional form."""

    parent: str
    child: str
    form: str           # "linear", "quadratic", "threshold", "interaction"
    coefficient: float
    # For threshold form
    threshold: float = 0.0
    # For interaction form: the second parent
    interaction_parent: str | None = None


@dataclass
class DAGSpec:
    """Complete specification of a procedurally generated DAG."""

    nodes: list[str]                 # All node names in topological order
    target: str                      # Target node name
    root_nodes: list[str]            # Nodes with no parents (exogenous)
    edges: list[Edge]                # All directed edges
    noise_scales: dict[str, float]   # Per-node noise magnitude
    node_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.node_index = {name: i for i, name in enumerate(self.nodes)}

    def get_parents(self, node: str) -> list[Edge]:
        """Get all incoming edges for a node."""
        return [e for e in self.edges if e.child == node]


# Functional form options
FUNCTIONAL_FORMS = ["linear", "quadratic", "threshold"]


def build_dag(
    rng: np.random.Generator,
    features: list[dict],
    target: dict,
    template: dict,
) -> DAGSpec:
    """Construct a random DAG from generated features.

    The algorithm:
    1. Create a random topological ordering of features + target
    2. For each non-root node, randomly assign 1-max_parents parents
       from earlier in the ordering
    3. For each edge, randomly pick a functional form and coefficient
    4. The target node is always last in the ordering
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

    for i, node in enumerate(all_nodes):
        if i == 0:
            # First node is always a root
            root_nodes.append(node)
            noise_scales[node] = float(rng.uniform(0.1, 1.0))
            continue

        # Determine number of parents for this node
        available_parents = all_nodes[:i]

        if node == target["name"]:
            # Target should depend on multiple features
            min_parents = max(2, n_features // 3)
            n_parents = rng.integers(min_parents, min(len(available_parents) + 1, n_features + 1))
        else:
            # For feature nodes: use edge density to decide if this node has parents
            if rng.random() > edge_density and i > 1:
                # Make it a root node
                root_nodes.append(node)
                noise_scales[node] = float(rng.uniform(0.1, 1.0))
                continue
            n_parents = rng.integers(1, min(max_parents + 1, len(available_parents) + 1))

        # Select parents randomly
        parent_indices = rng.choice(
            len(available_parents),
            size=min(int(n_parents), len(available_parents)),
            replace=False,
        )
        parents = [available_parents[j] for j in parent_indices]

        # Create edges from each parent
        for parent in parents:
            form = _select_functional_form(rng, nonlinear_prob, interaction_prob)

            coefficient = float(rng.normal(0, 1))
            # Scale coefficient to keep values in reasonable range
            coefficient *= 0.5

            edge = Edge(
                parent=parent,
                child=node,
                form=form,
                coefficient=coefficient,
            )

            if form == "threshold":
                edge.threshold = float(rng.normal(0, 1))

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

    return DAGSpec(
        nodes=all_nodes,
        target=target["name"],
        root_nodes=root_nodes,
        edges=edges,
        noise_scales=noise_scales,
    )


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
        return rng.choice(["quadratic", "threshold"])
    else:
        return "linear"
