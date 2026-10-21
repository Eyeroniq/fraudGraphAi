"""Temporal edge classifiers on the causal neighbourhood (task 3.4).

Both variants embed the target edge, turn each earlier incoming edge of the destination
into a message, aggregate the messages with PyG aggregation layers, and classify the
edge from [edge embedding || aggregate || log(1 + #neighbours)]. The 32-d layer before
the head is the edge embedding exported for the hybrid (3.5).

- "sage": GraphSAGE-style, mean + max over neighbour messages.
- "gine": GINE-style, messages conditioned on the target edge (MLP(nb + edge)), summed.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch_geometric.nn import aggr

from fraudgraph.graph.sampler import EDGE_FEATURES, NEIGHBOUR_FEATURES


def mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class TemporalEdgeGNN(nn.Module):
    def __init__(
        self, conv: str = "sage", hidden: int = 64, embedding: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if conv not in {"sage", "gine"}:
            raise ValueError(f"unknown conv {conv!r}")
        self.conv = conv
        d_edge, d_nb = len(EDGE_FEATURES), len(NEIGHBOUR_FEATURES)
        self.edge_mlp = mlp([d_edge, hidden, hidden], dropout)
        self.message_mlp = mlp([d_nb, hidden, hidden], dropout)
        if conv == "sage":
            self.aggregate: aggr.Aggregation = aggr.MultiAggregation(["mean", "max"])
            agg_dim = 2 * hidden
        else:
            self.condition = nn.Linear(hidden, hidden)
            self.gine_mlp = mlp([hidden, hidden, hidden], dropout)
            self.aggregate = aggr.SumAggregation()
            agg_dim = hidden
        self.to_embedding = mlp([hidden + agg_dim + 1, hidden, embedding], dropout)
        self.head = nn.Linear(embedding, 1)

    def embed(self, edge: torch.Tensor, nbs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = nbs.shape[0]
        h_edge = self.edge_mlp(edge)  # [n, hidden]
        rows, cols = mask.nonzero(as_tuple=True)  # flatten valid neighbours
        messages = self.message_mlp(nbs[rows, cols])  # [m, hidden]
        if self.conv == "gine":
            messages = self.gine_mlp(torch.relu(messages + self.condition(h_edge)[rows]))
        pooled = self.aggregate(messages, rows, dim_size=n)  # empty rows -> zeros
        count = torch.log1p(mask.sum(dim=1, keepdim=True).float())
        return torch.relu(self.to_embedding(torch.cat([h_edge, pooled, count], dim=1)))

    def forward(self, edge: torch.Tensor, nbs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.embed(edge, nbs, mask)).squeeze(-1)
        return logits
