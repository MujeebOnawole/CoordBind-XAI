"""
model.py - Coordination-Only RGCN (no binding head)

Same RGCN backbone as stability_gnn/model.py but:
  - No binding head (no triple readout, no logK1 prediction)
  - Coordination head only: per-atom sigmoid for donor prediction
  - 3 edge types (SINGLE, DOUBLE, TRIPLE) — no METAL_COORD
  - No node_roles needed (all atoms are ligand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data


class RGCNCoordOnly(nn.Module):
    """
    RGCN for coordination site prediction only.

    Forward pass:
        120-dim node features
            |
        Input projection (Linear -> ReLU -> Dropout)
            |
        [RGCNConv x N layers, 3 edge types] + residual + LayerNorm
            |
        Coordination Head: per-atom Linear -> sigmoid
            -> P(donor) per atom
    """

    def __init__(
        self,
        node_feature_dim: int = 120,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        num_edge_types: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RGCN convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types, aggr='mean')
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Coordination head: per-atom donor prediction
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Returns:
            coord_logits: (total_nodes,) — raw logits for BCE loss
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res

        coord_logits = self.coord_head(x).squeeze(-1)
        return coord_logits

    def predict_coordination(self, data: Data) -> torch.Tensor:
        """Return per-atom donor probabilities (sigmoid applied)."""
        return torch.sigmoid(self.forward(data))


def create_model(
    node_feature_dim: int = 120,
    hidden_dim: int = 128,
    num_conv_layers: int = 3,
    num_edge_types: int = 3,
    dropout: float = 0.2,
) -> RGCNCoordOnly:
    return RGCNCoordOnly(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        num_edge_types=num_edge_types,
        dropout=dropout,
    )
