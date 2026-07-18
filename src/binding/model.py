"""
model.py - RGCN Model for logK1 Stability Constant Prediction + Coordination Prediction

Architecture builds on two proven components:

1. METAL_COORD edge type (from tmGNN-XAI):
   Dative coordination bonds get a dedicated weight matrix in message
   passing, separate from covalent bonds. Ablation on 100,703 complexes
   showed 115-2,520% MAE degradation without it.

2. Donor-aware triple readout (motivated by tmGNN-XAI XAI finding):
   tmGNN-XAI analysis revealed donor atoms (N, O, S, P) appear in the
   top-5 most important atoms for >99.8% of complexes — confirming
   ligand field theory at scale. For stability constant prediction,
   we encode this finding architecturally: the readout pools metal,
   donor, and ligand representations separately, giving the output
   head explicit access to the metal-donor interface that determines
   binding strength.

3. Coordination prediction head (NEW):
   Per-atom sigmoid classifier predicts which ligand atoms coordinate
   to the metal. Trained jointly with logK1 via multi-task loss.
   Enables end-to-end prediction from plain ligand SMILES: first
   predict donors, then form dative bonds, then predict binding.

Edges: 4 types (SINGLE, DOUBLE, TRIPLE, METAL_COORD) — parsimonious,
each is a real chemical bond type.

Nodes: 120-dim features (element, properties, metal context) + role
labels (metal/donor/ligand) for readout pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, Optional


class RGCNStability(nn.Module):
    """
    Relational GCN for joint logK1 + coordination prediction.

    Forward pass:
        120-dim node features
            |
        Input projection (Linear → ReLU → Dropout)
            |
        [RGCNConv × N layers, 4 edge types] + residual + LayerNorm
            |
        ┌───────────────────────┬──────────────────────────┐
        │ Coordination Head     │ Binding Head             │
        │ Per-atom Linear→σ     │ Triple pooling:          │
        │ → P(donor) per node   │   metal/donor/ligand     │
        │                       │ → Concat → MLP → logK1   │
        └───────────────────────┴──────────────────────────┘
    """

    def __init__(
        self,
        node_feature_dim: int = 120,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        num_edge_types: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.num_edge_types = num_edge_types

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RGCN convolution layers with LayerNorm + residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_conv_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types, aggr='mean')
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # === Binding head: [metal_pool, donor_pool, ligand_pool] → logK1 ===
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # === Coordination head: per-atom donor prediction ===
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        batch = data.batch

        # Input projection
        x = self.input_proj(x)

        # Message passing with 4 relation-specific weight matrices
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res

        # --- Coordination head: per-atom donor probability ---
        coord_logits = self.coord_head(x).squeeze(-1)  # [total_nodes]

        # --- Binding head: triple readout → logK1 ---
        batch_size = batch.max().item() + 1

        if hasattr(data, 'node_roles') and data.node_roles is not None:
            roles = data.node_roles
            metal_pool = self._role_pool(x, batch, roles, role_id=1, batch_size=batch_size)
            donor_pool = self._role_pool(x, batch, roles, role_id=2, batch_size=batch_size)
            ligand_pool = self._role_pool(x, batch, roles, role_id=0, batch_size=batch_size)
        else:
            # Fallback: global mean pool repeated 3x for dimension compatibility
            pool = global_mean_pool(x, batch)
            metal_pool = donor_pool = ligand_pool = pool

        graph_repr = torch.cat([metal_pool, donor_pool, ligand_pool], dim=1)
        logk1 = self.output_head(graph_repr)

        return {
            'logk1': logk1,
            'coord_logits': coord_logits,
        }

    def predict_logk1(self, data: Data) -> torch.Tensor:
        """Convenience method: return only logK1 (backward compatible)."""
        return self.forward(data)['logk1']

    def predict_coordination(self, data: Data) -> torch.Tensor:
        """Return per-atom donor probabilities (sigmoid applied)."""
        out = self.forward(data)
        return torch.sigmoid(out['coord_logits'])

    def _role_pool(self, x, batch, roles, role_id, batch_size):
        """Mean pool over nodes with a specific role. Zeros if no nodes match."""
        mask = (roles == role_id)
        if mask.any():
            masked_x = x[mask]
            masked_batch = batch[mask]
            pool = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            pool.scatter_add_(0, masked_batch.unsqueeze(1).expand_as(masked_x), masked_x)
            counts = torch.zeros(batch_size, device=x.device)
            counts.scatter_add_(0, masked_batch, torch.ones_like(masked_batch, dtype=torch.float))
            counts = counts.clamp(min=1).unsqueeze(1)
            pool = pool / counts
        else:
            pool = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        return pool


def create_model(
    node_feature_dim: int = 120,
    hidden_dim: int = 128,
    num_conv_layers: int = 3,
    num_edge_types: int = 4,
    dropout: float = 0.2,
) -> RGCNStability:
    return RGCNStability(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        num_edge_types=num_edge_types,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
