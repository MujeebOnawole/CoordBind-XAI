"""
config.py - Configuration for Coordination-Only RGCN Ablation

Purpose: Train RGCN on Toney's 60K CSD ligands for coordination prediction
ONLY (no binding loss, no metal node, ligand-only graphs). This isolates
whether the RGCN architecture can match Toney's D-MPNN when given the same
data and task.

Comparison targets:
  - Run 3: Emergent coord from 3,769 unique binding ligands (bal_acc ~76%)
  - Run 4: Joint binding + 60K CSD coord via probing (bal_acc ~78%)
  - Toney: Dedicated D-MPNN on 70K CSD (bal_acc 96.9%)
  - THIS: RGCN on 60K CSD, coordination-only (isolates architecture)

Architecture: Same RGCN as stability_gnn but:
  - No metal node, no METAL_COORD edges (ligand-only graphs)
  - No binding head (coordination head only)
  - BCE loss only (no MSE on logK1)
  - Edge types: SINGLE=0, DOUBLE=1, TRIPLE=2 (3 types, no METAL_COORD)
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Edge types: covalent only (no METAL_COORD)
EDGE_TYPES = {
    'SINGLE': 0,
    'DOUBLE': 1,
    'TRIPLE': 2,
}
NUM_EDGE_TYPES = 3

# Node feature dimension: ligand atoms only (no metal features)
# Same as stability_gnn ligand atom features (120 dim with padding)
NODE_FEATURE_DIM = 120

# Toney data path resolution: coord_only/data/ → /data (HPC bind mount) → stability_gnn/data/
_candidates = [
    os.path.join(SCRIPT_DIR, 'data'),
    '/data',  # HPC: apptainer --bind ${DATA_DIR}:/data
    os.path.join(os.path.dirname(SCRIPT_DIR), 'data'),
    os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'data'),  # repo root
]
DATA_DIR = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])
TONEY_DIR = os.path.join(DATA_DIR, 'toney')
TONEY_TEST_CSV = os.path.join(TONEY_DIR, 'ligand_data_final_test.csv')

# Toney column names
TONEY_SMILES_COL = 'SMILES_without_X_rdkit'
TONEY_CATOM_COL = 'Connecting_Atom_Indices_rdkit_smiles'
TONEY_DENT_COL = 'Ligand_Denticities'
TONEY_SIMGROUP_COL = 'Similarity_Group'

# Search space for Optuna
SEARCH_SPACE = {
    'hidden_dim': [128, 256],
    'num_conv_layers': (3, 6),
    'dropout': (0.05, 0.3),
    'learning_rate': (3e-4, 5e-3),
    'weight_decay': (1e-6, 1e-3),
    'batch_size': [32],
}


@dataclass
class Configuration:
    """Configuration for coordination-only RGCN ablation."""

    task_name: str = "coord_only"
    classification: bool = True  # BCE, not MSE

    # Output
    output_dir: str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "output"))

    # Model
    node_feature_dim: int = NODE_FEATURE_DIM
    hidden_dim: int = 128
    num_conv_layers: int = 3
    num_edge_types: int = NUM_EDGE_TYPES
    dropout: float = 0.2

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15

    # CV
    n_repeats: int = 3
    n_folds: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


def get_config() -> Configuration:
    return Configuration()
