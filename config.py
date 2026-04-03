"""
config.py - Configuration for Stability Constant (logK1) GNN

Fully standalone -- all constants previously imported from parent tmGNN-XAI
are defined here directly.

Single-task prediction of metal-ligand stability constants using RGCN
with the same 4-edge-type architecture as tmGNN-XAI.

Data: NIST SRD 46 stability constants with dative bond SMILES
Target: logK1 (log10 of first stepwise stability constant)
Range: -1.91 to 29.70
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# METAL TYPES AND PROPERTIES (from parent tmGNN-XAI config.py)
# =============================================================================

TMC_METAL_TYPES = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Th", "U",
    "Al", "Ga", "In", "Sn", "Pb", "Bi",
    "Mg", "Ca", "Sr", "Ba",
    "other",
]

NUM_METAL_TYPES = 56

METAL_PROPERTIES = {
    "Sc": {"atomic_num": 21, "electronegativity": 1.36, "common_coord": 6, "ionic_radius": 0.75},
    "Ti": {"atomic_num": 22, "electronegativity": 1.54, "common_coord": 6, "ionic_radius": 0.61},
    "V": {"atomic_num": 23, "electronegativity": 1.63, "common_coord": 6, "ionic_radius": 0.64},
    "Cr": {"atomic_num": 24, "electronegativity": 1.66, "common_coord": 6, "ionic_radius": 0.62},
    "Mn": {"atomic_num": 25, "electronegativity": 1.55, "common_coord": 6, "ionic_radius": 0.83},
    "Fe": {"atomic_num": 26, "electronegativity": 1.83, "common_coord": 6, "ionic_radius": 0.78},
    "Co": {"atomic_num": 27, "electronegativity": 1.88, "common_coord": 6, "ionic_radius": 0.75},
    "Ni": {"atomic_num": 28, "electronegativity": 1.91, "common_coord": 6, "ionic_radius": 0.69},
    "Cu": {"atomic_num": 29, "electronegativity": 1.9, "common_coord": 4, "ionic_radius": 0.73},
    "Zn": {"atomic_num": 30, "electronegativity": 1.65, "common_coord": 4, "ionic_radius": 0.74},
    "Y": {"atomic_num": 39, "electronegativity": 1.22, "common_coord": 8, "ionic_radius": 0.9},
    "Zr": {"atomic_num": 40, "electronegativity": 1.33, "common_coord": 8, "ionic_radius": 0.84},
    "Nb": {"atomic_num": 41, "electronegativity": 1.6, "common_coord": 6, "ionic_radius": 0.72},
    "Mo": {"atomic_num": 42, "electronegativity": 2.16, "common_coord": 6, "ionic_radius": 0.69},
    "Tc": {"atomic_num": 43, "electronegativity": 1.9, "common_coord": 6, "ionic_radius": 0.65},
    "Ru": {"atomic_num": 44, "electronegativity": 2.2, "common_coord": 6, "ionic_radius": 0.68},
    "Rh": {"atomic_num": 45, "electronegativity": 2.28, "common_coord": 6, "ionic_radius": 0.67},
    "Pd": {"atomic_num": 46, "electronegativity": 2.2, "common_coord": 4, "ionic_radius": 0.86},
    "Ag": {"atomic_num": 47, "electronegativity": 1.93, "common_coord": 2, "ionic_radius": 1.15},
    "Cd": {"atomic_num": 48, "electronegativity": 1.69, "common_coord": 6, "ionic_radius": 0.95},
    "Hf": {"atomic_num": 72, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.83},
    "Ta": {"atomic_num": 73, "electronegativity": 1.5, "common_coord": 6, "ionic_radius": 0.64},
    "W": {"atomic_num": 74, "electronegativity": 2.36, "common_coord": 6, "ionic_radius": 0.66},
    "Re": {"atomic_num": 75, "electronegativity": 1.9, "common_coord": 6, "ionic_radius": 0.63},
    "Os": {"atomic_num": 76, "electronegativity": 2.2, "common_coord": 6, "ionic_radius": 0.63},
    "Ir": {"atomic_num": 77, "electronegativity": 2.2, "common_coord": 6, "ionic_radius": 0.68},
    "Pt": {"atomic_num": 78, "electronegativity": 2.28, "common_coord": 4, "ionic_radius": 0.8},
    "Au": {"atomic_num": 79, "electronegativity": 2.54, "common_coord": 4, "ionic_radius": 1.37},
    "Hg": {"atomic_num": 80, "electronegativity": 2.0, "common_coord": 4, "ionic_radius": 1.02},
    "La": {"atomic_num": 57, "electronegativity": 1.1, "common_coord": 9, "ionic_radius": 1.03},
    "Ce": {"atomic_num": 58, "electronegativity": 1.12, "common_coord": 8, "ionic_radius": 1.01},
    "Pr": {"atomic_num": 59, "electronegativity": 1.13, "common_coord": 8, "ionic_radius": 0.99},
    "Nd": {"atomic_num": 60, "electronegativity": 1.14, "common_coord": 8, "ionic_radius": 0.98},
    "Sm": {"atomic_num": 62, "electronegativity": 1.17, "common_coord": 8, "ionic_radius": 0.96},
    "Eu": {"atomic_num": 63, "electronegativity": 1.2, "common_coord": 8, "ionic_radius": 0.95},
    "Gd": {"atomic_num": 64, "electronegativity": 1.2, "common_coord": 8, "ionic_radius": 0.94},
    "Tb": {"atomic_num": 65, "electronegativity": 1.2, "common_coord": 8, "ionic_radius": 0.92},
    "Dy": {"atomic_num": 66, "electronegativity": 1.22, "common_coord": 8, "ionic_radius": 0.91},
    "Ho": {"atomic_num": 67, "electronegativity": 1.23, "common_coord": 8, "ionic_radius": 0.9},
    "Er": {"atomic_num": 68, "electronegativity": 1.24, "common_coord": 8, "ionic_radius": 0.89},
    "Tm": {"atomic_num": 69, "electronegativity": 1.25, "common_coord": 8, "ionic_radius": 0.88},
    "Yb": {"atomic_num": 70, "electronegativity": 1.1, "common_coord": 8, "ionic_radius": 0.87},
    "Lu": {"atomic_num": 71, "electronegativity": 1.27, "common_coord": 8, "ionic_radius": 0.86},
    "Th": {"atomic_num": 90, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.94},
    "U": {"atomic_num": 92, "electronegativity": 1.38, "common_coord": 8, "ionic_radius": 0.89},
    "Al": {"atomic_num": 13, "electronegativity": 1.61, "common_coord": 6, "ionic_radius": 0.54},
    "Ga": {"atomic_num": 31, "electronegativity": 1.81, "common_coord": 4, "ionic_radius": 0.62},
    "In": {"atomic_num": 49, "electronegativity": 1.78, "common_coord": 6, "ionic_radius": 0.8},
    "Sn": {"atomic_num": 50, "electronegativity": 1.96, "common_coord": 6, "ionic_radius": 0.69},
    "Pb": {"atomic_num": 82, "electronegativity": 2.33, "common_coord": 6, "ionic_radius": 1.19},
    "Bi": {"atomic_num": 83, "electronegativity": 2.02, "common_coord": 6, "ionic_radius": 1.03},
    "Mg": {"atomic_num": 12, "electronegativity": 1.31, "common_coord": 6, "ionic_radius": 0.72},
    "Ca": {"atomic_num": 20, "electronegativity": 1.0, "common_coord": 8, "ionic_radius": 1.0},
    "Sr": {"atomic_num": 38, "electronegativity": 0.95, "common_coord": 8, "ionic_radius": 1.18},
    "Ba": {"atomic_num": 56, "electronegativity": 0.89, "common_coord": 8, "ionic_radius": 1.35},
    "other": {"atomic_num": 0, "electronegativity": 1.5, "common_coord": 6, "ionic_radius": 0.8},
}

TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Th', 'U',
    'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi',
    'Mg', 'Ca', 'Sr', 'Ba',
}

# =============================================================================
# EDGE TYPES (4 real bond types)
# =============================================================================

EDGE_TYPES = {
    'SINGLE': 0,
    'DOUBLE': 1,
    'TRIPLE': 2,
    'METAL_COORD': 3,
}

NUM_EDGE_TYPES = 4

# =============================================================================
# NODE FEATURE DIMENSIONS
# =============================================================================

LINKER_ATOM_FEATURE_DIM = 100
METAL_NODE_FEATURE_DIM = 100

# =============================================================================
# PROPERTY CONFIGURATION
# =============================================================================

# Dataset statistics (from stability_constants_dative_clean.csv, 17740 rows)
LOGK1_STATS = {
    'mean': 6.76,
    'std': 5.55,
    'min': -1.91,
    'max': 29.70,
    'unit': 'log10',
}

# =============================================================================
# DATA PATHS
# =============================================================================

# On HPC: scripts and data/ are siblings in the same directory (TMC_BE/)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DATA_CSV = os.path.join(DATA_DIR, 'stability_constants_dative_clean.csv')

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class Configuration:
    """Configuration for logK1 stability constant prediction."""

    # Task
    task_name: str = "stability_logk1"
    classification: bool = False
    mtl_mode: bool = False  # Single-task only

    # Data paths
    data_csv: str = field(default_factory=lambda: DATA_CSV)
    output_dir: str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "output"))

    # Column names
    smiles_col: str = "smiles"
    target_col: str = "logK1"
    metal_col: str = "metal_type"
    id_col: str = "idx"

    # Property settings
    property_names: List[str] = field(default_factory=lambda: ['logK1'])
    primary_property: str = "logK1"
    property_mean: float = LOGK1_STATS['mean']  # For XAI agreement check

    # Model architecture
    node_feature_dim: int = METAL_NODE_FEATURE_DIM
    hidden_dim: int = 128
    num_conv_layers: int = 3
    num_edge_types: int = NUM_EDGE_TYPES
    dropout: float = 0.2

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15

    # NO normalization (logK1 range is reasonable for MSE)
    normalize_targets: bool = False

    # NO physics constraint
    use_physics_constraint: bool = False

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # CV settings
    n_repeats: int = 3
    n_folds: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def num_properties(self) -> int:
        return 1


# =============================================================================
# OPTUNA SEARCH SPACE
# =============================================================================

# Wide search space — use for new datasets or major architecture changes
# SEARCH_SPACE_WIDE = {
#     'hidden_dim': [64, 128, 256],
#     'num_conv_layers': (2, 5),         # int range
#     'dropout': (0.1, 0.4),             # float range
#     'learning_rate': (1e-4, 1e-2),     # log float range
#     'weight_decay': (1e-6, 1e-3),      # log float range
#     'batch_size': [16, 32, 64],
# }

# Narrowed search space — informed by Run 1 (25 trials, 19,788 entries).
# Run 1 converged to hidden=256, batch=32, low dropout, lr~1e-3.
# Formal charge fix adds signal to previously dead features but does not
# change the architecture, so the optimal region should be similar.
# Ranges narrowed around the Run 1 optimum to focus Optuna on fine-tuning.
SEARCH_SPACE = {
    'hidden_dim': [128, 256],           # 256 won Run 1 decisively; keep 128 as sanity check
    'num_conv_layers': (3, 5),          # 5 was best but 4 competitive; 2 always underperformed
    'dropout': (0.05, 0.25),            # Run 1 top-5 all in 0.10-0.14; widen slightly
    'learning_rate': (5e-4, 3e-3),      # Run 1 best at 1.08e-3; narrow around it
    'weight_decay': (1e-6, 5e-4),       # Run 1 best at 5.3e-5; light regularization preferred
    'batch_size': [32],                 # 32 consistently optimal; 16 too slow, 64 slightly worse
}


def get_config() -> Configuration:
    """Get default configuration."""
    return Configuration()
