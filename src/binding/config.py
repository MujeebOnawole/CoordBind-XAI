"""
config.py - Configuration for Stability Constant (logK1) GNN

Fully standalone -- all constants previously imported from parent tmGNN-XAI
are defined here directly.

Joint prediction of metal-ligand stability constants (logK1) AND
coordination sites using RGCN with 4-edge-type architecture.

Data sources:
  - Binding: Consolidated stability constants (IUPAC + NIST + LOGKPREDICT + Ga/In/Tl/Be)
  - Coordination: Toney et al. PNAS 2025 (53k CSD ligands with donor labels)

Target: logK1 (log10 of first stepwise stability constant) + per-atom donor prediction
Range: -1.91 to 30.70 (logK1); binary (coordination)
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict
from rdkit import Chem


# =============================================================================
# PERIODIC-TABLE-DERIVED METAL DETECTION
# =============================================================================

# Non-metals that appear in SMILES brackets but are NOT coordination metals.
_NON_METALS = {
    'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Si', 'P', 'S', 'Cl', 'Ar',
    'Ge', 'As', 'Se', 'Br', 'Kr',
    'Te', 'I', 'Xe', 'At', 'Rn', 'Og',
}


def is_metal(symbol: str) -> bool:
    """Check if an element symbol is a metal using the periodic table.

    Returns True for all metals: alkali, alkaline earth, transition,
    lanthanide, actinide, and post-transition metals.
    """
    try:
        pt = Chem.GetPeriodicTable()
        num = pt.GetAtomicNumber(symbol)
        return num > 0 and symbol not in _NON_METALS
    except Exception:
        return False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# METAL TYPES AND PROPERTIES (from parent tmGNN-XAI config.py)
# =============================================================================

TMC_METAL_TYPES = [
    # 3d transition metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d transition metals
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d transition metals
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    # Lanthanides (all 15)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides (with data)
    "Ac", "Th", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    # Post-transition metals
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
    # Alkaline earth metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    # Alkali metals
    "Li", "Na",
    "other",
]

NUM_METAL_TYPES = len(TMC_METAL_TYPES)  # 70

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
    "Pm": {"atomic_num": 61, "electronegativity": 1.13, "common_coord": 8, "ionic_radius": 0.97},
    "Ac": {"atomic_num": 89, "electronegativity": 1.1, "common_coord": 9, "ionic_radius": 1.12},
    "Np": {"atomic_num": 93, "electronegativity": 1.36, "common_coord": 8, "ionic_radius": 0.87},
    "Pu": {"atomic_num": 94, "electronegativity": 1.28, "common_coord": 8, "ionic_radius": 0.86},
    "Am": {"atomic_num": 95, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.98},
    "Cm": {"atomic_num": 96, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.97},
    "Bk": {"atomic_num": 97, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.96},
    "Cf": {"atomic_num": 98, "electronegativity": 1.3, "common_coord": 8, "ionic_radius": 0.95},
    "Tl": {"atomic_num": 81, "electronegativity": 1.62, "common_coord": 6, "ionic_radius": 1.50},
    "Be": {"atomic_num": 4, "electronegativity": 1.57, "common_coord": 4, "ionic_radius": 0.27},
    "Ra": {"atomic_num": 88, "electronegativity": 0.9, "common_coord": 8, "ionic_radius": 1.48},
    "Li": {"atomic_num": 3, "electronegativity": 0.98, "common_coord": 4, "ionic_radius": 0.59},
    "Na": {"atomic_num": 11, "electronegativity": 0.93, "common_coord": 6, "ionic_radius": 1.02},
    "other": {"atomic_num": 0, "electronegativity": 1.5, "common_coord": 6, "ionic_radius": 0.8},
}

# Derived from the periodic table — no hardcoded list needed.
TRANSITION_METALS = {
    symbol for symbol in [
        Chem.GetPeriodicTable().GetElementSymbol(z)
        for z in range(1, 119)
    ]
    if is_metal(symbol)
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

LINKER_ATOM_FEATURE_DIM = 120  # 40 base + 77 context + 3 padding (70 metal types)
METAL_NODE_FEATURE_DIM = 120   # 99 features + 21 padding (70 metal types)

# =============================================================================
# PROPERTY CONFIGURATION
# =============================================================================

# Dataset statistics (from stability_constants_dative_consolidated.csv, 19964 rows)
LOGK1_STATS = {
    'mean': 6.76,
    'std': 5.56,
    'min': -1.91,
    'max': 30.70,
    'unit': 'log10',
}

# =============================================================================
# DATA PATHS
# =============================================================================

# On HPC: scripts and data/ are siblings in the same directory (TMC_BE/)
# Resolve data/ from the script dir first (HPC layout, scripts and data siblings),
# then from the repository root (public repo layout: src/binding/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
_data_candidates = [
    os.path.join(SCRIPT_DIR, 'data'),
    os.path.join(_REPO_ROOT, 'data'),
    '/data',  # container bind mount
]
DATA_DIR = next((d for d in _data_candidates if os.path.isdir(d)), _data_candidates[0])
DATA_CSV = os.path.join(DATA_DIR, 'stability_constants_dative_consolidated.csv')

# Toney et al. (PNAS 2025) CSD coordination data
# Downloaded from Zenodo record 13840776 via download_toney_trainset.py
TONEY_DIR = os.path.join(DATA_DIR, 'toney')
TONEY_TRAIN_CSV = os.path.join(TONEY_DIR, 'ligand_data_final_train.csv')
TONEY_VAL_CSV = os.path.join(TONEY_DIR, 'ligand_data_final_val.csv')
TONEY_TEST_CSV = os.path.join(TONEY_DIR, 'ligand_data_final_test.csv')

# Column names in Toney's raw CSV (not the adapted format)
TONEY_SMILES_COL = 'SMILES_without_X_rdkit'
TONEY_CATOM_COL = 'Connecting_Atom_Indices_rdkit_smiles'
TONEY_DENT_COL = 'Ligand_Denticities'
TONEY_SYMBOLS_COL = 'Connecting_Atom_Symbols'
TONEY_SIMGROUP_COL = 'Similarity_Group'

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
    toney_csv: str = ""  # Path to Toney train CSV; empty = binding-only (Run 3 mode)
    toney_val_csv: str = ""  # Path to Toney val CSV (merged into training pool)
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

# Widened search space for Run 3 — architectural changes (coord head, 120-dim
# features, 69 metal types, consolidated dataset with Ga/In/Tl/Be) require
# re-exploration. coord_weight is new and needs range discovery.
SEARCH_SPACE = {
    'hidden_dim': [128, 256],           # 256 won Run 1; keep both
    'num_conv_layers': (3, 6),          # Wider: coord head may benefit from deeper backbone
    'dropout': (0.05, 0.3),             # Wider: joint training may need more regularization
    'learning_rate': (3e-4, 5e-3),      # Wider: new loss landscape from coord head
    'weight_decay': (1e-6, 1e-3),       # Wider: more parameters to regularize
    'batch_size': [32],                 # Keep fixed (consistently optimal)
    'coord_weight': (0.5, 20.0),        # Run 4: widened — 53k Toney coord-only entries
                                        # make coord loss dominant; Optuna finds balance
}


def get_config() -> Configuration:
    """Get default configuration."""
    return Configuration()
