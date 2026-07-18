"""
build_data.py - Build PyG graph dataset for joint logK1 + coordination prediction

Supports two data sources:
  1. Binding data: Dative-bond SMILES with logK1 stability constants (19,964 entries)
  2. Coordination data: Toney et al. PNAS 2025 CSD ligands with donor labels (53k+)

When --toney_csv is provided, builds ligand-only graphs (no metal node) for Toney
entries and merges with binding graphs. Toney entries have y=NaN (masked in binding
loss) and coord_labels from their CSV donor indices.

Usage:
    python build_data.py
    python build_data.py --toney_csv data/toney/ligand_data_final_train.csv
    python build_data.py --toney_csv data/toney/ligand_data_final_train.csv --toney_val_csv data/toney/ligand_data_final_val.csv
"""

import os
import ast
import csv
import sys
import argparse
import logging
import re
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from config import (
    get_config, SCRIPT_DIR,
    TMC_METAL_TYPES, METAL_PROPERTIES, TRANSITION_METALS,
    EDGE_TYPES, LINKER_ATOM_FEATURE_DIM, METAL_NODE_FEATURE_DIM, NUM_METAL_TYPES,
    TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL, TONEY_SIMGROUP_COL,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# METAL EXTRACTION FROM SMILES
# =============================================================================

def extract_metal_from_smiles(smiles: str) -> Optional[str]:
    """
    Extract metal element from tmQM SMILES.

    tmQM SMILES contain metals like [Y+3], [Sc+3], [Fe+2], etc.

    Args:
        smiles: tmQM SMILES string

    Returns:
        Metal element symbol or None if not found
    """
    if not smiles or pd.isna(smiles):
        return None

    # Pattern to match metal in brackets with optional charge
    # Examples: [Y+3], [Sc+3], [Fe+2], [Co], [Zn+2]
    pattern = r'\[([A-Z][a-z]?)[\+\-]?\d*\]'

    matches = re.findall(pattern, smiles)

    for match in matches:
        if match in TRANSITION_METALS:
            return match

    return None


def find_metal_atom_index(mol) -> Optional[int]:
    """
    Find the index of the metal atom in a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        Index of metal atom or None
    """
    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in TRANSITION_METALS:
            return idx
    return None


# =============================================================================
# ATOM FEATURE COMPUTATION
# =============================================================================

def one_hot(value, choices: List) -> List[float]:
    """Create one-hot encoding."""
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def get_metal_node_features(
    metal_type: str,
    formal_charge: int = None,
    S: int = None,
    mnd: int = None,
) -> np.ndarray:
    """
    Compute features for metal node with context features.

    Features (100 dims total):
    - is_metal: 1 dim (always 1.0)
    - metal_type_onehot: 56 dims
    - atomic_num_normalized: 1 dim
    - electronegativity: 1 dim
    - common_coordination: 1 dim
    - ionic_radius: 1 dim
    - is_transition_metal: 1 dim
    - is_lanthanide: 1 dim
    - is_3d_metal: 1 dim
    - is_4d_metal: 1 dim
    - is_5d_metal: 1 dim
    - period_features: 5 dims (one-hot for periods 3-7)
    - formal_charge: 7 dims (one-hot for charge = -3,-2,-1,0,+1,+2,+3+)
    - spin_multiplicity_S: 7 dims (one-hot for S = 1,2,3,4,5,6,7+)
    - coordination_number_mnd: 9 dims (one-hot for MND = 2,3,4,5,6,7,8,9+,unknown)
    - padding: remaining dims to 100

    Args:
        metal_type: Metal element symbol
        formal_charge: Metal atom formal charge from RDKit (e.g., +2 for [Fe+2], +3 for [Fe+3])
        S: Spin multiplicity (integer, 2S+1)
        mnd: Metal coordination number (number of dative bonds from RDKit)

    Returns:
        numpy array of shape (100,)
    """
    if metal_type not in METAL_PROPERTIES:
        metal_type = 'other'

    props = METAL_PROPERTIES.get(metal_type, METAL_PROPERTIES['other'])

    features = []

    # is_metal indicator (1 dim)
    features.append(1.0)

    # metal_type one-hot (56 dims)
    features.extend(one_hot(metal_type, TMC_METAL_TYPES))

    # atomic number normalized (1 dim)
    features.append(props['atomic_num'] / 100.0)

    # electronegativity normalized (1 dim)
    features.append(props['electronegativity'] / 4.0)

    # common coordination number normalized (1 dim)
    features.append(props['common_coord'] / 8.0)

    # ionic radius normalized (1 dim)
    features.append(props['ionic_radius'] / 2.0)

    # is_transition_metal (1 dim)
    d_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    }
    features.append(1.0 if metal_type in d_metals else 0.0)

    # is_lanthanide (1 dim)
    lanthanides = {'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
    features.append(1.0 if metal_type in lanthanides else 0.0)

    # is_3d_metal (1 dim)
    metals_3d = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'}
    features.append(1.0 if metal_type in metals_3d else 0.0)

    # is_4d_metal (1 dim)
    metals_4d = {'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'}
    features.append(1.0 if metal_type in metals_4d else 0.0)

    # is_5d_metal (1 dim)
    metals_5d = {'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
    features.append(1.0 if metal_type in metals_5d else 0.0)

    # Period features (5 dims)
    period_map = {
        'Sc': 4, 'Ti': 4, 'V': 4, 'Cr': 4, 'Mn': 4, 'Fe': 4, 'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4,
        'Y': 5, 'Zr': 5, 'Nb': 5, 'Mo': 5, 'Tc': 5, 'Ru': 5, 'Rh': 5, 'Pd': 5, 'Ag': 5, 'Cd': 5,
        'La': 6, 'Hf': 6, 'Ta': 6, 'W': 6, 'Re': 6, 'Os': 6, 'Ir': 6, 'Pt': 6, 'Au': 6, 'Hg': 6,
        'Th': 7, 'U': 7,
        'other': 5,
    }
    period = period_map.get(metal_type, 5)
    features.extend(one_hot(period, [3, 4, 5, 6, 7]))

    # =========================================================================
    # CONTEXT FEATURES (derived from RDKit molecule)
    # =========================================================================

    # Metal formal charge (7 dims: -3, -2, -1, 0, +1, +2, +3+)
    # Extracted from RDKit atom.GetFormalCharge() on the metal atom.
    # Critical for distinguishing oxidation states (e.g., Fe+2 vs Fe+3,
    # which have mean logK1 of 6.6 vs 12.8).
    charge_choices = [-3, -2, -1, 0, 1, 2, 3]  # 3 means 3 or higher
    if formal_charge is not None and not pd.isna(formal_charge):
        fc_val = int(formal_charge)
        if fc_val <= -3:
            fc_val = -3
        elif fc_val >= 3:
            fc_val = 3
        features.extend(one_hot(fc_val, charge_choices))
    else:
        # Unknown charge - all zeros (model will learn to handle)
        features.extend([0.0] * len(charge_choices))

    # Spin multiplicity S (7 dims: 1, 2, 3, 4, 5, 6, 7+)
    s_choices = [1, 2, 3, 4, 5, 6, 7]  # 7 means 7 or higher
    if S is not None and not pd.isna(S):
        s_val = int(S)
        if s_val <= 0:
            s_val = 1  # Minimum is singlet
        elif s_val >= 7:
            s_val = 7
        features.extend(one_hot(s_val, s_choices))
    else:
        # Unknown spin - all zeros
        features.extend([0.0] * len(s_choices))

    # Coordination number MND (9 dims: 2, 3, 4, 5, 6, 7, 8, 9+, unknown)
    mnd_choices = [2, 3, 4, 5, 6, 7, 8, 9]  # 9 means 9 or higher
    if mnd is not None and not pd.isna(mnd):
        mnd_val = int(mnd)
        if mnd_val <= 2:
            mnd_val = 2
        elif mnd_val >= 9:
            mnd_val = 9
        features.extend(one_hot(mnd_val, mnd_choices))
        features.append(0.0)  # Not unknown
    else:
        # Unknown coordination - one-hot for "unknown"
        features.extend([0.0] * len(mnd_choices))
        features.append(1.0)  # Unknown flag

    # Pad to 100 dims
    current_dim = len(features)
    padding_needed = METAL_NODE_FEATURE_DIM - current_dim
    if padding_needed > 0:
        features.extend([0.0] * padding_needed)
    elif padding_needed < 0:
        logger.warning(f"Metal node features exceed {METAL_NODE_FEATURE_DIM} dims: {current_dim}")

    return np.array(features[:METAL_NODE_FEATURE_DIM], dtype=np.float32)


def is_coordination_site(atom) -> bool:
    """
    Determine if an atom can coordinate to a metal center.

    Args:
        atom: RDKit atom object

    Returns:
        True if atom can coordinate to metal
    """
    symbol = atom.GetSymbol()

    if symbol == 'N':
        formal_charge = atom.GetFormalCharge()
        if formal_charge >= 1:
            return False
        return True

    elif symbol == 'O':
        return True

    elif symbol == 'S':
        return True

    elif symbol == 'P':
        return atom.GetFormalCharge() <= 0

    elif symbol in ['Cl', 'Br', 'I']:
        # Halides can coordinate
        return True

    return False


def get_ligand_atom_features(atom, metal_type: str = None) -> np.ndarray:
    """
    Compute features for ligand atoms (non-metal atoms).

    Features (100 dims):
    - Standard atom features (~40 dims)
    - Context features (~60 dims)

    Args:
        atom: RDKit atom object
        metal_type: Metal type for context

    Returns:
        numpy array of shape (100,)
    """
    features = []

    # Atom type one-hot (10 dims)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
    symbol = atom.GetSymbol()
    if symbol not in atom_types[:-1]:
        symbol = 'other'
    features.extend(one_hot(symbol, atom_types))

    # Degree one-hot (6 dims)
    degrees = [0, 1, 2, 3, 4, 5]
    degree = min(atom.GetDegree(), 5)
    features.extend(one_hot(degree, degrees))

    # Formal charge one-hot (5 dims)
    charges = [-2, -1, 0, 1, 2]
    charge = max(-2, min(2, atom.GetFormalCharge()))
    features.extend(one_hot(charge, charges))

    # Hybridization one-hot (5 dims)
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb = atom.GetHybridization()
    hyb_encoding = [0.0] * 5
    if hyb in hybridizations:
        hyb_encoding[hybridizations.index(hyb)] = 1.0
    features.extend(hyb_encoding)

    # Aromaticity (1 dim)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Ring membership (1 dim)
    features.append(1.0 if atom.IsInRing() else 0.0)

    # Total Hs one-hot (5 dims)
    total_hs = [0, 1, 2, 3, 4]
    num_hs = min(atom.GetTotalNumHs(), 4)
    features.extend(one_hot(num_hs, total_hs))

    # Chirality (3 dims)
    chirality_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    chiral = atom.GetChiralTag()
    chiral_encoding = [0.0] * 3
    if chiral in chirality_types:
        chiral_encoding[chirality_types.index(chiral)] = 1.0
    features.extend(chiral_encoding)

    # Implicit valence (4 dims)
    valences = [0, 1, 2, 3]
    valence = min(atom.GetImplicitValence(), 3)
    features.extend(one_hot(valence, valences))

    # --- Context features ---

    # is_metal (1 dim) - always 0 for ligand atoms
    features.append(0.0)

    # metal_type context one-hot (56 dims)
    if metal_type:
        features.extend(one_hot(metal_type, TMC_METAL_TYPES))
    else:
        features.extend([0.0] * len(TMC_METAL_TYPES))

    # is_coordination_site (1 dim)
    is_coord = is_coordination_site(atom)
    features.append(1.0 if is_coord else 0.0)

    # donor_type one-hot (5 dims)
    donor_types = ['N', 'O', 'S', 'P', 'other']
    if is_coord:
        donor = atom.GetSymbol() if atom.GetSymbol() in donor_types[:-1] else 'other'
    else:
        donor = 'other'
    features.extend(one_hot(donor, donor_types))

    # Pad to 100 dims
    current_dim = len(features)
    padding_needed = LINKER_ATOM_FEATURE_DIM - current_dim
    if padding_needed > 0:
        features.extend([0.0] * padding_needed)

    return np.array(features[:LINKER_ATOM_FEATURE_DIM], dtype=np.float32)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def construct_tmc_graph(
    smiles: str,
    metal_type: str = None,
) -> Data:
    """
    Construct graph from SMILES (metal complex with dative bonds).

    The SMILES contain the metal with -> notation for dative bonds.
    Example: "CN(C)C=O->[Y+3](<-O)(<-O)..."

    Metal formal charge and coordination number are extracted directly
    from the RDKit molecule (not passed as parameters), ensuring
    Fe+2 vs Fe+3 etc. are distinguished automatically.

    Edge types:
    - SINGLE (0): Single covalent bonds
    - DOUBLE (1): Double covalent bonds
    - TRIPLE (2): Triple covalent bonds
    - METAL_COORD (3): Dative bonds (->)

    Args:
        smiles: SMILES string with metal and dative bonds
        metal_type: Optional metal type (extracted if not provided)

    Returns:
        PyTorch Geometric Data object
    """
    if not smiles or pd.isna(smiles) or smiles.strip() == '':
        raise ValueError("Empty SMILES")

    # Extract metal if not provided
    if metal_type is None:
        metal_type = extract_metal_from_smiles(smiles)
        if metal_type is None:
            metal_type = 'other'

    # Parse SMILES with RDKit
    # RDKit handles dative bonds (->) as DATIVE bond type
    # Use relaxed sanitization: metals like Ga, In, Tl, Be have non-standard
    # valences that RDKit's strict sanitizer rejects when dative bonds are present.
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles[:100]}...")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception as e:
        raise ValueError(f"Sanitization failed for {smiles[:100]}: {e}")

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Empty molecule from SMILES")

    # Find metal atom index
    metal_idx = find_metal_atom_index(mol)

    # =========================================================================
    # Extract metal formal charge and coordination number from the molecule
    # =========================================================================
    metal_formal_charge = None
    metal_dative_count = None

    if metal_idx is not None:
        metal_atom = mol.GetAtomWithIdx(metal_idx)
        # Formal charge: RDKit correctly parses [Fe+2] -> 2, [Fe+3] -> 3
        metal_formal_charge = metal_atom.GetFormalCharge()

        # Coordination number: count dative bonds to the metal
        n_dative = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
                if bond.GetBeginAtomIdx() == metal_idx or bond.GetEndAtomIdx() == metal_idx:
                    n_dative += 1
        metal_dative_count = n_dative if n_dative > 0 else None

    # Build node features
    node_features = []
    coordination_sites = []

    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()

        if symbol in TRANSITION_METALS:
            # Metal node — formal charge and coordination from the molecule
            features = get_metal_node_features(
                symbol,
                formal_charge=metal_formal_charge,
                S=1,  # Spin unknown from SMILES — hardcoded, acknowledged limitation
                mnd=metal_dative_count,
            )
        else:
            # Ligand atom
            features = get_ligand_atom_features(atom, metal_type)

            # Track coordination sites
            if is_coordination_site(atom):
                coordination_sites.append(idx)

        node_features.append(features)

    # Build edges
    edge_index = []
    edge_type = []

    # Map RDKit bond types to our edge types
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: EDGE_TYPES['SINGLE'],
        Chem.rdchem.BondType.DOUBLE: EDGE_TYPES['DOUBLE'],
        Chem.rdchem.BondType.TRIPLE: EDGE_TYPES['TRIPLE'],
        Chem.rdchem.BondType.AROMATIC: EDGE_TYPES['SINGLE'],  # Aromaticity in nodes
        Chem.rdchem.BondType.DATIVE: EDGE_TYPES['METAL_COORD'],  # Dative bond
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_t = bond.GetBondType()
        edge_t = bond_type_map.get(bond_t, EDGE_TYPES['SINGLE'])

        # Bidirectional edges
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_type.append(edge_t)
        edge_type.append(edge_t)

    if len(edge_index) == 0:
        raise ValueError("No edges in molecule")

    # Convert to tensors
    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # Create edge_attr as one-hot (4 edge types)
    num_edge_types = 4
    num_edges = edge_type.shape[0]
    edge_attr = torch.zeros(num_edges, num_edge_types)
    for i, et in enumerate(edge_type):
        if 0 <= et < num_edge_types:
            edge_attr[i, et] = 1.0

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
    )

    # Store metadata as tensors or ints (not None or lists — PyG collation fails on those)
    data.metal_idx = metal_idx if metal_idx is not None else -1
    data.num_atoms = num_atoms
    data.num_coordination_sites = len(coordination_sites)
    data.metal_formal_charge = metal_formal_charge if metal_formal_charge is not None else -999
    data.spin_multiplicity = 1  # Unknown from SMILES — hardcoded
    data.coordination_number = metal_dative_count if metal_dative_count is not None else -999

    return data


# =============================================================================
# LIGAND-ONLY GRAPH CONSTRUCTION (for Toney CSD coordination data)
# =============================================================================

def construct_ligand_only_graph(
    smiles: str,
    coord_atom_indices: List[int],
) -> Data:
    """
    Construct graph from plain ligand SMILES (no metal, no dative bonds).

    Used for Toney et al. CSD coordination data where we have the ligand
    SMILES and known coordinating atom indices but no metal or binding data.

    These graphs have:
    - No metal node (node_roles all 0)
    - No METAL_COORD edges (only SINGLE/DOUBLE/TRIPLE)
    - coord_labels from the provided indices (1.0=donor, 0.0=non-donor)
    - y = NaN (no logK1 target — masked in binding loss)

    Args:
        smiles: Plain ligand SMILES (e.g., "c1ccncc1")
        coord_atom_indices: List of atom indices that are coordinating sites

    Returns:
        PyTorch Geometric Data object
    """
    if not smiles or pd.isna(smiles) or smiles.strip() == '':
        raise ValueError("Empty SMILES")

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles[:100]}...")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception as e:
        raise ValueError(f"Sanitization failed for {smiles[:100]}: {e}")

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Empty molecule from SMILES")

    # Validate coord_atom_indices
    valid_indices = [i for i in coord_atom_indices if 0 <= i < num_atoms]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid coord indices for {num_atoms}-atom molecule")

    # Build node features — all ligand atoms, no metal context
    node_features = []
    for atom in mol.GetAtoms():
        features = get_ligand_atom_features(atom, metal_type=None)
        node_features.append(features)

    # Build edges — only covalent bonds, no METAL_COORD
    edge_index = []
    edge_type = []

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: EDGE_TYPES['SINGLE'],
        Chem.rdchem.BondType.DOUBLE: EDGE_TYPES['DOUBLE'],
        Chem.rdchem.BondType.TRIPLE: EDGE_TYPES['TRIPLE'],
        Chem.rdchem.BondType.AROMATIC: EDGE_TYPES['SINGLE'],
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_t = bond.GetBondType()
        edge_t = bond_type_map.get(bond_t, EDGE_TYPES['SINGLE'])
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_type.append(edge_t)
        edge_type.append(edge_t)

    if len(edge_index) == 0:
        raise ValueError("No edges in molecule")

    # Convert to tensors
    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    num_edge_types = 4
    num_edges = edge_type.shape[0]
    edge_attr = torch.zeros(num_edges, num_edge_types)
    for i, et in enumerate(edge_type):
        if 0 <= et < num_edge_types:
            edge_attr[i, et] = 1.0

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
    )

    # NaN target — binding loss will mask this
    data.y = torch.tensor([float('nan')], dtype=torch.float32)

    # All ligand atoms (no metal, no donor role since no dative bonds in graph)
    data.node_roles = torch.zeros(num_atoms, dtype=torch.long)

    # Coordination labels from Toney's ground truth
    coord_labels = torch.zeros(num_atoms, dtype=torch.float32)
    for idx in valid_indices:
        coord_labels[idx] = 1.0
    # No -1.0 sentinel needed (no metal atoms to mask)
    data.coord_labels = coord_labels

    # Metadata
    data.metal_idx = -1
    data.num_atoms = num_atoms
    data.num_coordination_sites = len(valid_indices)
    data.metal_formal_charge = -999
    data.spin_multiplicity = -999
    data.coordination_number = len(valid_indices)
    data.source = 1  # 0=binding, 1=toney

    return data


def build_toney_dataset(toney_csv_paths: List[str]) -> Tuple[List[Data], pd.DataFrame]:
    """
    Build ligand-only graphs from Toney et al. CSD coordination data.

    Reads the raw Toney CSV(s) directly (not the adapted format) and
    constructs ligand-only PyG graphs with coordination labels.

    Args:
        toney_csv_paths: List of CSV file paths (e.g., [train.csv, val.csv])

    Returns:
        Tuple of (list of PyG Data objects, metadata DataFrame)
    """
    # Increase CSV field size limit for mol2 string columns
    csv.field_size_limit(2**30)

    all_dfs = []
    for csv_path in toney_csv_paths:
        logger.info(f"Loading Toney CSV: {csv_path}")
        # Only load columns we need (skip huge mol2 string columns)
        usecols = [TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL, TONEY_SIMGROUP_COL]
        df = pd.read_csv(csv_path, usecols=usecols)
        df['_source_file'] = os.path.basename(csv_path)
        all_dfs.append(df)
        logger.info(f"  Loaded {len(df)} rows")

    toney_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total Toney ligands: {len(toney_df)}")
    logger.info(f"Denticity distribution: {toney_df[TONEY_DENT_COL].value_counts().sort_index().to_dict()}")

    graphs = []
    meta_rows = []
    errors = []

    for idx, row in toney_df.iterrows():
        smiles = str(row[TONEY_SMILES_COL]).strip()
        if not smiles or smiles == 'nan':
            errors.append((idx, 'nan', 'Empty SMILES'))
            continue

        # Parse coordinating atom indices
        try:
            catom_indices = ast.literal_eval(str(row[TONEY_CATOM_COL]))
            if isinstance(catom_indices, (int, float)):
                catom_indices = [int(catom_indices)]
            else:
                catom_indices = [int(x) for x in catom_indices]
        except (ValueError, SyntaxError):
            errors.append((idx, smiles[:60], f'Cannot parse catom indices: {row[TONEY_CATOM_COL]}'))
            continue

        if len(catom_indices) == 0:
            errors.append((idx, smiles[:60], 'No coordinating atoms'))
            continue

        try:
            graph = construct_ligand_only_graph(smiles, catom_indices)
            graphs.append(graph)

            meta_rows.append({
                'smiles': smiles,
                'logK1': float('nan'),
                'metal_type': 'coord_only',  # Sentinel for stratification
                'group': 'training',  # Toney train+val go to our training pool
                'scaffold': '',  # Filled in later
                'source': 'toney',
                'denticities': int(row[TONEY_DENT_COL]),
                'similarity_group': int(row[TONEY_SIMGROUP_COL]),
            })

        except Exception as e:
            errors.append((idx, smiles[:60], str(e)))

        if (idx + 1) % 10000 == 0:
            logger.info(f"  Processed {idx+1}/{len(toney_df)} ligands ({len(graphs)} valid)")

    logger.info(f"Toney graphs: {len(graphs)}/{len(toney_df)} ({len(graphs)/len(toney_df)*100:.1f}%)")
    if errors:
        logger.info(f"Toney errors: {len(errors)}")
        for i, smi, err in errors[:10]:
            logger.info(f"  Row {i}: {smi} -> {err}")

    meta_df = pd.DataFrame(meta_rows)
    return graphs, meta_df


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_dataset(config):
    """
    Build PyG graph dataset from stability constants CSV.

    Steps:
    1. Load CSV with smiles, logK1, metal_type
    2. Convert each SMILES to PyG graph using construct_tmc_graph()
    3. Scaffold-based split (80/10/10)
    4. Save graphs and metadata
    """
    logger.info(f"Loading data from {config.data_csv}")
    df = pd.read_csv(config.data_csv)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")

    # Dataset statistics
    logger.info(f"\nlogK1 statistics:")
    logger.info(f"  Mean: {df[config.target_col].mean():.2f}")
    logger.info(f"  Std:  {df[config.target_col].std():.2f}")
    logger.info(f"  Min:  {df[config.target_col].min():.2f}")
    logger.info(f"  Max:  {df[config.target_col].max():.2f}")

    # Metal type distribution
    metal_counts = df[config.metal_col].value_counts()
    logger.info(f"\nMetal types ({len(metal_counts)}):")
    for metal, count in metal_counts.head(20).items():
        logger.info(f"  {metal}: {count}")
    if len(metal_counts) > 20:
        logger.info(f"  ... and {len(metal_counts) - 20} more")

    # =========================================================================
    # Convert SMILES to graphs
    # =========================================================================
    logger.info(f"\nConverting SMILES to PyG graphs...")

    graphs = []
    valid_indices = []
    errors = []

    for idx, row in df.iterrows():
        smiles = row[config.smiles_col]
        metal_type = row[config.metal_col]

        try:
            # Construct graph using local construct_tmc_graph()
            # Formal charge and coordination number are extracted from
            # the SMILES by RDKit (e.g., [Fe+2] -> charge=2, dative bond count -> mnd)
            graph = construct_tmc_graph(
                smiles=smiles,
                metal_type=metal_type,
            )

            # Assign logK1 as target
            graph.y = torch.tensor([row[config.target_col]], dtype=torch.float32)

            # Assign node roles: 0=ligand, 1=metal, 2=donor
            # Metal: atom with symbol in TRANSITION_METALS
            # Donor: non-metal atom that has a DATIVE bond to a metal
            # Ligand: everything else
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None
            n_nodes = graph.x.shape[0]
            node_roles = torch.zeros(n_nodes, dtype=torch.long)  # default: ligand (0)

            if mol is not None and mol.GetNumAtoms() == n_nodes:
                metal_indices = set()
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in TRANSITION_METALS:
                        node_roles[atom.GetIdx()] = 1  # metal
                        metal_indices.add(atom.GetIdx())

                # Donor: non-metal atom with dative bond to metal
                for bond in mol.GetBonds():
                    if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
                        begin = bond.GetBeginAtomIdx()
                        end = bond.GetEndAtomIdx()
                        if begin in metal_indices and end not in metal_indices:
                            node_roles[end] = 2  # donor
                        elif end in metal_indices and begin not in metal_indices:
                            node_roles[begin] = 2  # donor

            graph.node_roles = node_roles

            # Per-atom coordination labels for coordination prediction head.
            # 1.0 for donor atoms (role=2), 0.0 for ligand atoms (role=0),
            # -1.0 for metal atoms (role=1) — metal atoms are masked during
            # coordination loss computation (they are never donors).
            coord_labels = torch.full((n_nodes,), 0.0)
            coord_labels[node_roles == 2] = 1.0
            coord_labels[node_roles == 1] = -1.0  # mask sentinel
            graph.coord_labels = coord_labels
            graph.source = 0  # binding data

            # NOTE: Do NOT store Python strings (smiles, metal_type_str) on the
            # graph — older PyG versions can't collate them in DataLoader.
            # All metadata is in the companion CSV (stability_logk1_meta.csv).

            graphs.append(graph)
            valid_indices.append(idx)

        except Exception as e:
            errors.append((idx, smiles[:80] if smiles else 'None', str(e)))

    logger.info(f"Binding graphs: {len(graphs)}/{len(df)} ({len(graphs)/len(df)*100:.1f}%)")
    if errors:
        logger.info(f"Errors: {len(errors)}")
        for i, smi, err in errors[:10]:
            logger.info(f"  Row {i}: {smi} -> {err}")

    # Filter df to valid rows only
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    df_valid['source'] = 'binding'

    # =========================================================================
    # Merge Toney coordination data (if provided)
    # =========================================================================
    toney_csv_paths = []
    if config.toney_csv and os.path.exists(config.toney_csv):
        toney_csv_paths.append(config.toney_csv)
    if config.toney_val_csv and os.path.exists(config.toney_val_csv):
        toney_csv_paths.append(config.toney_val_csv)

    n_binding = len(graphs)
    if toney_csv_paths:
        logger.info(f"\n{'='*60}")
        logger.info("Building Toney CSD coordination graphs...")
        logger.info(f"{'='*60}")

        toney_graphs, toney_meta = build_toney_dataset(toney_csv_paths)

        # Merge graphs and metadata
        graphs.extend(toney_graphs)

        # Align toney_meta columns with df_valid
        for col in df_valid.columns:
            if col not in toney_meta.columns:
                toney_meta[col] = np.nan if col == config.target_col else ''
        toney_meta = toney_meta[[c for c in df_valid.columns if c in toney_meta.columns]]

        df_valid = pd.concat([df_valid, toney_meta], ignore_index=True)

        logger.info(f"\nMerged dataset: {len(graphs)} total ({n_binding} binding + {len(toney_graphs)} Toney)")
    else:
        logger.info("\nNo Toney CSV provided — binding-only mode (Run 3 compatible)")

    # =========================================================================
    # Scaffold-based train/val/test split
    # =========================================================================
    # Ligand scaffold splitting ensures the model generalises to unseen
    # ligand chemotypes, not just unseen metal-ligand combinations of known
    # scaffolds. This is the coordination chemistry equivalent of Murcko
    # scaffold splitting in drug discovery.
    #
    # For each complex, we extract the LIGAND scaffold (ignoring the metal)
    # and group all complexes with the same ligand scaffold together. Entire
    # scaffold groups are assigned to train, val, or test -- so if EDTA-Cu
    # is in train, EDTA-Fe is also in train (same ligand scaffold). The
    # test set contains ligand scaffolds the model has never seen.
    # =========================================================================
    logger.info(f"\nExtracting ligand scaffolds for splitting...")

    def get_ligand_scaffold(smiles):
        """Extract Murcko scaffold of the ligand (metal removed)."""
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                mol = None
        if mol is None:
            return 'unknown'

        # Remove metal atoms to get pure ligand scaffold
        rw = Chem.RWMol(mol)
        metal_atom_indices = []
        for atom in rw.GetAtoms():
            if atom.GetSymbol() in TRANSITION_METALS:
                metal_atom_indices.append(atom.GetIdx())

        # Remove metal atoms (reverse order to preserve indices)
        for atom_idx in sorted(metal_atom_indices, reverse=True):
            rw.RemoveAtom(atom_idx)

        try:
            ligand_mol = rw.GetMol()
            Chem.SanitizeMol(ligand_mol)
            # Get the largest fragment (in case removing metal splits the ligand)
            frags = Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
            if frags:
                ligand_mol = max(frags, key=lambda m: m.GetNumAtoms())
            scaffold = GetScaffoldForMol(ligand_mol)
            generic = MakeScaffoldGeneric(scaffold)
            return Chem.MolToSmiles(generic)
        except Exception:
            # If scaffold extraction fails, use the ligand SMILES as-is
            try:
                return Chem.MolToSmiles(ligand_mol)
            except Exception:
                return 'unknown'

    # Split scaffold extraction: binding entries use smiles_col, Toney entries
    # use their own smiles column (already in smiles_col after merge)
    scaffolds = df_valid[config.smiles_col].apply(get_ligand_scaffold)
    df_valid['scaffold'] = scaffolds

    scaffold_counts = scaffolds.value_counts()
    logger.info(f"  Unique scaffolds: {len(scaffold_counts)}")
    logger.info(f"  Top 5 scaffolds: {dict(scaffold_counts.head(5))}")

    # =========================================================================
    # Split strategy for joint dataset:
    #
    # BINDING entries: scaffold-split 80/10/10 (exactly as before)
    # TONEY entries: ALL go to 'training' (we use Toney's own test set for
    #   coordination evaluation, so Toney entries never go to our val/test)
    #
    # Cross-source scaffold overlap: if a Toney ligand shares a scaffold
    # with a binding entry, the Toney ligand still goes to training.
    # This is fine because Toney entries don't have logK1 and can't leak
    # binding information into the test set.
    # =========================================================================

    # Separate binding and Toney indices
    is_binding = df_valid['source'] == 'binding'
    binding_indices = df_valid[is_binding].index.tolist()
    toney_indices = df_valid[~is_binding].index.tolist()

    logger.info(f"  Binding entries for scaffold split: {len(binding_indices)}")
    logger.info(f"  Toney entries (all -> training): {len(toney_indices)}")

    # Scaffold-split ONLY binding entries
    binding_scaffold_groups = defaultdict(list)
    for idx in binding_indices:
        binding_scaffold_groups[df_valid.loc[idx, 'scaffold']].append(idx)

    n_binding_total = len(binding_indices)
    test_target = int(n_binding_total * config.test_ratio)
    val_target = int(n_binding_total * config.val_ratio)
    max_group_size = int(n_binding_total * 0.10)

    sorted_scaffolds = sorted(binding_scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)

    train_idx, val_idx, test_idx = [], [], []

    for scaffold, indices in sorted_scaffolds:
        if len(indices) > max_group_size:
            np.random.seed(42)
            np.random.shuffle(indices)
            n = len(indices)
            n_test = int(n * config.test_ratio)
            n_val = int(n * config.val_ratio)
            test_idx.extend(indices[:n_test])
            val_idx.extend(indices[n_test:n_test + n_val])
            train_idx.extend(indices[n_test + n_val:])
        else:
            if len(test_idx) < test_target:
                test_idx.extend(indices)
            elif len(val_idx) < val_target:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)

    # Toney entries all go to training
    train_idx.extend(toney_indices)

    # Assign group labels
    df_valid['group'] = 'training'
    df_valid.loc[val_idx, 'group'] = 'validation'
    df_valid.loc[test_idx, 'group'] = 'test'

    n_train_binding = sum(1 for i in train_idx if i in set(binding_indices))
    n_train_toney = sum(1 for i in train_idx if i in set(toney_indices))
    logger.info(f"  Train: {len(train_idx)} ({n_train_binding} binding + {n_train_toney} Toney)")
    logger.info(f"  Val:   {len(val_idx)} (binding only)")
    logger.info(f"  Test:  {len(test_idx)} (binding only)")

    # Scaffold overlap check (binding splits only — should be zero)
    train_scaffolds = set(df_valid.iloc[[i for i in train_idx if i in set(binding_indices)]]['scaffold'])
    val_scaffolds = set(df_valid.iloc[val_idx]['scaffold'])
    test_scaffolds = set(df_valid.iloc[test_idx]['scaffold'])

    tv_overlap = train_scaffolds & val_scaffolds
    tt_overlap = train_scaffolds & test_scaffolds
    vt_overlap = val_scaffolds & test_scaffolds
    logger.info(f"  Scaffold overlap train-val: {len(tv_overlap)} (should be 0)")
    logger.info(f"  Scaffold overlap train-test: {len(tt_overlap)} (should be 0)")
    logger.info(f"  Scaffold overlap val-test: {len(vt_overlap)} (should be 0)")

    # Metal distribution check (binding entries only — Toney has no metals)
    binding_df = df_valid[df_valid['source'] == 'binding']
    all_metals = set(binding_df[config.metal_col])
    train_metals = set(binding_df[binding_df['group'] == 'training'][config.metal_col])
    val_metals = set(binding_df[binding_df['group'] == 'validation'][config.metal_col])
    test_metals = set(binding_df[binding_df['group'] == 'test'][config.metal_col])

    logger.info(f"\n  All metals (binding): {len(all_metals)}")
    logger.info(f"  Train metals: {len(train_metals)}")
    logger.info(f"  Val metals: {len(val_metals)}")
    logger.info(f"  Test metals: {len(test_metals)}")

    missing_test = all_metals - test_metals
    if missing_test:
        logger.info(f"  Metals not in test (scaffold-driven): {missing_test}")

    # logK distribution check (binding entries only)
    for split, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = df_valid.iloc[idxs]
        bind_vals = split_df[split_df['source'] == 'binding'][config.target_col]
        if len(bind_vals) > 0:
            logger.info(f"  {split} logK1: mean={bind_vals.mean():.2f} std={bind_vals.std():.2f} range=[{bind_vals.min():.2f}, {bind_vals.max():.2f}] (n={len(bind_vals)})")
        toney_count = (split_df['source'] == 'toney').sum()
        if toney_count > 0:
            logger.info(f"  {split} Toney coord-only: {toney_count}")

    # =========================================================================
    # Save
    # =========================================================================
    os.makedirs(config.output_dir, exist_ok=True)

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    # Save graphs
    torch.save(graphs, graphs_path)
    logger.info(f"\nSaved {len(graphs)} graphs to {graphs_path}")

    # Save metadata
    meta_cols = [config.smiles_col, config.target_col, config.metal_col, 'group', 'scaffold', 'source']
    if 'n_atoms' in df_valid.columns:
        meta_cols.append('n_atoms')
    if 'n_dative_bonds' in df_valid.columns:
        meta_cols.append('n_dative_bonds')
    if 'donor_elements' in df_valid.columns:
        meta_cols.append('donor_elements')
    if 'denticities' in df_valid.columns:
        meta_cols.append('denticities')

    df_valid[meta_cols].to_csv(meta_path, index=False)
    logger.info(f"Saved metadata to {meta_path}")

    # Save dataset statistics for XAI
    binding_logk = binding_df[config.target_col]
    stats = {
        'logK1': {
            'mean': float(binding_logk.mean()),
            'std': float(binding_logk.std()),
            'min': float(binding_logk.min()),
            'max': float(binding_logk.max()),
        },
        'n_samples': len(graphs),
        'n_binding': n_binding,
        'n_toney': len(graphs) - n_binding,
        'n_metals': len(all_metals),
        'metals': sorted(list(all_metals)),
        'n_scaffolds': len(scaffold_counts),
        'split_method': 'ligand_scaffold (binding) + all-train (toney)',
        'split': {
            'train': len(train_idx),
            'train_binding': n_train_binding if toney_csv_paths else len(train_idx),
            'train_toney': n_train_toney if toney_csv_paths else 0,
            'val': len(val_idx),
            'test': len(test_idx),
        },
        'scaffold_overlap': {
            'train_val': len(tv_overlap),
            'train_test': len(tt_overlap),
            'val_test': len(vt_overlap),
        },
    }
    stats_path = os.path.join(config.output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved dataset statistics to {stats_path}")

    # Verify graph features
    sample = graphs[0]
    logger.info(f"\nSample graph:")
    logger.info(f"  Nodes: {sample.x.shape[0]}, Features: {sample.x.shape[1]}")
    logger.info(f"  Edges: {sample.edge_index.shape[1]}")
    logger.info(f"  Edge types: {torch.unique(sample.edge_type).tolist()}")
    logger.info(f"  Target (logK1): {sample.y.item():.2f}")

    # =========================================================================
    # Verify formal charge and coordination number encoding
    # =========================================================================
    charge_counts = Counter()
    mnd_counts = Counter()
    for g in graphs:
        fc = g.metal_formal_charge
        mnd_val = g.coordination_number
        charge_counts[int(fc)] += 1
        mnd_counts[int(mnd_val)] += 1

    logger.info(f"\nMetal formal charge distribution (from RDKit GetFormalCharge):")
    for fc in sorted(charge_counts.keys()):
        label = f"+{fc}" if fc > 0 else str(fc)
        logger.info(f"  charge={label}: {charge_counts[fc]}")

    logger.info(f"\nCoordination number distribution (dative bond count):")
    for mnd_val in sorted(mnd_counts.keys()):
        label = str(mnd_val) if mnd_val != -999 else "unknown"
        logger.info(f"  mnd={label}: {mnd_counts[mnd_val]}")

    # Spot-check: Fe+2 vs Fe+3 should have different feature vectors
    fe2_graphs = [g for i, g in enumerate(graphs) if df_valid.iloc[i][config.metal_col] == 'Fe' and g.metal_formal_charge == 2]
    fe3_graphs = [g for i, g in enumerate(graphs) if df_valid.iloc[i][config.metal_col] == 'Fe' and g.metal_formal_charge == 3]
    if fe2_graphs and fe3_graphs:
        fe2_metal = fe2_graphs[0].x[fe2_graphs[0].metal_idx].numpy()
        fe3_metal = fe3_graphs[0].x[fe3_graphs[0].metal_idx].numpy()
        diff_dims = int(np.sum(fe2_metal != fe3_metal))
        # Formal charge one-hot starts after: is_metal(1) + metal_type(NUM_METAL_TYPES) + 4 real-valued
        charge_start = 1 + NUM_METAL_TYPES + 4
        charge_end = charge_start + 7
        logger.info(f"\nFe+2 vs Fe+3 spot-check:")
        logger.info(f"  Fe+2 count: {len(fe2_graphs)}, Fe+3 count: {len(fe3_graphs)}")
        logger.info(f"  Feature dims that differ: {diff_dims}/{METAL_NODE_FEATURE_DIM}")
        logger.info(f"  Fe+2 charge one-hot (dims {charge_start}-{charge_end}): {fe2_metal[charge_start:charge_end].tolist()}")
        logger.info(f"  Fe+3 charge one-hot (dims {charge_start}-{charge_end}): {fe3_metal[charge_start:charge_end].tolist()}")
        if diff_dims > 0:
            logger.info(f"  CONFIRMED: Fe+2 and Fe+3 have different feature vectors")
        else:
            logger.warning(f"  WARNING: Fe+2 and Fe+3 have IDENTICAL features — charge encoding may not be working")

    return graphs, df_valid


def main():
    parser = argparse.ArgumentParser(description='Build graph dataset for logK1 + coordination prediction')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Binding stability constants CSV (dative SMILES)')
    parser.add_argument('--toney_csv', type=str, default=None,
                        help='Toney CSD train CSV (ligand_data_final_train.csv)')
    parser.add_argument('--toney_val_csv', type=str, default=None,
                        help='Toney CSD val CSV (ligand_data_final_val.csv) — merged into training')
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.data_csv:
        config.data_csv = args.data_csv
    if args.toney_csv:
        config.toney_csv = args.toney_csv
    if args.toney_val_csv:
        config.toney_val_csv = args.toney_val_csv

    logger.info("=" * 60)
    logger.info("Build Graph Dataset for logK1 + Coordination Prediction")
    logger.info("=" * 60)
    logger.info(f"Binding data: {config.data_csv}")
    if config.toney_csv:
        logger.info(f"Toney train:  {config.toney_csv}")
    if config.toney_val_csv:
        logger.info(f"Toney val:    {config.toney_val_csv}")
    else:
        logger.info("Toney data:   None (binding-only mode)")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Start: {datetime.now()}")

    build_dataset(config)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
