"""
build_data.py - Build ligand-only coordination dataset from Toney CSD data

Fully standalone — all shared functions copied locally (no imports from
stability_gnn). This allows the coord_only/ directory to be deployed to
HPC independently.

Reads Toney et al. (PNAS 2025) CSD coordination CSVs (train+val, ~60K
ligands) and builds ligand-only PyG graphs with coordination labels.
Split 90/10 train/val internally; Toney's held-out test (6,616) is
used by eval_toney.py.

Usage:
    python build_data.py --toney_csv train.csv val.csv
    python build_data.py --toney_csv train.csv val.csv --output_dir /path/to/output
"""

import os
import ast
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import List, Optional
from datetime import datetime

from config import EDGE_TYPES, NODE_FEATURE_DIM, get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (copied from stability_gnn/config.py for standalone operation)
# =============================================================================

# 70 metal types used in stability_gnn feature encoding
TMC_METAL_TYPES = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Li", "Na",
    "other",
]

LINKER_ATOM_FEATURE_DIM = 120  # Must match stability_gnn


# =============================================================================
# ATOM FEATURE FUNCTIONS (copied from stability_gnn/build_data.py)
# =============================================================================

def one_hot(value, choices: List) -> List[float]:
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def is_coordination_site(atom) -> bool:
    """Determine if an atom can coordinate to a metal center."""
    symbol = atom.GetSymbol()
    if symbol == 'N':
        return atom.GetFormalCharge() < 1
    elif symbol in ('O', 'S'):
        return True
    elif symbol == 'P':
        return atom.GetFormalCharge() <= 0
    elif symbol in ('Cl', 'Br', 'I'):
        return True
    return False


def get_ligand_atom_features(atom, metal_type: str = None) -> np.ndarray:
    """
    Compute 120-dim features for ligand atoms. Identical to stability_gnn version.

    For coord-only mode, metal_type is always None (no metal context).
    Feature vector is still 120-dim with zeros for metal context slots.
    """
    features = []

    # Atom type one-hot (10 dims)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
    symbol = atom.GetSymbol()
    if symbol not in atom_types[:-1]:
        symbol = 'other'
    features.extend(one_hot(symbol, atom_types))

    # Degree one-hot (6 dims)
    features.extend(one_hot(min(atom.GetDegree(), 5), [0, 1, 2, 3, 4, 5]))

    # Formal charge one-hot (5 dims)
    features.extend(one_hot(max(-2, min(2, atom.GetFormalCharge())), [-2, -1, 0, 1, 2]))

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
    features.extend(one_hot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4]))

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
    features.extend(one_hot(min(atom.GetImplicitValence(), 3), [0, 1, 2, 3]))

    # --- Context features ---

    # is_metal (1 dim) — always 0 for ligand atoms
    features.append(0.0)

    # metal_type context one-hot (70 dims) — all zeros in coord-only mode
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

    # Pad to 120 dims
    current_dim = len(features)
    padding_needed = LINKER_ATOM_FEATURE_DIM - current_dim
    if padding_needed > 0:
        features.extend([0.0] * padding_needed)

    return np.array(features[:LINKER_ATOM_FEATURE_DIM], dtype=np.float32)


# =============================================================================
# LIGAND-ONLY GRAPH FROM PLAIN SMILES + KNOWN DONOR INDICES
# =============================================================================

def construct_ligand_only_graph(
    smiles: str,
    coord_atom_indices: List[int],
) -> Optional[Data]:
    """
    Build ligand-only graph from plain SMILES with known donor indices.

    Used for evaluating on Toney's test set where we have the ligand SMILES
    and ground-truth coordinating atom indices.
    """
    if not smiles or pd.isna(smiles) or smiles.strip() == '':
        raise ValueError("Empty SMILES")

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles[:100]}")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception as e:
        raise ValueError(f"Sanitization failed: {e}")

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Empty molecule")

    valid_indices = [i for i in coord_atom_indices if 0 <= i < num_atoms]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid coord indices for {num_atoms}-atom molecule")

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        features = get_ligand_atom_features(atom, metal_type=None)
        node_features.append(features)

    # Edges (covalent only)
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
        bt = bond.GetBondType()
        et = bond_type_map.get(bt, EDGE_TYPES['SINGLE'])
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_type.append(et)
        edge_type.append(et)

    if len(edge_index) == 0:
        raise ValueError("No edges")

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)

    num_etypes = 3
    edge_attr = torch.zeros(len(edge_type), num_etypes)
    for i, et in enumerate(edge_type_t):
        if 0 <= et < num_etypes:
            edge_attr[i, et] = 1.0

    coord_labels = torch.zeros(num_atoms, dtype=torch.float32)
    for idx in valid_indices:
        coord_labels[idx] = 1.0

    data = Data(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr,
        edge_type=edge_type_t,
        coord_labels=coord_labels,
        num_atoms=num_atoms,
        num_donors=len(valid_indices),
    )

    return data


# =============================================================================
# TONEY CSV COLUMN NAMES
# =============================================================================

TONEY_SMILES_COL = 'SMILES_without_X_rdkit'
TONEY_CATOM_COL = 'Connecting_Atom_Indices_rdkit_smiles'
TONEY_DENT_COL = 'Ligand_Denticities'
TONEY_SIMGROUP_COL = 'Similarity_Group'


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_dataset(config, toney_csvs):
    """
    Build coordination-only dataset from Toney CSD train+val CSVs.

    Uses the same data Toney trained on (~60K ligands with explicit
    coordination labels). Their held-out test set (6,616) is used by
    eval_toney.py for fair comparison.

    Split: 90/10 train/val from Toney's train+val pool (no test —
    Toney's held-out test is the benchmark).
    """
    import csv as csv_mod
    csv_mod.field_size_limit(2**30)

    # Load all Toney CSVs
    all_dfs = []
    for csv_path in toney_csvs:
        logger.info(f"Loading Toney CSV: {csv_path}")
        usecols = [TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL]
        # Add similarity group if available
        try:
            test_df = pd.read_csv(csv_path, nrows=1)
            if TONEY_SIMGROUP_COL in test_df.columns:
                usecols.append(TONEY_SIMGROUP_COL)
        except Exception:
            pass
        df = pd.read_csv(csv_path, usecols=usecols)
        df['_source_file'] = os.path.basename(csv_path)
        all_dfs.append(df)
        logger.info(f"  Loaded {len(df)} rows")

    toney_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total Toney ligands: {len(toney_df)}")
    logger.info(f"Denticity distribution: {toney_df[TONEY_DENT_COL].value_counts().sort_index().to_dict()}")

    # Build graphs
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
            errors.append((idx, smiles[:60], f'Cannot parse catom indices'))
            continue

        if len(catom_indices) == 0:
            errors.append((idx, smiles[:60], 'No coordinating atoms'))
            continue

        try:
            graph = construct_ligand_only_graph(smiles, catom_indices)
            graphs.append(graph)

            meta_rows.append({
                'smiles': smiles,
                'n_donors': int(row[TONEY_DENT_COL]),
                'n_atoms': graph.num_atoms,
            })

        except Exception as e:
            errors.append((idx, smiles[:60], str(e)))

        if (idx + 1) % 10000 == 0:
            logger.info(f"  Processed {idx+1}/{len(toney_df)} ({len(graphs)} valid)")

    logger.info(f"Built {len(graphs)}/{len(toney_df)} graphs ({len(graphs)/len(toney_df)*100:.1f}%)")
    if errors:
        logger.info(f"Errors: {len(errors)}")
        for i, smi, err in errors[:10]:
            logger.info(f"  Row {i}: {smi} -> {err}")

    meta_df = pd.DataFrame(meta_rows)

    dent_dist = meta_df['n_donors'].value_counts().sort_index()
    logger.info(f"\nDenticity distribution:")
    for d, c in dent_dist.items():
        logger.info(f"  dent={d}: {c}")

    # Split: 90% train, 10% val (no test — Toney's held-out test is the benchmark)
    n_total = len(meta_df)
    n_val = int(n_total * 0.10)

    np.random.seed(42)
    indices = np.random.permutation(n_total)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    meta_df['group'] = 'training'
    meta_df.loc[val_idx, 'group'] = 'validation'

    logger.info(f"\nSplit: train={len(train_idx)}, val={len(val_idx)} (no test — use Toney held-out)")

    # Save
    os.makedirs(config.output_dir, exist_ok=True)

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    torch.save(graphs, graphs_path)
    logger.info(f"Saved {len(graphs)} graphs to {graphs_path}")

    meta_df.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata to {meta_path}")

    stats = {
        'n_total': len(graphs),
        'n_toney_input': len(toney_df),
        'n_errors': len(errors),
        'split': {
            'train': len(train_idx),
            'val': len(val_idx),
        },
        'denticity_distribution': {str(k): int(v) for k, v in dent_dist.items()},
    }
    with open(os.path.join(config.output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    sample = graphs[0]
    logger.info(f"\nSample: nodes={sample.x.shape[0]}, edges={sample.edge_index.shape[1]}, "
                f"donors={int(sample.coord_labels.sum())}")

    return graphs, meta_df


def main():
    parser = argparse.ArgumentParser(description='Build coord-only dataset from Toney CSD data')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--toney_csv', type=str, nargs='+', required=True,
                        help='Toney CSD CSV files (train and/or val)')
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Build Coordination-Only Dataset (Toney CSD)")
    logger.info("=" * 60)
    for csv_path in args.toney_csv:
        logger.info(f"Toney CSV: {csv_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Start: {datetime.now()}")

    build_dataset(config, args.toney_csv)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
