"""
eval_coordination_binding.py - Predict coordination sites from binding model

Evaluates whether a model trained on 19,964 stability constants can predict
which atoms coordinate to a metal — the same task Toney et al. (PNAS 2025)
achieved with 70,069 explicitly labeled CSD ligands.

Two complementary methods probe the model's coordination knowledge:

METHOD A — Coord Head Probing:
    For each candidate donor atom, build a graph with ONE METAL_COORD edge
    to that atom. The coord head (trained jointly with binding) evaluates
    whether that atom belongs in a coordination environment. Atoms with
    highest coord probability = predicted coordinating atoms.

METHOD B — Plausibility Probing:
    Same graphs, but rank candidates by predicted logK1. A complex with
    chemically valid coordination produces reasonable logK1; invalid
    coordination produces unreasonable logK1. This uses the binding
    head as a chemical plausibility detector.

Both methods use greedy sequential selection for multi-dentate ligands.

Why this works: the model processed 19,964 graphs with dative bonds. By
probing each candidate atom with a METAL_COORD edge, we present a
training-like graph and ask: "does this atom belong in coordination?"

Comparison with Toney et al. (PNAS 2025):
    - Toney: 70,069 CSD ligands, explicit coordination labels, metal-agnostic
    - Ours: 19,964 stability constants, NO coordination labels, metal-AWARE
    - Our model learns coordination as an emergent property of binding

Metrics (matching Toney et al. Table 1):
    - Overall accuracy, Positive accuracy, Negative accuracy
    - Balanced accuracy, Molecular accuracy

Additional outputs:
    - Predicted dative-bond SMILES for each ligand-metal combination
    - Per-metal coordination predictions (Toney is metal-agnostic)
    - Cross-metal consistency analysis

Usage:
    python eval_coordination_binding.py --checkpoint_dir output/cv_results/checkpoints
    python eval_coordination_binding.py --metals Cu Ni Fe Zn Co
"""

import os
import ast
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from config import (
    get_config, TRANSITION_METALS, TMC_METAL_TYPES,
    METAL_PROPERTIES, METAL_NODE_FEATURE_DIM, LINKER_ATOM_FEATURE_DIM,
    EDGE_TYPES,
)
from build_data import (
    get_metal_node_features, get_ligand_atom_features,
    one_hot, is_coordination_site,
)
from model import RGCNStability

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_probing_graph(
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
    donor_indices: List[int],
) -> Optional[Data]:
    """
    Build a graph from plain ligand SMILES + metal, with METAL_COORD edges
    to specific donor atoms.

    Args:
        ligand_smiles: Plain ligand SMILES (no metal, no dative bonds)
        metal_symbol: Metal element symbol
        metal_charge: Metal formal charge
        donor_indices: Atom indices to connect to metal via METAL_COORD

    Returns:
        PyG Data object, or None on failure
    """
    mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return None

    n_ligand_atoms = mol.GetNumAtoms()
    metal_idx = n_ligand_atoms

    # Build ligand node features
    node_features = []
    for atom in mol.GetAtoms():
        features = get_ligand_atom_features(atom, metal_symbol)
        node_features.append(features)

    # Metal node features
    mnd = len(donor_indices) if donor_indices else None
    metal_features = get_metal_node_features(
        metal_symbol,
        formal_charge=metal_charge,
        S=1,
        mnd=mnd,
    )
    node_features.append(metal_features)

    # Edges: ligand covalent bonds
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: EDGE_TYPES['SINGLE'],
        Chem.rdchem.BondType.DOUBLE: EDGE_TYPES['DOUBLE'],
        Chem.rdchem.BondType.TRIPLE: EDGE_TYPES['TRIPLE'],
        Chem.rdchem.BondType.AROMATIC: EDGE_TYPES['SINGLE'],
    }

    edge_src_dst = []
    edge_types = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        et = bond_type_map.get(bond.GetBondType(), EDGE_TYPES['SINGLE'])
        edge_src_dst.append([i, j])
        edge_src_dst.append([j, i])
        edge_types.append(et)
        edge_types.append(et)

    # Add METAL_COORD edges
    for donor_idx in donor_indices:
        edge_src_dst.append([donor_idx, metal_idx])
        edge_src_dst.append([metal_idx, donor_idx])
        edge_types.append(EDGE_TYPES['METAL_COORD'])
        edge_types.append(EDGE_TYPES['METAL_COORD'])

    if len(edge_src_dst) == 0:
        return None

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    edge_index = torch.tensor(edge_src_dst, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    num_edges = edge_type.shape[0]
    edge_attr = torch.zeros(num_edges, 4)
    for i_e in range(num_edges):
        et = edge_type[i_e].item()
        if 0 <= et < 4:
            edge_attr[i_e, et] = 1.0

    # Node roles: metal=1, connected donors=2, other=0
    n_total = n_ligand_atoms + 1
    node_roles = torch.zeros(n_total, dtype=torch.long)
    node_roles[metal_idx] = 1
    for donor_idx in donor_indices:
        node_roles[donor_idx] = 2

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        node_roles=node_roles,
    )
    data.metal_idx = metal_idx
    data.n_ligand_atoms = n_ligand_atoms

    return data


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_ensemble(checkpoint_dir: str, config, device) -> List[RGCNStability]:
    """Load top-5 checkpoints."""
    import glob

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    hp_path = os.path.join(os.path.dirname(checkpoint_dir), '..', 'best_hyperparameters.json')
    if not os.path.exists(hp_path):
        hp_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)),
                               'best_hyperparameters.json')

    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
    else:
        ckpt = torch.load(ckpt_files[0], map_location='cpu')
        hp = ckpt.get('hyperparameters', {})

    node_feature_dim = hp.get('node_feature_dim', config.node_feature_dim)

    all_ckpts = []
    for f in ckpt_files:
        ckpt = torch.load(f, map_location='cpu')
        val_mae = ckpt.get('val_mae', float('inf'))
        all_ckpts.append((f, val_mae, ckpt))

    all_ckpts.sort(key=lambda x: x[1])
    top_k = min(5, len(all_ckpts))

    models = []
    for path, mae, ckpt in all_ckpts[:top_k]:
        model = RGCNStability(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 256),
            num_conv_layers=hp.get('num_conv_layers', 4),
            num_edge_types=config.num_edge_types,
            dropout=hp.get('dropout', 0.1),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
        logger.info(f"  Loaded {os.path.basename(path)}: val_MAE={mae:.4f}")

    return models


# ============================================================================
# PROBING FUNCTIONS
# ============================================================================

def probe_candidate_coordination(
    models: List[RGCNStability],
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
    candidate_idx: int,
    existing_donors: List[int],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Probe a single candidate atom's coordination ability.

    Builds a graph with METAL_COORD edges to existing_donors + candidate_idx,
    then extracts the coord_head probability for the candidate atom AND the
    predicted logK1.

    Returns:
        coord_prob: Coordination probability for the candidate atom
        logk1: Predicted logK1 for this complex configuration
    """
    donor_set = existing_donors + [candidate_idx]
    data = build_probing_graph(ligand_smiles, metal_symbol, metal_charge, donor_set)
    if data is None:
        return 0.0, -999.0

    batch = Batch.from_data_list([data]).to(device)

    coord_probs_all = []
    logk1_all = []

    with torch.no_grad():
        for model in models:
            out = model(batch)
            probs = torch.sigmoid(out['coord_logits'])
            # Get probability for the candidate atom specifically
            coord_probs_all.append(probs[candidate_idx].item())
            logk1_all.append(out['logk1'].item())

    return np.mean(coord_probs_all), np.mean(logk1_all)


def greedy_coord_selection(
    models: List[RGCNStability],
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
    target_denticity: int,
    candidates: List[Tuple[int, str]],
    device: torch.device,
    method: str = 'coord',  # 'coord' or 'plausibility'
) -> Tuple[List[int], List[float]]:
    """
    Greedily select donor atoms by coordination signal.

    method='coord': rank by coord_head probability (coordination signal)
    method='plausibility': rank by predicted logK1 (chemical plausibility)

    At each step, probes all remaining candidates and selects the best.
    """
    selected = []
    scores = []
    remaining = [c[0] for c in candidates]

    for step in range(target_denticity):
        if not remaining:
            break

        best_idx = None
        best_score = -float('inf')

        for cand_idx in remaining:
            coord_prob, logk1 = probe_candidate_coordination(
                models, ligand_smiles, metal_symbol, metal_charge,
                cand_idx, selected, device,
            )

            score = coord_prob if method == 'coord' else logk1
            if score > best_score:
                best_score = score
                best_idx = cand_idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
            scores.append(best_score)
        else:
            break

    return selected, scores


def auto_denticity_selection(
    models: List[RGCNStability],
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
    candidates: List[Tuple[int, str]],
    device: torch.device,
    method: str = 'coord',
    max_denticity: int = 6,
    coord_threshold: float = 0.5,
) -> Tuple[List[int], List[float], int]:
    """
    Greedy selection with automatic denticity detection.

    For method='coord': stops when best candidate has P(donor) < coord_threshold
    For method='plausibility': stops when logK1 decreases
    """
    selected = []
    scores = []
    remaining = [c[0] for c in candidates]
    prev_logk1 = -float('inf')

    for step in range(min(max_denticity, len(remaining))):
        if not remaining:
            break

        best_idx = None
        best_score = -float('inf')
        best_logk1 = -float('inf')

        for cand_idx in remaining:
            coord_prob, logk1 = probe_candidate_coordination(
                models, ligand_smiles, metal_symbol, metal_charge,
                cand_idx, selected, device,
            )
            score = coord_prob if method == 'coord' else logk1
            if score > best_score:
                best_score = score
                best_idx = cand_idx
                best_logk1 = logk1

        if best_idx is None:
            break

        # Stopping criteria
        if method == 'coord' and best_score < coord_threshold:
            break
        if method == 'plausibility' and step > 0 and best_logk1 < prev_logk1:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        scores.append(best_score)
        prev_logk1 = best_logk1

    return selected, scores, len(selected)


# ============================================================================
# CANDIDATE IDENTIFICATION
# ============================================================================

def find_candidate_donors(ligand_smiles: str) -> List[Tuple[int, str]]:
    """Identify candidate coordinating atoms in a plain ligand SMILES."""
    mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
    if mol is None:
        return []
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return []

    candidates = []
    for atom in mol.GetAtoms():
        if is_coordination_site(atom):
            candidates.append((atom.GetIdx(), atom.GetSymbol()))
    return candidates


# ============================================================================
# DATIVE SMILES OUTPUT
# ============================================================================

def build_dative_smiles(
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
    donor_indices: List[int],
) -> Optional[str]:
    """Construct dative-bond SMILES from ligand + metal + predicted donors."""
    mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return None

    rwmol = Chem.RWMol(mol)
    metal_idx = rwmol.AddAtom(Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(metal_symbol)))
    rwmol.GetAtomWithIdx(metal_idx).SetFormalCharge(metal_charge)

    for donor_idx in donor_indices:
        rwmol.AddBond(donor_idx, metal_idx, Chem.rdchem.BondType.DATIVE)

    try:
        Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return Chem.MolToSmiles(rwmol)
    except Exception:
        donor_strs = [f"{mol.GetAtomWithIdx(d).GetSymbol()}@{d}" for d in donor_indices]
        return f"{ligand_smiles}.[{metal_symbol}+{metal_charge}]<-{'+'.join(donor_strs)}"


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    true_indices: List[int],
    pred_indices: List[int],
    n_atoms: int,
) -> Dict[str, float]:
    """Compute Toney-compatible per-atom and molecular metrics."""
    true_set = set(true_indices)
    pred_set = set(pred_indices)

    true_labels = np.zeros(n_atoms)
    pred_labels = np.zeros(n_atoms)
    for idx in true_set:
        if idx < n_atoms:
            true_labels[idx] = 1
    for idx in pred_set:
        if idx < n_atoms:
            pred_labels[idx] = 1

    tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (true_labels == 0)))
    tn = int(np.sum((pred_labels == 0) & (true_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (true_labels == 1)))

    overall_acc = (tp + tn) / n_atoms if n_atoms > 0 else 0
    pos_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    neg_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (pos_acc + neg_acc) / 2
    molecular_acc = 1.0 if true_set == pred_set else 0.0
    dent_correct = 1.0 if len(pred_set) == len(true_set) else 0.0

    return {
        'overall_acc': overall_acc, 'pos_acc': pos_acc, 'neg_acc': neg_acc,
        'balanced_acc': balanced_acc, 'molecular_acc': molecular_acc,
        'dent_correct': dent_correct,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'true_dent': len(true_set), 'pred_dent': len(pred_set),
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_method(
    method_name: str,
    method: str,
    df: pd.DataFrame,
    metals: List[str],
    charges: Dict[str, int],
    models: List[RGCNStability],
    device: torch.device,
    output_dir: str,
) -> Dict:
    """Run evaluation for one method (coord or plausibility) across all metals."""

    logger.info(f"\n{'#'*60}")
    logger.info(f"METHOD: {method_name}")
    logger.info(f"{'#'*60}")

    all_results = {}

    for metal in metals:
        charge = charges.get(metal, 2)
        logger.info(f"\n{'='*60}")
        logger.info(f"{method_name}: {metal}+{charge}")
        logger.info(f"{'='*60}")

        per_ligand = []
        failures = 0

        for idx, row in df.iterrows():
            smiles = row['smiles']
            true_indices = ast.literal_eval(row['coordinating_atom_indices'])
            true_symbols = ast.literal_eval(row['coordinating_atom_symbols'])
            true_dent = row['denticities']

            candidates = find_candidate_donors(smiles)
            if not candidates:
                failures += 1
                continue

            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                failures += 1
                continue
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                failures += 1
                continue
            n_atoms = mol.GetNumAtoms()

            # Known-denticity greedy selection
            pred_indices, pred_scores = greedy_coord_selection(
                models, smiles, metal, charge, true_dent, candidates, device,
                method=method,
            )

            # Auto-denticity greedy selection
            auto_indices, auto_scores, auto_dent = auto_denticity_selection(
                models, smiles, metal, charge, candidates, device,
                method=method,
            )

            # Dative SMILES output
            dative_smiles = build_dative_smiles(smiles, metal, charge, pred_indices)

            # Metrics
            metrics = compute_metrics(true_indices, pred_indices, n_atoms)
            auto_metrics = compute_metrics(true_indices, auto_indices, n_atoms)

            pred_symbols = []
            for pi in pred_indices:
                if pi < n_atoms:
                    pred_symbols.append(mol.GetAtomWithIdx(pi).GetSymbol())

            metrics['smiles'] = smiles
            metrics['metal'] = metal
            metrics['true_indices'] = str(true_indices)
            metrics['pred_indices'] = str(pred_indices)
            metrics['true_symbols'] = str(true_symbols)
            metrics['pred_symbols'] = str(pred_symbols)
            metrics['pred_scores'] = str([round(s, 4) for s in pred_scores])
            metrics['dative_smiles'] = dative_smiles or ''
            metrics['n_candidates'] = len(candidates)
            metrics['auto_dent'] = auto_dent
            metrics['auto_molecular_acc'] = auto_metrics['molecular_acc']

            per_ligand.append(metrics)

            if (idx + 1) % 20 == 0:
                logger.info(f"  Processed {idx+1}/{len(df)} ligands")

        if failures > 0:
            logger.warning(f"  Failed to parse {failures} ligands")

        # Aggregate
        results_df = pd.DataFrame(per_ligand)
        n = len(results_df)

        total_tp = results_df['tp'].sum()
        total_fp = results_df['fp'].sum()
        total_tn = results_df['tn'].sum()
        total_fn = results_df['fn'].sum()
        total_atoms = total_tp + total_fp + total_tn + total_fn

        overall_acc = (total_tp + total_tn) / total_atoms if total_atoms > 0 else 0
        pos_acc = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        neg_acc = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
        balanced_acc = (pos_acc + neg_acc) / 2
        molecular_acc = results_df['molecular_acc'].mean()
        auto_mol_acc = results_df['auto_molecular_acc'].mean()

        summary = {
            'metal': f"{metal}+{charge}",
            'n_ligands': n,
            'n_failed': failures,
            'overall_accuracy': round(overall_acc * 100, 1),
            'positive_accuracy': round(pos_acc * 100, 1),
            'negative_accuracy': round(neg_acc * 100, 1),
            'balanced_accuracy': round(balanced_acc * 100, 1),
            'molecular_accuracy': round(molecular_acc * 100, 1),
            'auto_dent_molecular_accuracy': round(auto_mol_acc * 100, 1),
        }

        logger.info(f"\n  {method_name} results for {metal}+{charge} ({n} ligands):")
        logger.info(f"  Overall accuracy:     {summary['overall_accuracy']:.1f}%")
        logger.info(f"  Positive accuracy:    {summary['positive_accuracy']:.1f}% (donor recall)")
        logger.info(f"  Negative accuracy:    {summary['negative_accuracy']:.1f}%")
        logger.info(f"  Balanced accuracy:    {summary['balanced_accuracy']:.1f}%")
        logger.info(f"  Molecular accuracy:   {summary['molecular_accuracy']:.1f}% (all atoms correct)")
        logger.info(f"  Auto-dent mol. acc:   {summary['auto_dent_molecular_accuracy']:.1f}%")

        for dent in sorted(results_df['true_dent'].unique()):
            subset = results_df[results_df['true_dent'] == dent]
            mol_acc = subset['molecular_acc'].mean() * 100
            logger.info(f"    Denticity {dent}: mol_acc={mol_acc:.1f}% (n={len(subset)})")

        all_results[f"{metal}+{charge}"] = summary

        fname = method.replace(' ', '_')
        results_df.to_csv(
            os.path.join(output_dir, f'{fname}_{metal}{charge}_per_ligand.csv'),
            index=False,
        )

    return all_results


def run_evaluation(
    example_csv: str,
    checkpoint_dir: str,
    metals: List[str],
    charges: Dict[str, int],
    output_dir: str,
):
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    df = pd.read_csv(example_csv)
    logger.info(f"Loaded {len(df)} ligands from {example_csv}")
    logger.info(f"Denticity distribution: {df['denticities'].value_counts().sort_index().to_dict()}")

    logger.info(f"\nLoading ensemble from {checkpoint_dir}")
    models = load_ensemble(checkpoint_dir, config, device)
    logger.info(f"Ensemble size: {len(models)}")

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Run both methods
    # =========================================================================
    results_A = evaluate_method(
        "A: Coord Head (coordination signal)",
        "coord", df, metals, charges, models, device, output_dir,
    )

    results_B = evaluate_method(
        "B: Plausibility (logK1 signal)",
        "plausibility", df, metals, charges, models, device, output_dir,
    )

    # =========================================================================
    # Cross-metal consistency (using Method A)
    # =========================================================================
    if len(metals) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-METAL CONSISTENCY (Method A: Coord Head)")
        logger.info(f"{'='*60}")
        logger.info("Toney: metal-agnostic (same prediction for all metals)")
        logger.info("Ours: metal-aware (predictions may differ by metal)")

        consistency_data = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            true_indices = set(ast.literal_eval(row['coordinating_atom_indices']))
            true_dent = row['denticities']

            candidates = find_candidate_donors(smiles)
            if not candidates:
                continue

            metal_preds = {}
            for metal in metals:
                charge = charges.get(metal, 2)
                pred_idx, _ = greedy_coord_selection(
                    models, smiles, metal, charge, true_dent, candidates, device,
                    method='coord',
                )
                metal_preds[metal] = set(pred_idx)

            if len(metal_preds) >= 2:
                pred_sets = list(metal_preds.values())
                all_same = all(s == pred_sets[0] for s in pred_sets)
                n_correct = sum(1 for s in pred_sets if s == true_indices)

                consistency_data.append({
                    'smiles': smiles[:60],
                    'true_dent': true_dent,
                    'consistent': all_same,
                    'n_metals_correct': n_correct,
                    'n_metals_tested': len(metals),
                })

        if consistency_data:
            cons_df = pd.DataFrame(consistency_data)
            pct_consistent = cons_df['consistent'].mean() * 100
            pct_any_correct = (cons_df['n_metals_correct'] > 0).mean() * 100
            pct_all_correct = (cons_df['n_metals_correct'] == len(metals)).mean() * 100

            logger.info(f"  Identical predictions across all metals: {pct_consistent:.1f}%")
            logger.info(f"  At least one metal correct:              {pct_any_correct:.1f}%")
            logger.info(f"  All metals correct:                      {pct_all_correct:.1f}%")

            cons_df.to_csv(os.path.join(output_dir, 'cross_metal_consistency.csv'), index=False)

    # =========================================================================
    # Comparison table
    # =========================================================================
    toney_metrics = {
        'overall_accuracy': 98.8,
        'positive_accuracy': 94.6,
        'negative_accuracy': 99.3,
        'balanced_accuracy': 96.9,
        'molecular_accuracy': 84.8,
    }

    for method_label, results in [("A: Coord Head", results_A), ("B: Plausibility", results_B)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON: {method_label} vs Toney et al. (PNAS 2025)")
        logger.info(f"{'='*60}")

        metal_keys = list(results.keys())
        header = f"{'Metric':<25} {'Toney(70k)':>12}"
        for mk in metal_keys:
            header += f" {'Ours('+mk+')':>12}"
        logger.info(header)
        logger.info("-" * len(header))

        for metric_name, toney_val in toney_metrics.items():
            label = metric_name.replace('_', ' ').title()
            row_str = f"{label:<25} {toney_val:>11.1f}%"
            for mk in metal_keys:
                our_val = results[mk].get(metric_name, 0)
                row_str += f" {our_val:>11.1f}%"
            logger.info(row_str)

    logger.info(f"\n--- Training Data Comparison ---")
    logger.info(f"Toney: 70,069 CSD ligands with explicit coordination labels")
    logger.info(f"Ours:  19,964 stability constants — NO coordination labels")
    logger.info(f"Our model is metal-aware; Toney's is metal-agnostic")
    logger.info(f"Coordination knowledge emerges from learning binding thermodynamics")

    # Save summary
    summary_all = {
        'methods': {
            'A_coord_head': {
                'description': (
                    'Probe each candidate with a METAL_COORD edge, use coord_head '
                    'probability to rank. The coord head evaluates whether each atom '
                    'belongs in a coordination environment.'
                ),
                'results': results_A,
            },
            'B_plausibility': {
                'description': (
                    'Probe each candidate with a METAL_COORD edge, use predicted '
                    'logK1 as a chemical plausibility signal. Valid coordination '
                    'produces reasonable logK1; invalid coordination does not.'
                ),
                'results': results_B,
            },
        },
        'toney_reference': toney_metrics,
        'evaluation': {
            'dataset': example_csv,
            'n_ligands': len(df),
            'metals_tested': metals,
            'selection': 'greedy_sequential',
            'denticity_mode': 'known (true denticity for fair comparison)',
        },
        'training_comparison': {
            'toney': {'n_training': 70069, 'labels': 'explicit coordination', 'metal_aware': False},
            'ours': {'n_training': 19964, 'labels': 'stability constants only', 'metal_aware': True},
        },
    }

    with open(os.path.join(output_dir, 'coord_eval_summary.json'), 'w') as f:
        json.dump(summary_all, f, indent=2)
    logger.info(f"\nSaved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate coordination prediction via model probing'
    )
    parser.add_argument('--example_csv', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--metals', nargs='+', default=['Cu', 'Ni', 'Fe', 'Zn', 'Co'])
    args = parser.parse_args()

    config = get_config()

    if args.example_csv is None:
        args.example_csv = os.path.join(
            os.path.dirname(config.data_csv), 'example_ligands.csv'
        )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    if args.output_dir is None:
        args.output_dir = os.path.join(config.output_dir, 'coord_eval_binding')

    metal_charges = {
        'Cu': 2, 'Ni': 2, 'Fe': 2, 'Zn': 2, 'Co': 2,
        'Mn': 2, 'Cd': 2, 'Pb': 2, 'Mg': 2, 'Ca': 2,
        'Pd': 2, 'Pt': 2, 'Ag': 1, 'Hg': 2, 'Au': 3,
        'La': 3, 'Gd': 3, 'Eu': 3, 'Y': 3, 'Al': 3,
        'Ga': 3, 'In': 3, 'Cr': 3, 'Ti': 4, 'Zr': 4,
        'Be': 2, 'Tl': 1,
    }

    logger.info("=" * 60)
    logger.info("Coordination Site Prediction Evaluation")
    logger.info("=" * 60)
    logger.info(f"Ligands: {args.example_csv}")
    logger.info(f"Checkpoints: {args.checkpoint_dir}")
    logger.info(f"Metals: {args.metals}")
    logger.info(f"\nMethod A: Coord head probing (coordination signal)")
    logger.info(f"Method B: LogK1 probing (chemical plausibility)")

    run_evaluation(
        example_csv=args.example_csv,
        checkpoint_dir=args.checkpoint_dir,
        metals=args.metals,
        charges=metal_charges,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
