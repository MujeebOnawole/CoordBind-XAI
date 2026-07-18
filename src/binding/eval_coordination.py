"""
eval_coordination.py - Evaluate coordination prediction head on Toney et al. test ligands

Zero-shot evaluation: model trained on 19,964 stability constant complexes,
evaluated on 99 ligands from Toney et al. (PNAS 2025) with known coordinating
atoms from CSD crystal structures.

Since our model is metal-aware (unlike Toney's), we evaluate with multiple
metals to test whether coordination predictions are chemically consistent.

Metrics (matching Toney et al. Table 1):
- Overall accuracy: per-atom (donor vs non-donor)
- Positive accuracy: recall on true coordinating atoms
- Negative accuracy: recall on true non-coordinating atoms
- Balanced accuracy: (positive + negative) / 2
- Molecular accuracy: fraction of ligands with ALL atoms correctly classified

Usage:
    python eval_coordination.py --checkpoint_dir output/cv_results/checkpoints
    python eval_coordination.py --checkpoint_dir output/cv_results/checkpoints --metals Cu Ni Fe Zn
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
)
from build_data import (
    get_metal_node_features, get_ligand_atom_features,
    one_hot, is_coordination_site,
)
from model import RGCNStability

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Edge types (no METAL_COORD — we are predicting which atoms should get it)
EDGE_TYPES = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'METAL_COORD': 3}


def build_ligand_metal_graph(
    ligand_smiles: str,
    metal_symbol: str,
    metal_charge: int,
) -> Optional[Data]:
    """
    Build a graph from plain ligand SMILES + metal, WITHOUT dative bonds.

    The metal node is added but NOT connected to any ligand atoms.
    The coordination head will predict which atoms should be donors.

    Args:
        ligand_smiles: Plain ligand SMILES (no metal, no dative bonds)
        metal_symbol: Metal element symbol (e.g., 'Cu')
        metal_charge: Metal formal charge (e.g., 2)

    Returns:
        PyG Data object with metal as isolated node, or None on failure
    """
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        # Try relaxed sanitization
        mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                return None
        else:
            return None

    n_ligand_atoms = mol.GetNumAtoms()

    # Build ligand node features
    node_features = []
    for atom in mol.GetAtoms():
        features = get_ligand_atom_features(atom, metal_symbol)
        node_features.append(features)

    # Add metal node (last node)
    metal_idx = n_ligand_atoms
    metal_features = get_metal_node_features(
        metal_symbol,
        formal_charge=metal_charge,
        S=1,
        mnd=None,  # Unknown — we're predicting coordination
    )
    node_features.append(metal_features)

    # Build edges (ligand bonds only, no metal-ligand edges)
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

    if len(edge_src_dst) == 0:
        return None

    # Tensors
    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    edge_index = torch.tensor(edge_src_dst, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    num_edges = edge_type.shape[0]
    edge_attr = torch.zeros(num_edges, 4)
    for i_e in range(num_edges):
        et = edge_type[i_e].item()
        if 0 <= et < 4:
            edge_attr[i_e, et] = 1.0

    # Node roles: metal=1, all ligand atoms=0 (no donors known)
    n_total = n_ligand_atoms + 1
    node_roles = torch.zeros(n_total, dtype=torch.long)
    node_roles[metal_idx] = 1

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


def load_ensemble(checkpoint_dir: str, config, device) -> List[RGCNStability]:
    """Load all checkpoints from directory."""
    import glob
    import re

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    # Load hyperparameters from first checkpoint or best_hyperparameters.json
    hp_path = os.path.join(os.path.dirname(checkpoint_dir), '..', 'best_hyperparameters.json')
    if not os.path.exists(hp_path):
        hp_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'best_hyperparameters.json')

    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp = json.load(f)
    else:
        # Get from first checkpoint
        ckpt = torch.load(ckpt_files[0], map_location='cpu')
        hp = ckpt.get('hyperparameters', {})

    node_feature_dim = hp.get('node_feature_dim', config.node_feature_dim)

    # Sort by val_mae, take top 5
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


def predict_coordination_ensemble(
    models: List[RGCNStability],
    data: Data,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict donor atoms using ensemble of models.

    Returns:
        predicted_indices: array of predicted donor atom indices
        mean_probs: per-atom mean donor probability across ensemble
    """
    batch = Batch.from_data_list([data]).to(device)
    n_ligand = data.n_ligand_atoms

    all_probs = []
    with torch.no_grad():
        for model in models:
            probs = model.predict_coordination(batch)
            # Only take ligand atom probabilities (exclude metal node)
            ligand_probs = probs[:n_ligand].cpu().numpy()
            all_probs.append(ligand_probs)

    mean_probs = np.mean(all_probs, axis=0)
    predicted_indices = np.where(mean_probs >= threshold)[0]

    return predicted_indices, mean_probs


def compute_metrics(
    true_indices: List[int],
    pred_indices: np.ndarray,
    n_atoms: int,
) -> Dict[str, float]:
    """Compute per-atom and molecular metrics for one ligand."""
    true_set = set(true_indices)
    pred_set = set(pred_indices.tolist())

    # Per-atom binary labels
    true_labels = np.zeros(n_atoms)
    pred_labels = np.zeros(n_atoms)
    for idx in true_set:
        if idx < n_atoms:
            true_labels[idx] = 1
    for idx in pred_set:
        if idx < n_atoms:
            pred_labels[idx] = 1

    # True positives, false positives, etc.
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))

    # Metrics
    overall_acc = (tp + tn) / n_atoms if n_atoms > 0 else 0
    pos_acc = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall on donors
    neg_acc = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall on non-donors
    balanced_acc = (pos_acc + neg_acc) / 2
    molecular_acc = 1.0 if true_set == pred_set else 0.0

    # Denticity accuracy (predicted count == true count)
    dent_correct = 1.0 if len(pred_set) == len(true_set) else 0.0

    return {
        'overall_acc': overall_acc,
        'pos_acc': pos_acc,
        'neg_acc': neg_acc,
        'balanced_acc': balanced_acc,
        'molecular_acc': molecular_acc,
        'dent_correct': dent_correct,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'true_dent': len(true_set),
        'pred_dent': len(pred_set),
    }


def run_evaluation(
    example_csv: str,
    checkpoint_dir: str,
    metals: List[str],
    charges: Dict[str, int],
    output_dir: str,
    threshold: float = 0.5,
):
    """Run coordination prediction evaluation."""

    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load ligand data
    df = pd.read_csv(example_csv)
    logger.info(f"Loaded {len(df)} ligands from {example_csv}")
    logger.info(f"Denticity distribution: {df['denticities'].value_counts().sort_index().to_dict()}")

    # Load models
    logger.info(f"\nLoading ensemble from {checkpoint_dir}")
    models = load_ensemble(checkpoint_dir, config, device)
    logger.info(f"Ensemble size: {len(models)}")

    # Evaluate per metal
    all_results = {}

    for metal in metals:
        charge = charges.get(metal, 2)
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with {metal}+{charge}")
        logger.info(f"{'='*60}")

        per_ligand = []
        failures = 0

        for idx, row in df.iterrows():
            smiles = row['smiles']
            true_indices = ast.literal_eval(row['coordinating_atom_indices'])
            true_symbols = ast.literal_eval(row['coordinating_atom_symbols'])
            true_dent = row['denticities']

            # Build graph
            data = build_ligand_metal_graph(smiles, metal, charge)
            if data is None:
                failures += 1
                continue

            # Predict
            pred_indices, probs = predict_coordination_ensemble(
                models, data, device, threshold=threshold
            )

            # Metrics
            metrics = compute_metrics(true_indices, pred_indices, data.n_ligand_atoms)
            metrics['smiles'] = smiles
            metrics['metal'] = metal
            metrics['true_indices'] = str(true_indices)
            metrics['pred_indices'] = str(pred_indices.tolist())
            metrics['true_symbols'] = str(true_symbols)

            per_ligand.append(metrics)

        if failures > 0:
            logger.warning(f"  Failed to parse {failures} ligands")

        # Aggregate metrics
        results_df = pd.DataFrame(per_ligand)
        n = len(results_df)

        # Overall per-atom metrics (pooled across all ligands)
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
        dent_acc = results_df['dent_correct'].mean()

        summary = {
            'metal': f"{metal}+{charge}",
            'n_ligands': n,
            'n_failed': failures,
            'overall_accuracy': round(overall_acc * 100, 1),
            'positive_accuracy': round(pos_acc * 100, 1),
            'negative_accuracy': round(neg_acc * 100, 1),
            'balanced_accuracy': round(balanced_acc * 100, 1),
            'molecular_accuracy': round(molecular_acc * 100, 1),
            'denticity_accuracy': round(dent_acc * 100, 1),
        }

        logger.info(f"\n  Results for {metal}+{charge} ({n} ligands):")
        logger.info(f"  Overall accuracy:     {summary['overall_accuracy']:.1f}%")
        logger.info(f"  Positive accuracy:    {summary['positive_accuracy']:.1f}% (recall on donors)")
        logger.info(f"  Negative accuracy:    {summary['negative_accuracy']:.1f}% (recall on non-donors)")
        logger.info(f"  Balanced accuracy:    {summary['balanced_accuracy']:.1f}%")
        logger.info(f"  Molecular accuracy:   {summary['molecular_accuracy']:.1f}% (all atoms correct)")
        logger.info(f"  Denticity accuracy:   {summary['denticity_accuracy']:.1f}% (correct count)")

        # Per-denticity breakdown
        for dent in sorted(results_df['true_dent'].unique()):
            subset = results_df[results_df['true_dent'] == dent]
            mol_acc = subset['molecular_acc'].mean() * 100
            logger.info(f"    Denticity {dent}: mol_acc={mol_acc:.1f}% (n={len(subset)})")

        all_results[f"{metal}+{charge}"] = summary

        # Save per-ligand results
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(
            os.path.join(output_dir, f'coord_eval_{metal}{charge}_per_ligand.csv'),
            index=False,
        )

    # =========================================================================
    # Cross-metal consistency analysis
    # =========================================================================
    if len(metals) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-METAL CONSISTENCY")
        logger.info(f"{'='*60}")
        logger.info("(Toney's model is metal-agnostic — same prediction for all metals)")
        logger.info("(Our model is metal-aware — predictions may vary by metal)")

        # For each ligand, check if predicted donors change across metals
        # This is a unique analysis that Toney cannot do
        consistency_data = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            true_indices = set(ast.literal_eval(row['coordinating_atom_indices']))
            metal_preds = {}

            for metal in metals:
                charge = charges.get(metal, 2)
                data = build_ligand_metal_graph(smiles, metal, charge)
                if data is None:
                    continue
                pred_idx, _ = predict_coordination_ensemble(
                    models, data, device, threshold=threshold
                )
                metal_preds[metal] = set(pred_idx.tolist())

            if len(metal_preds) >= 2:
                # Check if predictions are identical across metals
                pred_sets = list(metal_preds.values())
                all_same = all(s == pred_sets[0] for s in pred_sets)
                consistency_data.append({
                    'smiles': smiles[:50],
                    'true_dent': len(true_indices),
                    'consistent': all_same,
                    'n_unique_predictions': len(set(frozenset(s) for s in pred_sets)),
                })

        if consistency_data:
            cons_df = pd.DataFrame(consistency_data)
            pct_consistent = cons_df['consistent'].mean() * 100
            logger.info(f"  Ligands with identical predictions across all metals: {pct_consistent:.1f}%")
            logger.info(f"  Ligands with metal-dependent coordination: {100-pct_consistent:.1f}%")

    # =========================================================================
    # Comparison table (our results vs Toney et al.)
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH TONEY ET AL. (PNAS 2025)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<25} {'Toney (70k train)':>18} ", end="")
    for metal_key in all_results:
        logger.info(f"{'Ours (' + metal_key + ')':>18} ", end="")
    logger.info("")
    logger.info("-" * (25 + 18 + 18 * len(all_results)))

    toney_metrics = {
        'overall_accuracy': 98.8,
        'positive_accuracy': 94.6,
        'negative_accuracy': 99.3,
        'balanced_accuracy': 96.9,
        'molecular_accuracy': 84.8,
    }

    for metric_name, toney_val in toney_metrics.items():
        label = metric_name.replace('_', ' ').title()
        logger.info(f"{label:<25} {toney_val:>17.1f}%", end="")
        for metal_key, summary in all_results.items():
            our_val = summary[metric_name]
            logger.info(f" {our_val:>17.1f}%", end="")
        logger.info("")

    logger.info(f"\nNote: Toney trained on 70,069 CSD ligands (coordination labels only).")
    logger.info(f"We trained on {config.data_csv.split('/')[-1]} (stability constants + coordination).")
    logger.info(f"Our coordination labels are learned as a byproduct of binding prediction.")

    # Save summary
    summary_all = {
        'toney_reference': toney_metrics,
        'our_results': all_results,
        'evaluation': {
            'dataset': example_csv,
            'n_ligands': len(df),
            'threshold': threshold,
            'metals_tested': metals,
        },
    }

    with open(os.path.join(output_dir, 'coord_eval_summary.json'), 'w') as f:
        json.dump(summary_all, f, indent=2)
    logger.info(f"\nSaved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate coordination prediction on Toney et al. ligands'
    )
    parser.add_argument('--example_csv', type=str, default=None,
                        help='Path to example_ligands.csv from Toney et al.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with model checkpoints')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for evaluation results')
    parser.add_argument('--metals', nargs='+', default=['Cu', 'Ni', 'Fe', 'Zn', 'Co'],
                        help='Metals to evaluate (default: Cu Ni Fe Zn Co)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for donor classification')
    args = parser.parse_args()

    config = get_config()

    # Defaults
    if args.example_csv is None:
        args.example_csv = os.path.join(
            os.path.dirname(config.data_csv), 'example_ligands.csv'
        )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    if args.output_dir is None:
        args.output_dir = os.path.join(config.output_dir, 'coord_eval')

    # Common oxidation states
    metal_charges = {
        'Cu': 2, 'Ni': 2, 'Fe': 2, 'Zn': 2, 'Co': 2,
        'Mn': 2, 'Cd': 2, 'Pb': 2, 'Mg': 2, 'Ca': 2,
        'Pd': 2, 'Pt': 2, 'Ag': 1, 'Hg': 2, 'Au': 3,
        'La': 3, 'Gd': 3, 'Eu': 3, 'Y': 3, 'Al': 3,
        'Ga': 3, 'In': 3, 'Cr': 3, 'Ti': 4, 'Zr': 4,
        'Be': 2, 'Tl': 1,
    }

    logger.info("=" * 60)
    logger.info("Coordination Prediction Evaluation")
    logger.info("=" * 60)
    logger.info(f"Ligands: {args.example_csv}")
    logger.info(f"Checkpoints: {args.checkpoint_dir}")
    logger.info(f"Metals: {args.metals}")
    logger.info(f"Threshold: {args.threshold}")

    run_evaluation(
        example_csv=args.example_csv,
        checkpoint_dir=args.checkpoint_dir,
        metals=args.metals,
        charges=metal_charges,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
