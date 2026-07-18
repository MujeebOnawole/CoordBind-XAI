"""
xai_coord.py - Perturbation-based XAI for coordination-only RGCN

Identifies which atom features drive donor predictions and systematic
FP/FN patterns. Uses occlusion (node masking) — same approach as the
binding XAI in stability_gnn/xai.py.

For each ligand in the Toney test set:
  1. Get baseline ensemble prediction (per-atom P(donor))
  2. For each atom, mask its features → re-predict → measure attribution
     Attribution = baseline_prob - masked_prob  (positive = atom drives donor pred)
  3. Analyze: which features/atom types cause FP/FN errors

This provides insight Toney's D-MPNN cannot: WHY specific atoms are
predicted (or missed) as donors.

Runs on HPC (needs model checkpoints + Toney test CSV + GPU).

Usage:
    python xai_coord.py
    python xai_coord.py --max_samples 500   # quick test
    python xai_coord.py --output_dir /path/to/output
"""

import os
import ast
import csv
import json
import glob
import re
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem

from config import get_config, TONEY_TEST_CSV, TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL
from model import RGCNCoordOnly
from loss import compute_coord_metrics
from build_data import construct_ligand_only_graph
from eval_toney import load_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ATTRIBUTION VIA NODE MASKING (OCCLUSION)
# =============================================================================

def compute_attributions(graph, model, device, threshold=0.5):
    """
    Compute per-atom attributions via node feature masking.

    For each atom i:
      - Mask atom i's features to zeros
      - Re-run model → get masked predictions for ALL atoms
      - Attribution_i = baseline_prob_i - masked_prob_i
        (positive = this atom's features push its own prediction toward donor)
      - Cross-attribution: how masking atom i affects OTHER atoms' predictions

    Returns:
        self_attr: (n_atoms,) — how each atom's features affect its OWN donor prob
        cross_attr: (n_atoms, n_atoms) — how masking atom i affects atom j
        baseline_probs: (n_atoms,) — unmasked ensemble probabilities
    """
    n_atoms = graph.x.shape[0]

    # Baseline prediction
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        baseline_logits = model(batch)
        baseline_probs = torch.sigmoid(baseline_logits).cpu().numpy()

    # Per-atom masking
    self_attr = np.zeros(n_atoms)
    cross_attr = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        masked_graph = graph.clone()
        masked_graph.x = graph.x.clone()
        masked_graph.x[i] = 0.0  # Zero out atom i's features

        masked_batch = Batch.from_data_list([masked_graph]).to(device)
        with torch.no_grad():
            masked_logits = model(masked_batch)
            masked_probs = torch.sigmoid(masked_logits).cpu().numpy()

        # Self-attribution: how much does masking atom i change its own prediction
        self_attr[i] = baseline_probs[i] - masked_probs[i]

        # Cross-attribution: how masking atom i affects all atoms
        cross_attr[i, :] = baseline_probs - masked_probs

    return self_attr, cross_attr, baseline_probs


def compute_ensemble_attributions(graph, models, device, threshold=0.5):
    """Compute attributions using the best model (first in ensemble) for XAI,
    but ensemble for baseline prediction."""
    n_atoms = graph.x.shape[0]

    # Ensemble baseline
    all_probs = []
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        for model in models:
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    baseline_probs = np.mean(all_probs, axis=0)

    # Use best model (first) for attributions (save compute)
    best_model = models[0]

    # Get best model baseline for consistent attribution
    with torch.no_grad():
        best_logits = best_model(batch)
        best_baseline = torch.sigmoid(best_logits).cpu().numpy()

    self_attr = np.zeros(n_atoms)

    for i in range(n_atoms):
        masked_graph = graph.clone()
        masked_graph.x = graph.x.clone()
        masked_graph.x[i] = 0.0

        masked_batch = Batch.from_data_list([masked_graph]).to(device)
        with torch.no_grad():
            masked_logits = best_model(masked_batch)
            masked_probs = torch.sigmoid(masked_logits).cpu().numpy()

        self_attr[i] = best_baseline[i] - masked_probs[i]

    return self_attr, baseline_probs


# =============================================================================
# MAIN XAI ANALYSIS
# =============================================================================

def run_xai(config, max_samples=None, threshold=0.5, toney_csv_override=None):
    """Run XAI analysis on Toney test set."""

    # Load Toney test CSV
    toney_csv = toney_csv_override or TONEY_TEST_CSV
    csv.field_size_limit(2**30)
    toney_df = pd.read_csv(toney_csv,
                           usecols=[TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL])
    logger.info(f"Toney test set: {len(toney_df)} ligands")

    if max_samples and max_samples < len(toney_df):
        toney_df = toney_df.sample(n=max_samples, random_state=42)
        logger.info(f"Subsampled to {len(toney_df)} ligands")

    # Load ensemble
    ckpt_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    sample_graphs = torch.load(graphs_path, map_location='cpu')
    node_feature_dim = sample_graphs[0].x.shape[1]
    del sample_graphs

    models = load_ensemble(ckpt_dir, config, hp, device, node_feature_dim)

    # Process each ligand
    atom_records = []
    ligand_records = []
    parse_failures = 0

    for idx, row in tqdm(toney_df.iterrows(), total=len(toney_df), desc="XAI"):
        smiles = str(row[TONEY_SMILES_COL]).strip()
        if not smiles or smiles == 'nan':
            parse_failures += 1
            continue

        try:
            catom_indices = ast.literal_eval(str(row[TONEY_CATOM_COL]))
            if isinstance(catom_indices, (int, float)):
                catom_indices = [int(catom_indices)]
            else:
                catom_indices = [int(x) for x in catom_indices]
        except (ValueError, SyntaxError):
            parse_failures += 1
            continue

        true_dent = int(row[TONEY_DENT_COL])

        try:
            graph = construct_ligand_only_graph(smiles, catom_indices)
        except Exception:
            parse_failures += 1
            continue

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            parse_failures += 1
            continue
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            parse_failures += 1
            continue

        # Compute attributions
        self_attr, baseline_probs = compute_ensemble_attributions(
            graph, models, device, threshold)

        true_labels = graph.coord_labels.numpy()
        pred_binary = (baseline_probs >= threshold).astype(int)

        # Classify each atom
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx >= len(baseline_probs):
                break

            atom = mol.GetAtomWithIdx(atom_idx)
            true_lab = int(true_labels[atom_idx])
            pred_lab = int(pred_binary[atom_idx])

            # Error type
            if pred_lab == 1 and true_lab == 1:
                error_type = 'TP'
            elif pred_lab == 1 and true_lab == 0:
                error_type = 'FP'
            elif pred_lab == 0 and true_lab == 0:
                error_type = 'TN'
            else:
                error_type = 'FN'

            # Neighbor context
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol()
                         for n in atom.GetNeighbors()]
            n_donor_neighbors = sum(1 for n in atom.GetNeighbors()
                                    if int(true_labels[n.GetIdx()]) == 1
                                    and n.GetIdx() < len(true_labels))

            atom_records.append({
                'smiles': smiles,
                'atom_idx': atom_idx,
                'symbol': atom.GetSymbol(),
                'hybridization': str(atom.GetHybridization()).split('.')[-1],
                'aromatic': atom.GetIsAromatic(),
                'in_ring': atom.IsInRing(),
                'degree': atom.GetDegree(),
                'formal_charge': atom.GetFormalCharge(),
                'n_Hs': atom.GetTotalNumHs(),
                'true_label': true_lab,
                'pred_label': pred_lab,
                'prob': float(baseline_probs[atom_idx]),
                'self_attribution': float(self_attr[atom_idx]),
                'error_type': error_type,
                'true_dent': true_dent,
                'neighbor_symbols': ','.join(sorted(neighbors)),
                'n_donor_neighbors': n_donor_neighbors,
            })

        # Ligand-level summary
        tp_idx = [i for i in range(len(pred_binary))
                  if pred_binary[i] == 1 and true_labels[i] == 1]
        fp_idx = [i for i in range(len(pred_binary))
                  if pred_binary[i] == 1 and true_labels[i] == 0]
        fn_idx = [i for i in range(len(pred_binary))
                  if pred_binary[i] == 0 and true_labels[i] == 1]

        ligand_records.append({
            'smiles': smiles,
            'true_dent': true_dent,
            'pred_dent': int(pred_binary.sum()),
            'n_atoms': mol.GetNumAtoms(),
            'mol_acc': 1.0 if np.array_equal(pred_binary, true_labels.astype(int)) else 0.0,
            'n_tp': len(tp_idx),
            'n_fp': len(fp_idx),
            'n_fn': len(fn_idx),
            'mean_tp_attr': float(np.mean([self_attr[i] for i in tp_idx])) if tp_idx else 0.0,
            'mean_fp_attr': float(np.mean([self_attr[i] for i in fp_idx])) if fp_idx else 0.0,
            'mean_fn_attr': float(np.mean([self_attr[i] for i in fn_idx])) if fn_idx else 0.0,
            'max_attr': float(np.max(self_attr)),
            'min_attr': float(np.min(self_attr)),
        })

    atom_df = pd.DataFrame(atom_records)
    ligand_df = pd.DataFrame(ligand_records)
    logger.info(f"\nProcessed: {len(ligand_df)} ligands, {len(atom_df)} atoms "
                f"({parse_failures} parse failures)")

    # ---- SAVE RAW DATA ----
    xai_dir = os.path.join(config.output_dir, 'xai_results')
    os.makedirs(xai_dir, exist_ok=True)

    atom_df.to_csv(os.path.join(xai_dir, 'xai_per_atom.csv'), index=False)
    ligand_df.to_csv(os.path.join(xai_dir, 'xai_per_ligand.csv'), index=False)

    # ---- ANALYSIS ----
    summary = analyze_xai_results(atom_df, ligand_df)

    with open(os.path.join(xai_dir, 'xai_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved XAI results to {xai_dir}/")
    return summary


# =============================================================================
# XAI ANALYSIS
# =============================================================================

def analyze_xai_results(atom_df, ligand_df):
    """Analyze attribution patterns to understand FP/FN causes."""
    summary = {}

    # ---- 1. ATTRIBUTION STATISTICS BY ERROR TYPE ----
    logger.info(f"\n{'='*60}")
    logger.info("ATTRIBUTION STATISTICS BY ERROR TYPE")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Type':>4} {'Count':>7} {'MeanProb':>9} {'MeanAttr':>9} "
                f"{'StdAttr':>8} {'MedianAttr':>10}")
    logger.info(f"  {'-'*52}")

    attr_by_error = {}
    for etype in ['TP', 'FP', 'TN', 'FN']:
        sub = atom_df[atom_df['error_type'] == etype]
        if len(sub) == 0:
            continue
        stats = {
            'count': len(sub),
            'mean_prob': float(sub['prob'].mean()),
            'mean_attr': float(sub['self_attribution'].mean()),
            'std_attr': float(sub['self_attribution'].std()),
            'median_attr': float(sub['self_attribution'].median()),
        }
        attr_by_error[etype] = stats
        logger.info(f"  {etype:>4} {stats['count']:>7} {stats['mean_prob']:>9.4f} "
                    f"{stats['mean_attr']:>9.4f} {stats['std_attr']:>8.4f} "
                    f"{stats['median_attr']:>10.4f}")

    summary['attribution_by_error_type'] = attr_by_error

    # Key insight: do FP atoms have high self-attribution (model confidently wrong)
    # or low (model uncertain but threshold pushes them over)?
    if 'FP' in attr_by_error and 'TP' in attr_by_error:
        fp_attr = attr_by_error['FP']['mean_attr']
        tp_attr = attr_by_error['TP']['mean_attr']
        logger.info(f"\n  FP vs TP attribution: FP={fp_attr:.4f}, TP={tp_attr:.4f}")
        if fp_attr < tp_attr * 0.5:
            logger.info(f"  → FP atoms have LOW self-attribution — model is uncertain, "
                        f"threshold is the issue")
        else:
            logger.info(f"  → FP atoms have HIGH self-attribution — model is confidently "
                        f"wrong about these atoms")

    # ---- 2. FP ANALYSIS: WHAT MAKES THE MODEL OVER-PREDICT? ----
    logger.info(f"\n{'='*60}")
    logger.info("FALSE POSITIVE DEEP DIVE")
    logger.info(f"{'='*60}")

    fp_df = atom_df[atom_df['error_type'] == 'FP']
    if len(fp_df) > 0:
        # By atom type
        logger.info(f"\n  FP by atom type:")
        fp_by_sym = fp_df.groupby('symbol').agg(
            count=('error_type', 'size'),
            mean_prob=('prob', 'mean'),
            mean_attr=('self_attribution', 'mean'),
        ).sort_values('count', ascending=False)

        total_fp = len(fp_df)
        fp_atom_breakdown = {}
        for sym, row in fp_by_sym.head(15).iterrows():
            pct = row['count'] / total_fp * 100
            logger.info(f"    {sym:>3}: {int(row['count']):>5} ({pct:>5.1f}%) "
                        f"prob={row['mean_prob']:.3f} attr={row['mean_attr']:.4f}")
            fp_atom_breakdown[sym] = {
                'count': int(row['count']), 'pct': float(pct),
                'mean_prob': float(row['mean_prob']),
                'mean_attr': float(row['mean_attr']),
            }
        summary['fp_by_atom_type'] = fp_atom_breakdown

        # By hybridization
        logger.info(f"\n  FP by atom type + hybridization:")
        fp_by_hyb = fp_df.groupby(['symbol', 'hybridization']).agg(
            count=('error_type', 'size'),
            mean_prob=('prob', 'mean'),
            mean_attr=('self_attribution', 'mean'),
        ).sort_values('count', ascending=False)

        fp_hyb_breakdown = {}
        for (sym, hyb), row in fp_by_hyb.head(15).iterrows():
            key = f"{sym}_{hyb}"
            pct = row['count'] / total_fp * 100
            logger.info(f"    {sym:>3}-{hyb:<6}: {int(row['count']):>5} ({pct:>5.1f}%) "
                        f"prob={row['mean_prob']:.3f} attr={row['mean_attr']:.4f}")
            fp_hyb_breakdown[key] = {
                'count': int(row['count']), 'pct': float(pct),
                'mean_prob': float(row['mean_prob']),
                'mean_attr': float(row['mean_attr']),
            }
        summary['fp_by_hybridization'] = fp_hyb_breakdown

        # FP proximity to true donors
        logger.info(f"\n  FP proximity to true donors:")
        fp_near_donor = fp_df['n_donor_neighbors'].value_counts().sort_index()
        for n_don, count in fp_near_donor.items():
            pct = count / total_fp * 100
            logger.info(f"    {n_don} donor neighbors: {count} ({pct:.1f}%)")

        summary['fp_donor_proximity'] = {
            int(k): int(v) for k, v in fp_near_donor.items()
        }

        # FP ring/aromatic context
        fp_aromatic = fp_df['aromatic'].sum()
        fp_ring = fp_df['in_ring'].sum()
        logger.info(f"\n  FP context:")
        logger.info(f"    Aromatic: {fp_aromatic} ({fp_aromatic/total_fp:.1%})")
        logger.info(f"    In ring:  {fp_ring} ({fp_ring/total_fp:.1%})")

    # ---- 3. FN ANALYSIS: WHAT MAKES THE MODEL MISS DONORS? ----
    logger.info(f"\n{'='*60}")
    logger.info("FALSE NEGATIVE DEEP DIVE")
    logger.info(f"{'='*60}")

    fn_df = atom_df[atom_df['error_type'] == 'FN']
    if len(fn_df) > 0:
        # By atom type
        logger.info(f"\n  FN by atom type:")
        fn_by_sym = fn_df.groupby('symbol').agg(
            count=('error_type', 'size'),
            mean_prob=('prob', 'mean'),
            mean_attr=('self_attribution', 'mean'),
        ).sort_values('count', ascending=False)

        total_fn = len(fn_df)
        fn_atom_breakdown = {}
        for sym, row in fn_by_sym.head(15).iterrows():
            pct = row['count'] / total_fn * 100
            logger.info(f"    {sym:>3}: {int(row['count']):>5} ({pct:>5.1f}%) "
                        f"prob={row['mean_prob']:.3f} attr={row['mean_attr']:.4f}")
            fn_atom_breakdown[sym] = {
                'count': int(row['count']), 'pct': float(pct),
                'mean_prob': float(row['mean_prob']),
                'mean_attr': float(row['mean_attr']),
            }
        summary['fn_by_atom_type'] = fn_atom_breakdown

        # By hybridization
        logger.info(f"\n  FN by atom type + hybridization:")
        fn_by_hyb = fn_df.groupby(['symbol', 'hybridization']).agg(
            count=('error_type', 'size'),
            mean_prob=('prob', 'mean'),
            mean_attr=('self_attribution', 'mean'),
        ).sort_values('count', ascending=False)

        fn_hyb_breakdown = {}
        for (sym, hyb), row in fn_by_hyb.head(15).iterrows():
            key = f"{sym}_{hyb}"
            pct = row['count'] / total_fn * 100
            logger.info(f"    {sym:>3}-{hyb:<6}: {int(row['count']):>5} ({pct:>5.1f}%) "
                        f"prob={row['mean_prob']:.3f} attr={row['mean_attr']:.4f}")
            fn_hyb_breakdown[key] = {
                'count': int(row['count']), 'pct': float(pct),
                'mean_prob': float(row['mean_prob']),
                'mean_attr': float(row['mean_attr']),
            }
        summary['fn_by_hybridization'] = fn_hyb_breakdown

    # ---- 4. ATTRIBUTION LANDSCAPE: TP vs FP separation ----
    logger.info(f"\n{'='*60}")
    logger.info("ATTRIBUTION LANDSCAPE (can FP/TP be separated?)")
    logger.info(f"{'='*60}")

    tp_df = atom_df[atom_df['error_type'] == 'TP']
    if len(tp_df) > 0 and len(fp_df) > 0:
        # Probability distributions
        tp_probs = tp_df['prob'].values
        fp_probs = fp_df['prob'].values

        logger.info(f"  TP probs: mean={np.mean(tp_probs):.4f}, "
                    f"median={np.median(tp_probs):.4f}, "
                    f"P10={np.percentile(tp_probs, 10):.4f}")
        logger.info(f"  FP probs: mean={np.mean(fp_probs):.4f}, "
                    f"median={np.median(fp_probs):.4f}, "
                    f"P90={np.percentile(fp_probs, 90):.4f}")

        # How many FPs have prob < 0.7 (recoverable with threshold tuning)?
        fp_below_07 = (fp_probs < 0.7).sum()
        fp_below_06 = (fp_probs < 0.6).sum()
        tp_above_06 = (tp_probs >= 0.6).sum()
        tp_above_07 = (tp_probs >= 0.7).sum()

        logger.info(f"\n  FP with prob < 0.6: {fp_below_06} ({fp_below_06/len(fp_probs):.1%})")
        logger.info(f"  FP with prob < 0.7: {fp_below_07} ({fp_below_07/len(fp_probs):.1%})")
        logger.info(f"  TP with prob >= 0.6: {tp_above_06} ({tp_above_06/len(tp_probs):.1%})")
        logger.info(f"  TP with prob >= 0.7: {tp_above_07} ({tp_above_07/len(tp_probs):.1%})")

        # Self-attribution distributions
        tp_attrs = tp_df['self_attribution'].values
        fp_attrs = fp_df['self_attribution'].values

        logger.info(f"\n  TP self-attr: mean={np.mean(tp_attrs):.4f}, "
                    f"median={np.median(tp_attrs):.4f}")
        logger.info(f"  FP self-attr: mean={np.mean(fp_attrs):.4f}, "
                    f"median={np.median(fp_attrs):.4f}")

        summary['tp_fp_separation'] = {
            'tp_mean_prob': float(np.mean(tp_probs)),
            'tp_median_prob': float(np.median(tp_probs)),
            'fp_mean_prob': float(np.mean(fp_probs)),
            'fp_median_prob': float(np.median(fp_probs)),
            'fp_below_0.6': int(fp_below_06),
            'fp_below_0.7': int(fp_below_07),
            'tp_above_0.6': int(tp_above_06),
            'tp_above_0.7': int(tp_above_07),
            'tp_mean_attr': float(np.mean(tp_attrs)),
            'fp_mean_attr': float(np.mean(fp_attrs)),
        }

    # ---- 5. LIGAND-LEVEL: DO FAILED MOLECULES HAVE DISTINCT PATTERNS? ----
    logger.info(f"\n{'='*60}")
    logger.info("FAILED vs CORRECT MOLECULE COMPARISON")
    logger.info(f"{'='*60}")

    correct = ligand_df[ligand_df['mol_acc'] == 1.0]
    failed = ligand_df[ligand_df['mol_acc'] == 0.0]

    if len(correct) > 0 and len(failed) > 0:
        logger.info(f"  Correct: {len(correct)} mols")
        logger.info(f"  Failed:  {len(failed)} mols")
        logger.info(f"\n  {'':>20} {'Correct':>10} {'Failed':>10}")
        logger.info(f"  {'-'*42}")

        for col in ['n_atoms', 'true_dent', 'max_attr', 'min_attr']:
            c_val = correct[col].mean()
            f_val = failed[col].mean()
            logger.info(f"  {col:>20} {c_val:>10.2f} {f_val:>10.2f}")

        logger.info(f"\n  Mean TP attribution:")
        logger.info(f"    Correct mols: {correct['mean_tp_attr'].mean():.4f}")
        logger.info(f"    Failed mols:  {failed['mean_tp_attr'].mean():.4f}")

        if 'mean_fp_attr' in failed.columns:
            fp_failed = failed[failed['n_fp'] > 0]
            if len(fp_failed) > 0:
                logger.info(f"  Mean FP attribution (failed mols with FP):")
                logger.info(f"    {fp_failed['mean_fp_attr'].mean():.4f}")

        summary['correct_vs_failed'] = {
            'n_correct': len(correct),
            'n_failed': len(failed),
            'correct_mean_atoms': float(correct['n_atoms'].mean()),
            'failed_mean_atoms': float(failed['n_atoms'].mean()),
            'correct_mean_dent': float(correct['true_dent'].mean()),
            'failed_mean_dent': float(failed['true_dent'].mean()),
        }

    # ---- 6. GLOBAL SUMMARY ----
    summary['n_ligands'] = len(ligand_df)
    summary['n_atoms'] = len(atom_df)
    summary['error_counts'] = {
        etype: int((atom_df['error_type'] == etype).sum())
        for etype in ['TP', 'FP', 'TN', 'FN']
    }

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Ligands: {len(ligand_df)}, Atoms: {len(atom_df)}")
    for etype in ['TP', 'FP', 'TN', 'FN']:
        logger.info(f"  {etype}: {summary['error_counts'][etype]}")

    return summary


# =============================================================================
# SLURM ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XAI for coordination-only RGCN')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root output dir (with checkpoints, graphs)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max ligands to process (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--toney_csv', type=str, default=None,
                        help='Path to Toney test CSV (overrides config default)')
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Coordination-Only RGCN - XAI Analysis")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now()}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Threshold: {args.threshold}")

    run_xai(config, args.max_samples, args.threshold,
            toney_csv_override=args.toney_csv)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
