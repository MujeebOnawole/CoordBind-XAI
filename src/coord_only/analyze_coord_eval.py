"""
analyze_coord_eval.py - Extended evaluation metrics for coordination-only RGCN

Computes MCC, precision, F1, threshold sweep, and per-atom-type error analysis
from existing Toney eval results. Two modes:

  MODE 1 (--csv_only): Works from toney_per_ligand.csv alone (no model needed).
    Computes MCC, precision, F1, per-denticity breakdown from aggregate tp/fp/tn/fn.

  MODE 2 (full): Re-runs ensemble on Toney test set to get per-atom probabilities.
    Adds: threshold sweep, per-atom-type FP/FN rates, probability calibration,
    optimal threshold for MCC and mol_acc.

Usage:
    # Mode 1: From existing CSV (runs locally, no GPU)
    python analyze_coord_eval.py --csv_only --results_dir output/toney_eval

    # Mode 2: Full re-eval (needs model checkpoints + Toney test CSV)
    python analyze_coord_eval.py --results_dir output/toney_eval
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================

def compute_mcc(tp, fp, tn, fn):
    """Matthews Correlation Coefficient — balanced for class imbalance."""
    num = (tp * tn) - (fp * fn)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / denom if denom > 0 else 0.0


def compute_all_metrics(tp, fp, tn, fn):
    """Compute full classification metrics from confusion matrix."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # = donor recall
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    neg_acc = tn / max(tn + fp, 1)  # = specificity
    balanced_acc = (recall + neg_acc) / 2
    overall_acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    mcc = compute_mcc(tp, fp, tn, fn)
    npv = tn / max(tn + fn, 1)  # negative predictive value
    fpr = fp / max(fp + tn, 1)  # false positive rate
    fnr = fn / max(fn + tp, 1)  # false negative rate

    return {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'balanced_acc': float(balanced_acc),
        'overall_acc': float(overall_acc),
        'neg_acc': float(neg_acc),
        'npv': float(npv),
        'fpr': float(fpr),
        'fnr': float(fnr),
    }


# =============================================================================
# MODE 1: CSV-ONLY ANALYSIS
# =============================================================================

def analyze_from_csv(results_dir):
    """Compute extended metrics from existing toney_per_ligand.csv."""
    csv_path = os.path.join(results_dir, 'toney_per_ligand.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Per-ligand CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} ligands from {csv_path}")

    # --- Micro-averaged metrics (aggregate confusion matrix) ---
    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    total_tn = df['tn'].sum()
    total_fn = df['fn'].sum()

    micro = compute_all_metrics(total_tp, total_fp, total_tn, total_fn)

    logger.info(f"\n{'='*60}")
    logger.info("MICRO-AVERAGED METRICS (all atoms pooled)")
    logger.info(f"{'='*60}")
    logger.info(f"  MCC:          {micro['mcc']:.4f}")
    logger.info(f"  Precision:    {micro['precision']:.4f}")
    logger.info(f"  Recall:       {micro['recall']:.4f}")
    logger.info(f"  F1:           {micro['f1']:.4f}")
    logger.info(f"  Balanced Acc: {micro['balanced_acc']:.4f}")
    logger.info(f"  Overall Acc:  {micro['overall_acc']:.4f}")
    logger.info(f"  Neg Acc:      {micro['neg_acc']:.4f}")
    logger.info(f"  FPR:          {micro['fpr']:.4f}")
    logger.info(f"  FNR:          {micro['fnr']:.4f}")
    logger.info(f"  Confusion:    TP={total_tp}  FP={total_fp}  TN={total_tn}  FN={total_fn}")

    # --- Class balance ---
    total_pos = total_tp + total_fn
    total_neg = total_tn + total_fp
    total_all = total_pos + total_neg
    logger.info(f"\n  Class balance: {total_pos} donors ({total_pos/total_all:.1%}) "
                f"vs {total_neg} non-donors ({total_neg/total_all:.1%})")
    logger.info(f"  Imbalance ratio: 1:{total_neg/max(total_pos,1):.1f}")

    # --- Molecular-level metrics ---
    mol_acc = df['molecular_acc'].mean()
    dent_correct = df['dent_correct'].mean()
    logger.info(f"\n  Molecular Acc:  {mol_acc:.4f} ({mol_acc:.1%})")
    logger.info(f"  Denticity Acc:  {dent_correct:.4f} ({dent_correct:.1%})")

    # --- Error breakdown: FP vs FN contribution to mol_acc failures ---
    failed = df[df['molecular_acc'] == 0.0].copy()
    n_failed = len(failed)
    if n_failed > 0:
        fp_only = failed[(failed['fp'] > 0) & (failed['fn'] == 0)]
        fn_only = failed[(failed['fn'] > 0) & (failed['fp'] == 0)]
        both = failed[(failed['fp'] > 0) & (failed['fn'] > 0)]

        logger.info(f"\n{'='*60}")
        logger.info(f"MOLECULAR ACCURACY FAILURE ANALYSIS ({n_failed} failed ligands)")
        logger.info(f"{'='*60}")
        logger.info(f"  FP-only failures: {len(fp_only)} ({len(fp_only)/n_failed:.1%}) — over-predicted donors")
        logger.info(f"  FN-only failures: {len(fn_only)} ({len(fn_only)/n_failed:.1%}) — missed true donors")
        logger.info(f"  Both FP+FN:       {len(both)} ({len(both)/n_failed:.1%}) — mixed errors")

        # Average FP/FN counts in failed molecules
        logger.info(f"\n  Mean FP per failed mol: {failed['fp'].mean():.2f}")
        logger.info(f"  Mean FN per failed mol: {failed['fn'].mean():.2f}")

        # FP distribution
        fp_dist = failed['fp'].value_counts().sort_index()
        logger.info(f"\n  FP count distribution in failed mols:")
        for count, freq in fp_dist.head(8).items():
            logger.info(f"    {count} FP: {freq} mols")

        # FN distribution
        fn_dist = failed['fn'].value_counts().sort_index()
        logger.info(f"\n  FN count distribution in failed mols:")
        for count, freq in fn_dist.head(8).items():
            logger.info(f"    {count} FN: {freq} mols")

    # --- Denticity over/under prediction ---
    df['dent_error'] = df['pred_dent'] - df['true_dent']
    logger.info(f"\n{'='*60}")
    logger.info("DENTICITY PREDICTION ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"  Mean dent error (pred - true): {df['dent_error'].mean():.3f}")
    logger.info(f"  Std dent error:                {df['dent_error'].std():.3f}")

    over_pred = (df['dent_error'] > 0).sum()
    under_pred = (df['dent_error'] < 0).sum()
    exact = (df['dent_error'] == 0).sum()
    logger.info(f"  Over-predicted:  {over_pred} ({over_pred/len(df):.1%})")
    logger.info(f"  Under-predicted: {under_pred} ({under_pred/len(df):.1%})")
    logger.info(f"  Exact:           {exact} ({exact/len(df):.1%})")

    dent_error_dist = df['dent_error'].value_counts().sort_index()
    logger.info(f"\n  Denticity error distribution:")
    for err, count in dent_error_dist.items():
        logger.info(f"    {err:+d}: {count} ({count/len(df):.1%})")

    # --- Per-denticity extended metrics ---
    logger.info(f"\n{'='*60}")
    logger.info("PER-DENTICITY METRICS")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Dent':>4} {'n':>6} {'MCC':>7} {'Prec':>7} {'Recall':>7} "
                f"{'F1':>7} {'BalAcc':>7} {'MolAcc':>7} {'DentAcc':>8}")
    logger.info(f"  {'-'*63}")

    per_dent = {}
    for d in sorted(df['true_dent'].unique()):
        sub = df[df['true_dent'] == d]
        d_tp = sub['tp'].sum()
        d_fp = sub['fp'].sum()
        d_tn = sub['tn'].sum()
        d_fn = sub['fn'].sum()
        d_metrics = compute_all_metrics(d_tp, d_fp, d_tn, d_fn)
        d_metrics['count'] = len(sub)
        d_metrics['mol_acc'] = float(sub['molecular_acc'].mean())
        d_metrics['dent_correct'] = float(sub['dent_correct'].mean())
        per_dent[int(d)] = d_metrics

        logger.info(f"  {d:>4} {len(sub):>6} {d_metrics['mcc']:>7.4f} "
                    f"{d_metrics['precision']:>7.4f} {d_metrics['recall']:>7.4f} "
                    f"{d_metrics['f1']:>7.4f} {d_metrics['balanced_acc']:>7.4f} "
                    f"{d_metrics['mol_acc']:>7.4f} {d_metrics['dent_correct']:>8.4f}")

    # --- Comparison table ---
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH PUBLISHED WORK")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Metric':<18} {'Toney D-MPNN':>14} {'RGCN coord_only':>16} {'Delta':>8}")
    logger.info(f"  {'-'*58}")

    toney_ref = {
        'balanced_acc': 0.969, 'recall': 0.946, 'neg_acc': 0.993,
        'mol_acc': 0.848, 'overall_acc': 0.988,
    }
    for metric, toney_val in toney_ref.items():
        our_val = micro.get(metric, mol_acc if metric == 'mol_acc' else None)
        if our_val is None:
            continue
        delta = our_val - toney_val
        marker = '+' if delta > 0 else ''
        logger.info(f"  {metric:<18} {toney_val:>14.3f} {our_val:>16.4f} {marker}{delta:>7.3f}")

    logger.info(f"  {'mcc':<18} {'N/A':>14} {micro['mcc']:>16.4f} {'':>8}")
    logger.info(f"  {'precision':<18} {'N/A':>14} {micro['precision']:>16.4f} {'':>8}")
    logger.info(f"  {'f1':<18} {'N/A':>14} {micro['f1']:>16.4f} {'':>8}")

    # --- Save extended results ---
    output = {
        'micro_metrics': micro,
        'molecular_acc': float(mol_acc),
        'dent_correct': float(dent_correct),
        'n_evaluated': len(df),
        'class_balance': {
            'n_donors': int(total_pos),
            'n_non_donors': int(total_neg),
            'donor_fraction': float(total_pos / total_all),
            'imbalance_ratio': float(total_neg / max(total_pos, 1)),
        },
        'failure_analysis': {
            'n_failed': int(n_failed),
            'fp_only_failures': int(len(fp_only)) if n_failed > 0 else 0,
            'fn_only_failures': int(len(fn_only)) if n_failed > 0 else 0,
            'both_fp_fn_failures': int(len(both)) if n_failed > 0 else 0,
            'mean_fp_per_failed': float(failed['fp'].mean()) if n_failed > 0 else 0,
            'mean_fn_per_failed': float(failed['fn'].mean()) if n_failed > 0 else 0,
        },
        'denticity_analysis': {
            'mean_error': float(df['dent_error'].mean()),
            'std_error': float(df['dent_error'].std()),
            'over_predicted_frac': float(over_pred / len(df)),
            'under_predicted_frac': float(under_pred / len(df)),
            'exact_frac': float(exact / len(df)),
        },
        'per_denticity': per_dent,
    }

    out_path = os.path.join(results_dir, 'extended_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved extended metrics to {out_path}")

    return output


# =============================================================================
# MODE 2: FULL RE-EVAL WITH THRESHOLD SWEEP + ATOM-TYPE ANALYSIS
# =============================================================================

def full_analysis(results_dir, output_dir_root, toney_csv_override=None):
    """Re-run ensemble to get per-atom probabilities for threshold sweep and atom-type analysis."""
    import ast
    import csv
    import torch
    from torch_geometric.data import Batch
    from tqdm import tqdm
    from rdkit import Chem

    # Import coord_only modules
    from config import get_config, TONEY_TEST_CSV, TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL
    from model import RGCNCoordOnly
    from build_data import construct_ligand_only_graph
    from eval_toney import load_ensemble

    config = get_config()
    if output_dir_root:
        config.output_dir = output_dir_root

    # Load Toney test CSV
    toney_csv = toney_csv_override or TONEY_TEST_CSV
    csv.field_size_limit(2**30)
    toney_df = pd.read_csv(toney_csv,
                           usecols=[TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL])
    logger.info(f"Toney test set: {len(toney_df)} ligands")

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

    # Collect per-atom predictions
    all_atom_records = []  # (smiles, atom_idx, atom_symbol, true_label, prob, true_dent)
    ligand_records = []

    for idx, row in tqdm(toney_df.iterrows(), total=len(toney_df), desc="Re-evaluating"):
        smiles = str(row[TONEY_SMILES_COL]).strip()
        if not smiles or smiles == 'nan':
            continue

        try:
            catom_indices = ast.literal_eval(str(row[TONEY_CATOM_COL]))
            if isinstance(catom_indices, (int, float)):
                catom_indices = [int(catom_indices)]
            else:
                catom_indices = [int(x) for x in catom_indices]
        except (ValueError, SyntaxError):
            continue

        true_dent = int(row[TONEY_DENT_COL])

        try:
            graph = construct_ligand_only_graph(smiles, catom_indices)
        except Exception:
            continue

        # Parse molecule for atom symbols
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            continue

        # Ensemble prediction
        batch = Batch.from_data_list([graph]).to(device)
        all_probs = []
        with torch.no_grad():
            for model in models:
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        ensemble_probs = np.mean(all_probs, axis=0)
        true_labels = graph.coord_labels.numpy()

        # Per-atom records
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx >= len(ensemble_probs):
                break
            atom = mol.GetAtomWithIdx(atom_idx)
            symbol = atom.GetSymbol()
            hyb = str(atom.GetHybridization()).split('.')[-1]
            aromatic = atom.GetIsAromatic()
            in_ring = atom.IsInRing()
            degree = atom.GetDegree()

            all_atom_records.append({
                'smiles': smiles,
                'atom_idx': atom_idx,
                'symbol': symbol,
                'hybridization': hyb,
                'aromatic': aromatic,
                'in_ring': in_ring,
                'degree': degree,
                'true_label': int(true_labels[atom_idx]),
                'prob': float(ensemble_probs[atom_idx]),
                'true_dent': true_dent,
            })

        # Ligand-level record with per-atom probs for threshold sweep
        ligand_records.append({
            'smiles': smiles,
            'true_dent': true_dent,
            'true_labels': true_labels.tolist(),
            'probs': ensemble_probs.tolist(),
        })

    atom_df = pd.DataFrame(all_atom_records)
    logger.info(f"Collected {len(atom_df)} atom records from {len(ligand_records)} ligands")

    # ---- THRESHOLD SWEEP ----
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD SWEEP")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Thresh':>7} {'MCC':>7} {'Prec':>7} {'Recall':>7} "
                f"{'F1':>7} {'BalAcc':>7} {'MolAcc':>7}")
    logger.info(f"  {'-'*50}")

    thresholds = np.arange(0.30, 0.76, 0.05)
    sweep_results = []

    for thresh in thresholds:
        pred_binary = (atom_df['prob'] >= thresh).astype(int).values
        true_binary = atom_df['true_label'].values

        tp = int(((pred_binary == 1) & (true_binary == 1)).sum())
        fp = int(((pred_binary == 1) & (true_binary == 0)).sum())
        tn = int(((pred_binary == 0) & (true_binary == 0)).sum())
        fn = int(((pred_binary == 0) & (true_binary == 1)).sum())

        m = compute_all_metrics(tp, fp, tn, fn)

        # Molecular accuracy at this threshold
        mol_correct = 0
        for rec in ligand_records:
            probs = np.array(rec['probs'])
            labels = np.array(rec['true_labels'])
            preds = (probs >= thresh).astype(int)
            if np.array_equal(preds, labels.astype(int)):
                mol_correct += 1
        m['mol_acc'] = mol_correct / max(len(ligand_records), 1)
        m['threshold'] = float(thresh)

        sweep_results.append(m)
        logger.info(f"  {thresh:>7.2f} {m['mcc']:>7.4f} {m['precision']:>7.4f} "
                    f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['balanced_acc']:>7.4f} "
                    f"{m['mol_acc']:>7.4f}")

    # Find optimal thresholds
    best_mcc_row = max(sweep_results, key=lambda x: x['mcc'])
    best_mol_row = max(sweep_results, key=lambda x: x['mol_acc'])
    best_f1_row = max(sweep_results, key=lambda x: x['f1'])

    logger.info(f"\n  Optimal for MCC:     thresh={best_mcc_row['threshold']:.2f} "
                f"(MCC={best_mcc_row['mcc']:.4f}, mol_acc={best_mcc_row['mol_acc']:.4f})")
    logger.info(f"  Optimal for mol_acc: thresh={best_mol_row['threshold']:.2f} "
                f"(MCC={best_mol_row['mcc']:.4f}, mol_acc={best_mol_row['mol_acc']:.4f})")
    logger.info(f"  Optimal for F1:      thresh={best_f1_row['threshold']:.2f} "
                f"(MCC={best_f1_row['mcc']:.4f}, mol_acc={best_f1_row['mol_acc']:.4f})")

    # ---- PER-ATOM-TYPE ERROR ANALYSIS ----
    logger.info(f"\n{'='*60}")
    logger.info("PER-ATOM-TYPE ERROR ANALYSIS (threshold=0.5)")
    logger.info(f"{'='*60}")

    atom_df['pred_0.5'] = (atom_df['prob'] >= 0.5).astype(int)
    atom_df['correct'] = (atom_df['pred_0.5'] == atom_df['true_label']).astype(int)
    atom_df['is_fp'] = ((atom_df['pred_0.5'] == 1) & (atom_df['true_label'] == 0)).astype(int)
    atom_df['is_fn'] = ((atom_df['pred_0.5'] == 0) & (atom_df['true_label'] == 1)).astype(int)

    logger.info(f"\n  {'Symbol':>8} {'Total':>7} {'Donors':>7} {'DonFrac':>8} "
                f"{'FPR':>7} {'FNR':>7} {'MCC':>7} {'MeanProb':>9}")
    logger.info(f"  {'-'*66}")

    per_atom_type = {}
    for sym in sorted(atom_df['symbol'].unique()):
        sub = atom_df[atom_df['symbol'] == sym]
        if len(sub) < 10:
            continue

        s_tp = int(((sub['pred_0.5'] == 1) & (sub['true_label'] == 1)).sum())
        s_fp = int(((sub['pred_0.5'] == 1) & (sub['true_label'] == 0)).sum())
        s_tn = int(((sub['pred_0.5'] == 0) & (sub['true_label'] == 0)).sum())
        s_fn = int(((sub['pred_0.5'] == 0) & (sub['true_label'] == 1)).sum())

        s_metrics = compute_all_metrics(s_tp, s_fp, s_tn, s_fn)
        n_donors = int(sub['true_label'].sum())
        donor_frac = n_donors / len(sub)
        mean_prob = sub['prob'].mean()

        per_atom_type[sym] = {
            **s_metrics,
            'count': len(sub),
            'n_donors': n_donors,
            'donor_frac': float(donor_frac),
            'mean_prob': float(mean_prob),
        }

        logger.info(f"  {sym:>8} {len(sub):>7} {n_donors:>7} {donor_frac:>8.3f} "
                    f"{s_metrics['fpr']:>7.4f} {s_metrics['fnr']:>7.4f} "
                    f"{s_metrics['mcc']:>7.4f} {mean_prob:>9.4f}")

    # ---- FP HOTSPOTS: which atom types generate the most false positives ----
    logger.info(f"\n{'='*60}")
    logger.info("FALSE POSITIVE HOTSPOTS (atoms wrongly predicted as donors)")
    logger.info(f"{'='*60}")

    fp_atoms = atom_df[atom_df['is_fp'] == 1]
    if len(fp_atoms) > 0:
        fp_by_type = fp_atoms.groupby('symbol').agg(
            count=('is_fp', 'sum'),
            mean_prob=('prob', 'mean'),
        ).sort_values('count', ascending=False)

        total_fp = fp_by_type['count'].sum()
        logger.info(f"  Total FP atoms: {total_fp}")
        for sym, row in fp_by_type.head(15).iterrows():
            pct = row['count'] / total_fp * 100
            logger.info(f"  {sym:>4}: {int(row['count']):>5} FP ({pct:>5.1f}%) "
                        f"mean_prob={row['mean_prob']:.3f}")

        # FP by hybridization
        logger.info(f"\n  FP by hybridization:")
        fp_by_hyb = fp_atoms.groupby(['symbol', 'hybridization']).size().sort_values(ascending=False)
        for (sym, hyb), count in fp_by_hyb.head(15).items():
            logger.info(f"    {sym}-{hyb}: {count}")

        # FP by aromatic/ring context
        fp_aromatic = fp_atoms['aromatic'].sum()
        fp_ring = fp_atoms['in_ring'].sum()
        logger.info(f"\n  FP in aromatic: {fp_aromatic} ({fp_aromatic/len(fp_atoms):.1%})")
        logger.info(f"  FP in ring:     {fp_ring} ({fp_ring/len(fp_atoms):.1%})")

    # ---- FN HOTSPOTS: which atom types are missed as donors ----
    logger.info(f"\n{'='*60}")
    logger.info("FALSE NEGATIVE HOTSPOTS (true donors missed by model)")
    logger.info(f"{'='*60}")

    fn_atoms = atom_df[atom_df['is_fn'] == 1]
    if len(fn_atoms) > 0:
        fn_by_type = fn_atoms.groupby('symbol').agg(
            count=('is_fn', 'sum'),
            mean_prob=('prob', 'mean'),
        ).sort_values('count', ascending=False)

        total_fn = fn_by_type['count'].sum()
        logger.info(f"  Total FN atoms: {total_fn}")
        for sym, row in fn_by_type.head(15).iterrows():
            pct = row['count'] / total_fn * 100
            logger.info(f"  {sym:>4}: {int(row['count']):>5} FN ({pct:>5.1f}%) "
                        f"mean_prob={row['mean_prob']:.3f}")

        # FN by hybridization
        logger.info(f"\n  FN by hybridization:")
        fn_by_hyb = fn_atoms.groupby(['symbol', 'hybridization']).size().sort_values(ascending=False)
        for (sym, hyb), count in fn_by_hyb.head(15).items():
            logger.info(f"    {sym}-{hyb}: {count}")

    # ---- PROBABILITY CALIBRATION ----
    logger.info(f"\n{'='*60}")
    logger.info("PROBABILITY CALIBRATION (binned)")
    logger.info(f"{'='*60}")

    bins = np.arange(0.0, 1.05, 0.1)
    atom_df['prob_bin'] = pd.cut(atom_df['prob'], bins=bins, include_lowest=True)
    calib = atom_df.groupby('prob_bin', observed=True).agg(
        count=('true_label', 'size'),
        actual_pos_rate=('true_label', 'mean'),
        mean_pred=('prob', 'mean'),
    )
    logger.info(f"  {'Bin':>14} {'Count':>7} {'ActualRate':>11} {'MeanPred':>9}")
    for bin_label, row in calib.iterrows():
        logger.info(f"  {str(bin_label):>14} {int(row['count']):>7} "
                    f"{row['actual_pos_rate']:>11.4f} {row['mean_pred']:>9.4f}")

    # ---- SAVE EVERYTHING ----
    full_output = {
        'threshold_sweep': sweep_results,
        'optimal_thresholds': {
            'best_mcc': {'threshold': best_mcc_row['threshold'], 'mcc': best_mcc_row['mcc'],
                         'mol_acc': best_mcc_row['mol_acc'], 'f1': best_mcc_row['f1']},
            'best_mol_acc': {'threshold': best_mol_row['threshold'], 'mcc': best_mol_row['mcc'],
                             'mol_acc': best_mol_row['mol_acc'], 'f1': best_mol_row['f1']},
            'best_f1': {'threshold': best_f1_row['threshold'], 'mcc': best_f1_row['mcc'],
                        'mol_acc': best_f1_row['mol_acc'], 'f1': best_f1_row['f1']},
        },
        'per_atom_type': per_atom_type,
        'calibration': {str(k): {'count': int(v['count']),
                                  'actual_pos_rate': float(v['actual_pos_rate']),
                                  'mean_pred': float(v['mean_pred'])}
                        for k, v in calib.iterrows()},
    }

    out_path = os.path.join(results_dir, 'extended_analysis_full.json')
    with open(out_path, 'w') as f:
        json.dump(full_output, f, indent=2)

    # Save atom-level CSV for further analysis
    atom_csv_path = os.path.join(results_dir, 'toney_per_atom.csv')
    atom_df.to_csv(atom_csv_path, index=False)
    logger.info(f"\nSaved full analysis to {out_path}")
    logger.info(f"Saved per-atom CSV ({len(atom_df)} atoms) to {atom_csv_path}")

    return full_output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extended coord eval metrics')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing toney_per_ligand.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root output dir with checkpoints (for full mode)')
    parser.add_argument('--csv_only', action='store_true',
                        help='Only analyze existing CSV (no model needed)')
    parser.add_argument('--toney_csv', type=str, default=None,
                        help='Path to Toney test CSV (overrides config default)')
    args = parser.parse_args()

    # Resolve results_dir
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(script_dir, 'output', 'toney_eval')

    logger.info("=" * 60)
    logger.info("Coordination-Only RGCN - Extended Evaluation")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now()}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Mode: {'CSV-only' if args.csv_only else 'Full (model re-eval)'}")

    # Mode 1: always run CSV analysis
    csv_results = analyze_from_csv(args.results_dir)

    # Mode 2: full re-eval if requested
    if not args.csv_only:
        full_analysis(args.results_dir, args.output_dir, toney_csv_override=args.toney_csv)

    logger.info(f"\nEnd: {datetime.now()}")


if __name__ == '__main__':
    main()
