"""
eval_toney.py - Evaluate coordination-only RGCN on Toney's 6,616 test set

Direct evaluation (no binding-based probing needed):
  - Build ligand-only graph from SMILES
  - Run through coord head -> P(donor) per atom
  - Compare to ground truth coordination labels

This is a clean apples-to-apples comparison with:
  - Toney's D-MPNN: 96.9% balanced acc, 94.6% donor recall
  - Run 3 binding-probing: ~76% balanced acc, ~56% donor recall
  - Run 4 binding-probing: ~78% balanced acc, ~60% donor recall

Usage:
    python eval_toney.py
    python eval_toney.py --output_dir /path/to/output
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
from torch_geometric.data import Batch
from datetime import datetime
from tqdm import tqdm

from config import get_config, TONEY_TEST_CSV, TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL
from model import RGCNCoordOnly
from loss import compute_coord_metrics
from build_data import construct_ligand_only_graph  # Local standalone version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble(checkpoints_dir, config, hp, device, node_feature_dim, top_k=5):
    """Load best-per-fold ensemble."""
    ckpt_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")

    all_ckpts = {}
    for f in ckpt_files:
        basename = os.path.basename(f)
        match = re.match(r'repeat(\d+)_fold(\d+)_best\.pt', basename)
        if not match:
            continue
        repeat_id = int(match.group(1))
        fold_id = int(match.group(2))
        ckpt = torch.load(f, map_location='cpu')
        val_score = ckpt.get('val_balanced_acc', 0.0)
        all_ckpts[(repeat_id, fold_id)] = (f, val_score, ckpt)

    # Best per repeat, then fill with best per uncovered fold
    repeats = sorted(set(r for r, f in all_ckpts.keys()))
    selected = {}
    used_folds = set()

    for rep in repeats:
        rep_ckpts = {k: v for k, v in all_ckpts.items() if k[0] == rep}
        best_key = max(rep_ckpts, key=lambda k: rep_ckpts[k][1])
        selected[best_key] = rep_ckpts[best_key]
        used_folds.add(best_key[1])

    all_folds = sorted(set(f for r, f in all_ckpts.keys()))
    for fold_id in [f for f in all_folds if f not in used_folds]:
        if len(selected) >= top_k:
            break
        fold_ckpts = {k: v for k, v in all_ckpts.items() if k[1] == fold_id}
        best_key = max(fold_ckpts, key=lambda k: fold_ckpts[k][1])
        selected[best_key] = fold_ckpts[best_key]

    top_ckpts = [selected[k] for k in sorted(selected.keys())]

    logger.info(f"Ensemble ({len(top_ckpts)} models):")
    for f, score, _ in top_ckpts:
        logger.info(f"  {os.path.basename(f)}: val_bal_acc = {score:.4f}")

    models = []
    for _, _, ckpt in top_ckpts:
        model = RGCNCoordOnly(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 128),
            num_conv_layers=hp.get('num_conv_layers', 3),
            num_edge_types=config.num_edge_types,
            dropout=0.0,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)

    return models


def evaluate_on_toney(config, top_k=5, toney_csv_override=None):
    """Evaluate coord-only ensemble on Toney's 6,616 test ligands."""

    # Load Toney test CSV
    toney_csv = toney_csv_override or TONEY_TEST_CSV
    if not os.path.exists(toney_csv):
        raise FileNotFoundError(f"Toney test CSV not found: {toney_csv}")

    csv.field_size_limit(2**30)
    usecols = [TONEY_SMILES_COL, TONEY_CATOM_COL, TONEY_DENT_COL]
    toney_df = pd.read_csv(toney_csv, usecols=usecols)
    logger.info(f"Toney test set: {len(toney_df)} ligands")

    # Load ensemble
    ckpt_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine node feature dim from a sample graph
    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    sample_graphs = torch.load(graphs_path)
    node_feature_dim = sample_graphs[0].x.shape[1]
    del sample_graphs

    models = load_ensemble(ckpt_dir, config, hp, device, node_feature_dim, top_k)

    # Evaluate each ligand
    results = []
    parse_failures = 0

    for idx, row in tqdm(toney_df.iterrows(), total=len(toney_df), desc="Evaluating"):
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

        # Ensemble prediction
        all_probs = []
        batch = Batch.from_data_list([graph]).to(device)

        with torch.no_grad():
            for model in models:
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        ensemble_probs = np.mean(all_probs, axis=0)
        true_labels = graph.coord_labels.numpy()

        # Per-atom metrics
        metrics = compute_coord_metrics(ensemble_probs, true_labels)

        # Molecular accuracy
        pred_binary = (ensemble_probs >= 0.5).astype(int)
        labels_int = true_labels.astype(int)
        mol_acc = 1.0 if np.array_equal(pred_binary, labels_int) else 0.0

        # Predicted denticity
        pred_dent = int(pred_binary.sum())
        dent_correct = 1.0 if pred_dent == true_dent else 0.0

        results.append({
            'smiles': smiles,
            'true_dent': true_dent,
            'pred_dent': pred_dent,
            'dent_correct': dent_correct,
            'molecular_acc': mol_acc,
            'balanced_acc': metrics['balanced_acc'],
            'donor_recall': metrics['donor_recall'],
            'neg_acc': metrics['neg_acc'],
            'overall_acc': metrics['overall_acc'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
        })

    logger.info(f"\nEvaluated: {len(results)} ligands ({parse_failures} parse failures)")

    # Aggregate metrics
    results_df = pd.DataFrame(results)

    # Micro-averaged metrics (sum tp/fp/tn/fn across all ligands)
    total_tp = results_df['tp'].sum()
    total_fp = results_df['fp'].sum()
    total_tn = results_df['tn'].sum()
    total_fn = results_df['fn'].sum()

    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro_neg_acc = total_tn / max(total_tn + total_fp, 1)
    micro_bal_acc = (micro_recall + micro_neg_acc) / 2
    micro_overall = (total_tp + total_tn) / max(total_tp + total_fp + total_tn + total_fn, 1)
    mol_acc = results_df['molecular_acc'].mean()
    dent_correct = results_df['dent_correct'].mean()

    summary = {
        'n_evaluated': len(results),
        'n_parse_failures': parse_failures,
        'ensemble_size': len(models),
        'micro_metrics': {
            'overall_acc': float(micro_overall),
            'donor_recall': float(micro_recall),
            'neg_acc': float(micro_neg_acc),
            'balanced_acc': float(micro_bal_acc),
        },
        'molecular_acc': float(mol_acc),
        'dent_correct': float(dent_correct),
        'comparison': {
            'toney_ref': {'balanced_acc': 0.969, 'donor_recall': 0.946, 'mol_acc': 0.848},
            'run3_cu': {'balanced_acc': 0.760, 'donor_recall': 0.563, 'mol_acc': 0.318},
            'run4_cu': {'balanced_acc': 0.777, 'donor_recall': 0.594, 'mol_acc': 0.349},
        },
    }

    # Per-denticity breakdown
    dent_breakdown = {}
    for d in sorted(results_df['true_dent'].unique()):
        subset = results_df[results_df['true_dent'] == d]
        d_tp = subset['tp'].sum()
        d_fp = subset['fp'].sum()
        d_tn = subset['tn'].sum()
        d_fn = subset['fn'].sum()
        d_recall = d_tp / max(d_tp + d_fn, 1)
        d_neg = d_tn / max(d_tn + d_fp, 1)
        dent_breakdown[int(d)] = {
            'count': len(subset),
            'donor_recall': float(d_recall),
            'neg_acc': float(d_neg),
            'balanced_acc': float((d_recall + d_neg) / 2),
            'mol_acc': float(subset['molecular_acc'].mean()),
        }
    summary['per_denticity'] = dent_breakdown

    # Save
    eval_dir = os.path.join(config.output_dir, 'toney_eval')
    os.makedirs(eval_dir, exist_ok=True)

    results_df.to_csv(os.path.join(eval_dir, 'toney_per_ligand.csv'), index=False)
    with open(os.path.join(eval_dir, 'toney_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print
    logger.info(f"\n{'='*60}")
    logger.info("COORDINATION-ONLY RGCN vs TONEY BENCHMARK")
    logger.info(f"{'='*60}")
    logger.info(f"  Balanced Acc:  {micro_bal_acc:.1%}  (Toney: 96.9%, Run4: 77.7%)")
    logger.info(f"  Donor Recall:  {micro_recall:.1%}  (Toney: 94.6%, Run4: 59.4%)")
    logger.info(f"  Neg Acc:       {micro_neg_acc:.1%}  (Toney: 99.3%, Run4: 96.1%)")
    logger.info(f"  Molecular Acc: {mol_acc:.1%}  (Toney: 84.8%, Run4: 34.9%)")
    logger.info(f"\nPer-denticity:")
    for d, m in dent_breakdown.items():
        logger.info(f"  dent={d}: n={m['count']}, bal_acc={m['balanced_acc']:.1%}, mol_acc={m['mol_acc']:.1%}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--toney_csv', type=str, default=None,
                        help='Path to Toney test CSV (overrides config default)')
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Coordination-Only RGCN - Toney Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now()}")

    evaluate_on_toney(config, args.top_k, toney_csv_override=args.toney_csv)
    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
