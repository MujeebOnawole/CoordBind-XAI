"""
final_eval.py - Ensemble Evaluation on Test Set for logK1 Stability Constant

Best-per-fold ensemble selection (same strategy as tmGNN-XAI):
1. Pick single best checkpoint per repeat (all repeats represented)
2. Fill remaining slots with best-per-fold from uncovered folds

Reports MAE, RMSE, R2 on held-out test set.

Usage:
    python final_eval.py --top_k 5
    python final_eval.py --top_k 5 --output_dir /path/to/output
"""

import os
import re
import json
import argparse
import glob
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from config import get_config, LOGK1_STATS
from model import RGCNStability
from data_module import load_dataset, get_data_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble(checkpoints_dir, config, hp, device, node_feature_dim, top_k=5):
    """
    Load diverse ensemble: best-per-fold with all repeats represented.

    Selection strategy (identical to tmGNN-XAI):
    1. Best fold per repeat (guarantees all repeats)
    2. Fill remaining with best-per-fold from uncovered folds
    """
    ckpt_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    # Parse all checkpoints
    all_ckpts = {}
    for f in ckpt_files:
        basename = os.path.basename(f)
        match = re.match(r'repeat(\d+)_fold(\d+)_best\.pt', basename)
        if not match:
            continue
        repeat_id = int(match.group(1))
        fold_id = int(match.group(2))
        ckpt = torch.load(f, map_location='cpu')
        val_mae = ckpt.get('val_mae', float('inf'))
        all_ckpts[(repeat_id, fold_id)] = (f, val_mae, ckpt)

    if not all_ckpts:
        raise FileNotFoundError(f"No valid checkpoints in {checkpoints_dir}")

    # Step 1: Best fold per repeat
    repeats = sorted(set(r for r, f in all_ckpts.keys()))
    selected = {}
    used_folds = set()

    for rep in repeats:
        rep_ckpts = {k: v for k, v in all_ckpts.items() if k[0] == rep}
        best_key = min(rep_ckpts, key=lambda k: rep_ckpts[k][1])
        selected[best_key] = rep_ckpts[best_key]
        used_folds.add(best_key[1])

    # Step 2: Fill with uncovered folds
    all_folds = sorted(set(f for r, f in all_ckpts.keys()))
    for fold_id in [f for f in all_folds if f not in used_folds]:
        if len(selected) >= top_k:
            break
        fold_ckpts = {k: v for k, v in all_ckpts.items() if k[1] == fold_id}
        best_key = min(fold_ckpts, key=lambda k: fold_ckpts[k][1])
        selected[best_key] = fold_ckpts[best_key]

    top_checkpoints = [selected[k] for k in sorted(selected.keys())]

    # Log
    repeats_used = sorted(set(k[0] for k in selected.keys()))
    folds_used = sorted(set(k[1] for k in selected.keys()))
    logger.info(f"Ensemble ({len(top_checkpoints)} models):")
    logger.info(f"  Repeats: {repeats_used}, Folds: {folds_used}")
    for f, mae, _ in top_checkpoints:
        logger.info(f"  {os.path.basename(f)}: val_MAE = {mae:.4f}")

    # Load models
    models = []
    for _, _, ckpt in top_checkpoints:
        model = RGCNStability(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 128),
            num_conv_layers=hp.get('num_conv_layers', 3),
            num_edge_types=config.num_edge_types,
            dropout=hp.get('dropout', 0.2),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)

    return models


def ensemble_predict(models, data_loader, device):
    """Get ensemble predictions (mean + std of all models)."""
    all_preds = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            model_preds = []
            for model in models:
                preds = model(batch)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                model_preds.append(preds.cpu().numpy())

            # Stack: (n_models, batch_size, 1)
            stacked = np.stack(model_preds, axis=0)
            ensemble_mean = np.mean(stacked, axis=0)
            ensemble_std = np.std(stacked, axis=0)

            all_preds.append(ensemble_mean)
            all_stds.append(ensemble_std)

            targets = batch.y
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            all_targets.append(targets.cpu().numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_stds, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def evaluate_ensemble(config, top_k=5):
    """Evaluate ensemble on test set."""

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")
    checkpoints_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')

    # Load data
    graphs, meta_df = load_dataset(graphs_path, meta_path)
    _, _, test_graphs = get_data_splits(graphs, meta_df)

    node_feature_dim = graphs[0].x.shape[1]
    logger.info(f"Node feature dimension: {node_feature_dim}")

    if len(test_graphs) == 0:
        logger.warning("No test data found!")
        return None

    logger.info(f"Test set: {len(test_graphs)} samples")

    # Load hyperparameters
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ensemble
    models = load_ensemble(checkpoints_dir, config, hp, device, node_feature_dim, top_k)

    # Predict
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    preds, stds, targets = ensemble_predict(models, test_loader, device)

    # Flatten
    preds_flat = preds.flatten()
    stds_flat = stds.flatten()
    targets_flat = targets.flatten()

    # Metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
    r2 = r2_score(targets_flat, preds_flat)
    pearson_r = np.corrcoef(targets_flat, preds_flat)[0, 1]

    results = {
        'ensemble_size': len(models),
        'test_samples': len(test_graphs),
        'logK1': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'unit': 'log10',
        },
        'mean_uncertainty': float(np.mean(stds_flat)),
    }

    # Save results
    with open(os.path.join(config.output_dir, 'ensemble_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions with metadata
    test_mask = meta_df['group'] == 'test'
    test_meta = meta_df[test_mask].reset_index(drop=True)

    pred_df = pd.DataFrame({
        'logK1_true': targets_flat,
        'logK1_pred': preds_flat,
        'logK1_std': stds_flat,
        'logK1_error': preds_flat - targets_flat,
        'logK1_abs_error': np.abs(preds_flat - targets_flat),
    })

    # Add metadata columns if available
    for col in ['smiles', 'metal_type', 'donor_elements']:
        if col in test_meta.columns and len(test_meta) == len(pred_df):
            pred_df[col] = test_meta[col].values

    pred_df.to_csv(os.path.join(config.output_dir, 'test_predictions.csv'), index=False)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"logK1: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.3f}, Pearson r={pearson_r:.3f}")
    logger.info(f"Mean uncertainty (ensemble std): {np.mean(stds_flat):.4f}")
    logger.info(f"Ensemble size: {len(models)}")
    logger.info(f"Test samples: {len(test_graphs)}")

    # Per-metal breakdown
    if 'metal_type' in pred_df.columns:
        logger.info(f"\nPer-metal MAE:")
        metal_maes = pred_df.groupby('metal_type').apply(
            lambda g: mean_absolute_error(g['logK1_true'], g['logK1_pred'])
        ).sort_values()
        for metal, m_mae in metal_maes.items():
            n = (pred_df['metal_type'] == metal).sum()
            logger.info(f"  {metal}: MAE={m_mae:.4f} (n={n})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Ensemble evaluation for logK1 prediction')
    parser.add_argument('--top_k', type=int, default=5, help='Max ensemble size')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("logK1 Stability Constant - Ensemble Evaluation")
    logger.info("=" * 60)
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Start: {datetime.now()}")

    evaluate_ensemble(config, args.top_k)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
