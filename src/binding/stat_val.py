"""
stat_val.py - Statistical Validation (3x5 CV) for logK1 Stability Constant Prediction

Repeated stratified k-fold cross-validation with metal_type stratification.
Uses RepeatedStratifiedKFold to ensure all metal types are represented in
each fold (as much as possible).

Usage:
    python stat_val.py --n_repeats 3 --n_folds 5
    python stat_val.py --output_dir /path/to/output
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from torch_geometric.loader import DataLoader
from datetime import datetime
import logging

from config import get_config, LOGK1_STATS
from model import RGCNStability
from data_module import load_dataset
from loss import mse_loss, joint_loss, compute_mae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    weight_decay,
    max_epochs=100,
    patience=15,
    coord_weight=1.0,
):
    """
    Train model for one fold with joint logK1 + coordination loss.

    Returns:
        Tuple of (best_val_mae, best_state_dict)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss_total = 0
        train_loss_bind = 0
        train_loss_coord = 0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            targets = batch.y

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

            # Joint loss: binding + coordination
            coord_labels = batch.coord_labels if hasattr(batch, 'coord_labels') else None
            if coord_labels is not None:
                losses = joint_loss(outputs, targets, coord_labels, coord_weight=coord_weight)
                loss = losses['total']
                train_loss_bind += losses['binding'].item()
                train_loss_coord += losses['coordination'].item()
            else:
                loss = mse_loss(outputs, targets)
                train_loss_bind += loss.item()

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1

        train_loss_total /= max(n_batches, 1)
        train_loss_bind /= max(n_batches, 1)
        train_loss_coord /= max(n_batches, 1)

        # Validation (logK1 MAE is the primary metric for early stopping)
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                logk1_pred = outputs['logk1']
                targets = batch.y

                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                if logk1_pred.dim() == 1:
                    logk1_pred = logk1_pred.unsqueeze(1)

                val_preds.append(logk1_pred.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        val_mae = compute_mae(val_preds, val_targets)
        scheduler.step(val_mae)

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:3d} | Loss: {train_loss_total:.4f} "
                        f"(bind={train_loss_bind:.4f}, coord={train_loss_coord:.4f}) "
                        f"| Val MAE: {val_mae:.4f}")

        # Early stopping on logK1 MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    return best_val_mae, best_state


def run_statistical_validation(config, n_repeats=3, n_folds=5):
    """
    Run repeated k-fold cross-validation.

    Uses RepeatedStratifiedKFold with metal_type as stratification variable
    to ensure balanced metal representation across folds.
    """

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Graphs not found: {graphs_path}. Run build_data.py first.")

    # Load dataset
    graphs, meta_df = load_dataset(graphs_path, meta_path)
    logger.info(f"Loaded {len(graphs)} complexes")

    node_feature_dim = graphs[0].x.shape[1]
    logger.info(f"Node feature dimension: {node_feature_dim}")

    # Get train+val indices (exclude test)
    train_val_mask = meta_df['group'].isin(['training', 'validation'])
    train_val_indices = meta_df[train_val_mask].index.tolist()
    train_val_indices = [i for i in train_val_indices if i < len(graphs)]
    train_val_graphs = [graphs[i] for i in train_val_indices]

    # Get metal types for stratification
    # Toney coord-only entries have metal_type='coord_only' — treat as a category
    train_val_metals = meta_df.iloc[train_val_indices]['metal_type'].fillna('coord_only').values

    # Group rare metals for stratification
    from collections import Counter
    metal_counts = Counter(train_val_metals)
    rare_metals = {m for m, c in metal_counts.items() if c < n_folds}
    stratify_labels = np.array([
        m if m not in rare_metals else 'rare'
        for m in train_val_metals
    ])

    # Count sources
    sources = meta_df.iloc[train_val_indices].get('source', pd.Series(['binding'] * len(train_val_indices)))
    n_binding = (sources == 'binding').sum()
    n_toney = (sources == 'toney').sum()

    logger.info(f"Train+Val: {len(train_val_graphs)} graphs ({n_binding} binding + {n_toney} Toney coord-only)")
    logger.info(f"Metal types: {len(set(train_val_metals))}")
    logger.info(f"Rare metals (< {n_folds} samples): {len(rare_metals)}")

    # Load hyperparameters
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            hp = json.load(f)
        logger.info(f"Loaded hyperparameters from {hp_path}")
    else:
        hp = {
            'hidden_dim': 128,
            'num_conv_layers': 3,
            'dropout': 0.2,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'batch_size': 32,
        }
        logger.info("Using default hyperparameters")

    logger.info(f"Hyperparameters: {hp}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Results storage
    results = []
    checkpoints_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Repeated Stratified K-Fold CV
    try:
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
        fold_iterator = rskf.split(train_val_indices, stratify_labels)
        logger.info(f"Using RepeatedStratifiedKFold (stratified by metal_type)")
    except ValueError:
        # Fall back to RepeatedKFold if stratification fails
        logger.warning("Stratification failed (some classes too small). Falling back to RepeatedKFold.")
        rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
        fold_iterator = rkf.split(train_val_indices)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
        repeat_num = fold_idx // n_folds
        fold_num = fold_idx % n_folds

        logger.info(f"\n--- Repeat {repeat_num+1}/{n_repeats}, Fold {fold_num+1}/{n_folds} ---")

        train_graphs_fold = [train_val_graphs[i] for i in train_idx]
        val_graphs_fold = [train_val_graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs_fold, batch_size=hp.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_graphs_fold, batch_size=hp.get('batch_size', 32), shuffle=False)

        model = RGCNStability(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 128),
            num_conv_layers=hp.get('num_conv_layers', 3),
            num_edge_types=config.num_edge_types,
            dropout=hp.get('dropout', 0.2),
        )

        val_mae, best_state = train_fold(
            model, train_loader, val_loader, device,
            hp.get('learning_rate', 1e-3),
            hp.get('weight_decay', 1e-5),
            max_epochs=config.max_epochs,
            patience=config.patience,
            coord_weight=hp.get('coord_weight', 1.0),
        )

        logger.info(f"  logK1 MAE = {val_mae:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoints_dir, f"repeat{repeat_num}_fold{fold_num}_best.pt")
        torch.save({
            'model_state_dict': best_state,
            'val_mae': val_mae,
            'repeat': repeat_num,
            'fold': fold_num,
            'hyperparameters': hp,
            'node_feature_dim': node_feature_dim,
        }, ckpt_path)

        results.append({
            'repeat': repeat_num,
            'fold': fold_num,
            'logK1_mae': val_mae,
        })

    # =========================================================================
    # Summary statistics
    # =========================================================================
    results_df = pd.DataFrame(results)
    maes = results_df['logK1_mae'].values

    summary = {
        'n_repeats': n_repeats,
        'n_folds': n_folds,
        'total_folds': len(results),
        'node_feature_dim': node_feature_dim,
        'logK1_mae': {
            'mean': float(np.mean(maes)),
            'std': float(np.std(maes)),
            'min': float(np.min(maes)),
            'max': float(np.max(maes)),
            'unit': 'log10',
        },
        'hyperparameters': hp,
    }

    # Save results
    results_dir = os.path.join(config.output_dir, 'cv_results')
    os.makedirs(results_dir, exist_ok=True)

    results_df.to_csv(os.path.join(results_dir, 'cv_fold_results.csv'), index=False)

    with open(os.path.join(results_dir, 'cv_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(config.output_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"logK1 MAE: {np.mean(maes):.4f} +/- {np.std(maes):.4f} (log10 units)")
    logger.info(f"  Min: {np.min(maes):.4f}, Max: {np.max(maes):.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Statistical validation for logK1 prediction')
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("logK1 Stability Constant - Statistical Validation")
    logger.info("=" * 60)
    logger.info(f"CV: {args.n_repeats} repeats x {args.n_folds} folds")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Start: {datetime.now()}")

    run_statistical_validation(config, args.n_repeats, args.n_folds)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
