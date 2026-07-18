"""
stat_val.py - Statistical validation (3x5 CV) for coordination-only RGCN

Trains coordination-only model on 20K binding ligands.
Primary metric: balanced accuracy on per-atom donor prediction.

Usage:
    python stat_val.py --n_repeats 3 --n_folds 5
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from torch_geometric.loader import DataLoader
from datetime import datetime
import logging

from config import get_config
from model import RGCNCoordOnly
from loss import coord_loss, compute_coord_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_fold(model, train_loader, val_loader, device, lr, wd, max_epochs=100, patience=15):
    """Train one fold. Returns (best_bal_acc, best_state_dict)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_bal_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = coord_loss(logits, batch.coord_labels)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(batch.coord_labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = compute_coord_metrics(preds, labels)
        bal_acc = metrics['balanced_acc']

        scheduler.step(bal_acc)

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:3d} | Loss: {total_loss/max(n_batches,1):.4f} | "
                        f"Bal.Acc: {bal_acc:.4f} | Recall: {metrics['donor_recall']:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    return best_bal_acc, best_state


def run_statistical_validation(config, n_repeats=3, n_folds=5):
    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    graphs = torch.load(graphs_path)
    meta_df = pd.read_csv(meta_path)
    node_feature_dim = graphs[0].x.shape[1]

    # Train+val indices
    train_val_mask = meta_df['group'].isin(['training', 'validation'])
    train_val_indices = [i for i in meta_df[train_val_mask].index.tolist() if i < len(graphs)]
    train_val_graphs = [graphs[i] for i in train_val_indices]

    logger.info(f"Train+Val: {len(train_val_graphs)} graphs")

    # Load HP
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}
    logger.info(f"Hyperparameters: {hp}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []
    ckpt_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(train_val_indices)):
        repeat_num = fold_idx // n_folds
        fold_num = fold_idx % n_folds

        logger.info(f"\n--- Repeat {repeat_num+1}/{n_repeats}, Fold {fold_num+1}/{n_folds} ---")

        train_graphs_fold = [train_val_graphs[i] for i in train_idx]
        val_graphs_fold = [train_val_graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs_fold, batch_size=hp.get('batch_size', 32), shuffle=True)
        val_loader = DataLoader(val_graphs_fold, batch_size=hp.get('batch_size', 32), shuffle=False)

        model = RGCNCoordOnly(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 128),
            num_conv_layers=hp.get('num_conv_layers', 3),
            num_edge_types=config.num_edge_types,
            dropout=hp.get('dropout', 0.2),
        )

        bal_acc, best_state = train_fold(
            model, train_loader, val_loader, device,
            lr=hp.get('learning_rate', 1e-3),
            wd=hp.get('weight_decay', 1e-5),
            max_epochs=config.max_epochs,
            patience=config.patience,
        )

        logger.info(f"  Balanced Acc = {bal_acc:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f"repeat{repeat_num}_fold{fold_num}_best.pt")
        torch.save({
            'model_state_dict': best_state,
            'val_balanced_acc': bal_acc,
            'repeat': repeat_num,
            'fold': fold_num,
            'hyperparameters': hp,
            'node_feature_dim': node_feature_dim,
        }, ckpt_path)

        results.append({
            'repeat': repeat_num,
            'fold': fold_num,
            'balanced_acc': bal_acc,
        })

    # Summary
    results_df = pd.DataFrame(results)
    accs = results_df['balanced_acc'].values

    summary = {
        'n_repeats': n_repeats,
        'n_folds': n_folds,
        'balanced_acc': {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'min': float(np.min(accs)),
            'max': float(np.max(accs)),
        },
        'hyperparameters': hp,
    }

    cv_dir = os.path.join(config.output_dir, 'cv_results')
    os.makedirs(cv_dir, exist_ok=True)
    results_df.to_csv(os.path.join(cv_dir, 'cv_fold_results.csv'), index=False)

    for path in [os.path.join(cv_dir, 'cv_statistics.json'),
                 os.path.join(config.output_dir, 'cv_summary.json')]:
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    logger.info(f"\nBalanced Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Coordination-Only RGCN - Statistical Validation")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now()}")

    run_statistical_validation(config, args.n_repeats, args.n_folds)
    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
