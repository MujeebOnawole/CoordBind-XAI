"""
hyper.py - Hyperparameter tuning for coordination-only RGCN

Optimizes balanced accuracy on validation set (not MAE — this is classification).
3-fold CV on train+val set.

Usage:
    python hyper.py --n_trials 25
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
import logging
import pandas as pd

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    raise ImportError("optuna required: pip install optuna")

from config import get_config, SEARCH_SPACE
from loss import coord_loss, compute_coord_metrics
from model import RGCNCoordOnly

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.mode == 'max':
            score = val_score
            improved = self.best_score is None or score > self.best_score + self.min_delta
        else:
            score = -val_score
            improved = self.best_score is None or score > self.best_score + self.min_delta

        if improved:
            self.best_score = score if self.mode == 'max' else -score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, train_loader, optimizer, device):
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

    return total_loss / max(n_batches, 1)


def validate(model, val_loader, device):
    """Validate and return balanced accuracy."""
    model.eval()
    all_preds = []
    all_labels = []

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
    return metrics['balanced_acc']


def objective(trial: Trial, config, graphs, meta_df, device) -> float:
    hidden_dim = trial.suggest_categorical('hidden_dim', SEARCH_SPACE['hidden_dim'])
    num_conv_layers = trial.suggest_int('num_conv_layers', *SEARCH_SPACE['num_conv_layers'])
    dropout = trial.suggest_float('dropout', *SEARCH_SPACE['dropout'])
    learning_rate = trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True)
    batch_size = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])

    node_feature_dim = graphs[0].x.shape[1]

    # Train+val indices
    train_val_mask = meta_df['group'].isin(['training', 'validation'])
    train_val_indices = meta_df[train_val_mask].index.tolist()
    train_val_indices = [i for i in train_val_indices if i < len(graphs)]

    # 3-fold CV
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(train_val_indices)):
        train_idx = [train_val_indices[i] for i in train_fold_idx]
        val_idx = [train_val_indices[i] for i in val_fold_idx]

        train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False)

        model = RGCNCoordOnly(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            num_edge_types=config.num_edge_types,
            dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=10, mode='max')

        best_bal_acc = 0.0
        for epoch in range(50):
            train_epoch(model, train_loader, optimizer, device)
            bal_acc = validate(model, val_loader, device)

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc

            if early_stopping(bal_acc):
                break

            # Optuna pruning — report as negative (minimize direction)
            trial.report(-bal_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_scores.append(best_bal_acc)

    # Return negative balanced acc (Optuna minimizes)
    return -np.mean(fold_scores)


def run_hyperparameter_search(config, n_trials=25):
    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Graphs not found: {graphs_path}. Run build_data.py first.")

    graphs = torch.load(graphs_path)
    meta_df = pd.read_csv(meta_path)

    node_feature_dim = graphs[0].x.shape[1]
    logger.info(f"Loaded {len(graphs)} graphs, feature dim={node_feature_dim}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(
        direction='minimize',  # minimizing -balanced_acc
        study_name='coord_only_hp',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, config, graphs, meta_df, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best balanced acc: {-study.best_value:.4f}")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Save
    results_dir = os.path.join(config.output_dir, 'hyper_results')
    os.makedirs(results_dir, exist_ok=True)

    best_params = study.best_params.copy()
    best_params['best_balanced_acc'] = -study.best_value
    best_params['node_feature_dim'] = node_feature_dim

    for path in [os.path.join(results_dir, 'best_hyperparameters.json'),
                 os.path.join(config.output_dir, 'best_hyperparameters.json')]:
        with open(path, 'w') as f:
            json.dump(best_params, f, indent=2)

    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("Coordination-Only RGCN - Hyperparameter Tuning")
    logger.info("=" * 60)
    logger.info(f"Start: {datetime.now()}")

    run_hyperparameter_search(config, args.n_trials)
    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
