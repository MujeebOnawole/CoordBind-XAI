"""
hyper.py - Hyperparameter Tuning for logK1 Stability Constant Prediction

Single-task Optuna optimization using 3-fold CV on train+val set.
Optimizes MAE on logK1.

Usage:
    python hyper.py --n_trials 25
    python hyper.py --n_trials 50 --output_dir /path/to/output
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
    raise ImportError("optuna is required. Install with: pip install optuna")

from config import get_config, SEARCH_SPACE
from loss import mse_loss, compute_mae
from model import RGCNStability

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        targets = batch.y

        optimizer.zero_grad()
        outputs = model(batch)

        # Reshape targets
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        loss = mse_loss(outputs, targets)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float('inf')


def validate(model, val_loader, device):
    """Validate model and return MAE."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) == 0:
        return float('inf')

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return compute_mae(all_preds, all_targets)


def objective(trial: Trial, config, graphs, meta_df, device, node_feature_dim) -> float:
    """Optuna objective function."""

    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', SEARCH_SPACE['hidden_dim'])
    num_conv_layers = trial.suggest_int('num_conv_layers', *SEARCH_SPACE['num_conv_layers'])
    dropout = trial.suggest_float('dropout', *SEARCH_SPACE['dropout'])
    learning_rate = trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True)
    batch_size = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])

    # Get train+val indices (exclude test)
    train_val_mask = meta_df['group'].isin(['training', 'validation'])
    train_val_indices = meta_df[train_val_mask].index.tolist()
    train_val_indices = [i for i in train_val_indices if i < len(graphs)]

    if trial.number == 0:
        logger.info(f"Train+val size: {len(train_val_indices)}")

    # 3-Fold CV
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    for fold, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(train_val_indices)):
        train_idx = [train_val_indices[i] for i in train_fold_idx]
        val_idx = [train_val_indices[i] for i in val_fold_idx]

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

        model = RGCNStability(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            num_edge_types=config.num_edge_types,
            dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=10)

        max_epochs = 50
        best_mae = float('inf')

        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_mae = validate(model, val_loader, device)

            if val_mae < best_mae:
                best_mae = val_mae

            if early_stopping(val_mae):
                break

            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_maes.append(best_mae)

    return np.mean(fold_maes)


def run_hyperparameter_search(config, n_trials: int = 25):
    """Run hyperparameter search."""

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")

    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Graphs not found: {graphs_path}. Run build_data.py first.")

    # Load data
    logger.info(f"Loading graphs from {graphs_path}")
    graphs = torch.load(graphs_path)

    node_feature_dim = graphs[0].x.shape[1]

    logger.info("=" * 60)
    logger.info("GRAPH DATA DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Number of graphs: {len(graphs)}")
    logger.info(f"Node feature dimension: {node_feature_dim}")

    sample = graphs[0]
    logger.info(f"Sample graph - nodes: {sample.x.shape[0]}, edges: {sample.edge_index.shape[1]}")
    if hasattr(sample, 'edge_type'):
        logger.info(f"Unique edge types: {torch.unique(sample.edge_type).tolist()}")

    # Check labels
    valid_labels = sum(1 for g in graphs if hasattr(g, 'y') and g.y is not None and not torch.isnan(g.y).all())
    logger.info(f"Graphs with valid labels: {valid_labels}/{len(graphs)}")
    logger.info("=" * 60)

    meta_df = pd.read_csv(meta_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='stability_logk1_hp',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    logger.info(f"\nStarting hyperparameter search with {n_trials} trials...")

    study.optimize(
        lambda trial: objective(trial, config, graphs, meta_df, device, node_feature_dim),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETER SEARCH RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val MAE: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save results
    results_dir = os.path.join(config.output_dir, 'hyper_results')
    os.makedirs(results_dir, exist_ok=True)

    best_params = study.best_params.copy()
    best_params['best_mae'] = study.best_value
    best_params['node_feature_dim'] = node_feature_dim

    best_hp_path = os.path.join(results_dir, 'best_hyperparameters.json')
    with open(best_hp_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(os.path.join(config.output_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        if trial.state.name == 'COMPLETE':
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
            })
    with open(os.path.join(results_dir, 'all_trials.json'), 'w') as f:
        json.dump(trials_data, f, indent=2)

    return study.best_params


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for logK1 prediction')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("logK1 Stability Constant - Hyperparameter Tuning")
    logger.info("=" * 60)
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Start: {datetime.now()}")

    best_params = run_hyperparameter_search(config, args.n_trials)

    logger.info(f"\nBest hyperparameters: {best_params}")
    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
