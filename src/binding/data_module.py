"""
data_module.py - Data loading utilities for logK1 stability constant prediction

Simplified from tmGNN-XAI data_module:
- Single target (logK1)
- Stratified splitting by metal_type
- Labels already assigned in build_data.py
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_dataset(
    graphs_path: str,
    meta_path: str,
) -> Tuple[List[Data], pd.DataFrame]:
    """
    Load graphs and metadata.

    Args:
        graphs_path: Path to .pt file with graphs
        meta_path: Path to .csv file with metadata

    Returns:
        Tuple of (graphs list, metadata DataFrame)
    """
    logger.info(f"Loading graphs from {graphs_path}")
    graphs = torch.load(graphs_path)

    logger.info(f"Loading metadata from {meta_path}")
    meta_df = pd.read_csv(meta_path)

    # Ensure labels are assigned (should already be from build_data.py)
    labels_missing = 0
    for i, graph in enumerate(graphs):
        if not hasattr(graph, 'y') or graph.y is None:
            if i < len(meta_df) and 'logK1' in meta_df.columns:
                val = meta_df.iloc[i]['logK1']
                graph.y = torch.tensor([float(val)], dtype=torch.float32)
            else:
                labels_missing += 1

    if labels_missing > 0:
        logger.warning(f"{labels_missing} graphs have no labels")

    logger.info(f"Loaded {len(graphs)} graphs")
    return graphs, meta_df


def get_data_splits(
    graphs: List[Data],
    meta_df: pd.DataFrame,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Split graphs by group column in metadata.

    Args:
        graphs: List of PyG Data objects
        meta_df: Metadata DataFrame with 'group' column

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    train_graphs = []
    val_graphs = []
    test_graphs = []

    for i, graph in enumerate(graphs):
        if i < len(meta_df):
            group = meta_df.iloc[i].get('group', 'training')
            if group == 'training':
                train_graphs.append(graph)
            elif group in ['valid', 'validation']:
                val_graphs.append(graph)
            elif group == 'test':
                test_graphs.append(graph)
            else:
                train_graphs.append(graph)

    logger.info(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
    return train_graphs, val_graphs, test_graphs


def create_dataloaders(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test."""
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
