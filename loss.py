"""
loss.py - Simple MSE Loss for logK1 Stability Constant Prediction

Simplified from tmGNN-XAI MTL loss:
- No masked MTL loss (single target, no NaN values)
- No inverse-variance weighting (single property)
- No physics constraint (no HL_Gap = LUMO - HOMO)
- Just MSE loss
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


def mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss for single-task logK1 prediction.

    Args:
        outputs: Model predictions (batch, 1)
        targets: Ground truth logK1 (batch, 1)

    Returns:
        Scalar MSE loss
    """
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    # Handle any NaN values (shouldn't happen but be safe)
    valid_mask = ~torch.isnan(targets)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    diff = outputs - targets
    squared_errors = diff ** 2
    masked_errors = torch.where(valid_mask, squared_errors, torch.zeros_like(squared_errors))

    loss = masked_errors.sum() / valid_mask.sum()

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    return loss


def compute_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute MAE for logK1.

    Args:
        predictions: Predictions array (N,) or (N, 1)
        targets: Targets array (N,) or (N, 1)

    Returns:
        MAE value
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    mask = ~np.isnan(targets)
    if mask.sum() == 0:
        return float('nan')

    return float(np.mean(np.abs(predictions[mask] - targets[mask])))
