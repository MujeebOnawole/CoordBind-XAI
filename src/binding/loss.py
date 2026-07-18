"""
loss.py - Joint Loss for logK1 + Coordination Prediction

Two components:
1. MSE loss on logK1 (binding strength)
2. BCE loss on per-atom coordination labels (which atoms are donors)

Combined: L = MSE(logK1) + coord_weight * BCE(coord_labels)

The coordination labels use -1.0 as a mask sentinel for metal atoms,
which are excluded from the BCE computation (metals are never donors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict


def joint_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    coord_labels: torch.Tensor,
    coord_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute joint logK1 + coordination prediction loss.

    Args:
        outputs: Dict with 'logk1' (batch, 1) and 'coord_logits' (total_nodes,)
        targets: Ground truth logK1 (batch, 1)
        coord_labels: Per-atom labels: 1.0=donor, 0.0=non-donor, -1.0=metal (masked)
        coord_weight: Weight for coordination loss relative to binding loss

    Returns:
        Dict with 'total', 'binding', 'coordination' loss tensors
    """
    # === Binding loss (MSE on logK1) ===
    logk1_pred = outputs['logk1']
    if logk1_pred.dim() == 1:
        logk1_pred = logk1_pred.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    # Index into valid entries BEFORE computing loss to avoid NaN gradient
    # propagation. torch.where masks values but NaN gradients still flow
    # through the computation graph (0 * NaN = NaN in IEEE 754).
    valid_mask = ~torch.isnan(targets.squeeze())
    if valid_mask.sum() > 0:
        valid_pred = logk1_pred.squeeze()[valid_mask]
        valid_targets = targets.squeeze()[valid_mask]
        binding_loss = F.mse_loss(valid_pred, valid_targets)
    else:
        binding_loss = torch.tensor(0.0, device=logk1_pred.device, requires_grad=True)

    # === Coordination loss (BCE on per-atom donor prediction) ===
    coord_logits = outputs['coord_logits']

    # Mask out metal atoms (coord_labels == -1.0) — they are never donors
    atom_mask = coord_labels >= 0.0
    if atom_mask.sum() > 0:
        masked_logits = coord_logits[atom_mask]
        masked_labels = coord_labels[atom_mask]

        # Class weighting: donors are rare (~10-15% of non-metal atoms)
        n_pos = masked_labels.sum().clamp(min=1)
        n_neg = (1.0 - masked_labels).sum().clamp(min=1)
        pos_weight = n_neg / n_pos

        coord_loss = F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_labels,
            pos_weight=pos_weight.expand_as(masked_logits),
        )
    else:
        coord_loss = torch.tensor(0.0, device=coord_logits.device, requires_grad=True)

    total = binding_loss + coord_weight * coord_loss

    return {
        'total': total,
        'binding': binding_loss.detach(),
        'coordination': coord_loss.detach(),
    }


def mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss for single-task logK1 prediction (backward compatible).

    Args:
        outputs: Model predictions (batch, 1) — accepts either Tensor or Dict
        targets: Ground truth logK1 (batch, 1)

    Returns:
        Scalar MSE loss
    """
    # Handle dict output from new model
    if isinstance(outputs, dict):
        outputs = outputs['logk1']

    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    # Index into valid entries BEFORE computing loss (same NaN gradient fix)
    valid_mask = ~torch.isnan(targets.squeeze())
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    valid_pred = outputs.squeeze()[valid_mask]
    valid_targets = targets.squeeze()[valid_mask]

    return F.mse_loss(valid_pred, valid_targets)


def compute_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute MAE for logK1."""
    predictions = predictions.flatten()
    targets = targets.flatten()

    mask = ~np.isnan(targets)
    if mask.sum() == 0:
        return float('nan')

    return float(np.mean(np.abs(predictions[mask] - targets[mask])))
