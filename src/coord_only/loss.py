"""
loss.py - Coordination-only BCE loss

No binding loss. Pure per-atom donor classification.
"""

import torch
import torch.nn.functional as F
import numpy as np


def coord_loss(coord_logits: torch.Tensor, coord_labels: torch.Tensor) -> torch.Tensor:
    """
    BCE loss for coordination site prediction.

    Args:
        coord_logits: (total_nodes,) raw logits from model
        coord_labels: (total_nodes,) 1.0=donor, 0.0=non-donor

    Returns:
        Scalar BCE loss with class weighting
    """
    # All labels should be 0 or 1 (no -1 sentinel since no metal atoms)
    valid_mask = coord_labels >= 0.0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=coord_logits.device, requires_grad=True)

    logits = coord_logits[valid_mask]
    labels = coord_labels[valid_mask]

    # Class weighting: donors are rare (~10-20% of atoms)
    n_pos = labels.sum().clamp(min=1)
    n_neg = (1.0 - labels).sum().clamp(min=1)
    pos_weight = n_neg / n_pos

    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight.expand_as(logits)
    )


def compute_coord_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """
    Compute coordination prediction metrics.

    Args:
        predictions: (n_atoms,) sigmoid probabilities
        labels: (n_atoms,) binary labels

    Returns:
        Dict with tp, fp, tn, fn, balanced_acc, donor_recall, neg_acc
    """
    pred_binary = (predictions >= threshold).astype(int)
    labels_int = labels.astype(int)

    tp = int(((pred_binary == 1) & (labels_int == 1)).sum())
    fp = int(((pred_binary == 1) & (labels_int == 0)).sum())
    tn = int(((pred_binary == 0) & (labels_int == 0)).sum())
    fn = int(((pred_binary == 0) & (labels_int == 1)).sum())

    donor_recall = tp / max(tp + fn, 1)
    neg_acc = tn / max(tn + fp, 1)
    balanced_acc = (donor_recall + neg_acc) / 2.0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'donor_recall': donor_recall,
        'neg_acc': neg_acc,
        'balanced_acc': balanced_acc,
        'overall_acc': (tp + tn) / max(tp + fp + tn + fn, 1),
    }
