"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Completed implementation for Lab Day 2 (DeepWeeds classification).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_criterion(kind: str = "ce", **kw):
    """Trả về một callable loss function.

    Args:
        kind: 'ce', 'ls' (label smoothing), 'focal', 'ce_weighted'
        **kw: additional arguments like smoothing, gamma, alpha, weight

    Returns:
        callable(logits, target) -> loss scalar
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get('smoothing', 0.1)
        return LabelSmoothingCE(smoothing)
    elif kind == "focal":
        gamma = kw.get('gamma', 2.0)
        alpha = kw.get('alpha', None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get('weight', None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smoothed target: (1 - eps) * one_hot + eps / K
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalLoss(nn.Module):
    """Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a tensor of class weights

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(logits.device)[target]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Compute class weights from class counts in training set.

    - beta=0: weights = 1/n_c (inversely proportional to count), normalized to mean 1
    - beta>0: effective number of samples: w_c = (1 - beta) / (1 - beta ** n_c)

    Args:
        counts: array-like of class counts in training set
        beta: 0 for inverse frequency, >0 for class-balanced effective number

    Returns:
        torch.Tensor of weights for each class
    """
    counts = np.array(counts, dtype=np.float32)
    n_classes = len(counts)

    if beta == 0:
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.mean()  # Normalize so mean weight = 1
    else:
        # Class-balanced effective number of samples
        # w_c = (1 - beta) / (1 - beta ** n_c)
        weights = (1.0 - beta) / (1.0 - beta ** counts)
        weights = weights / weights.sum() * n_classes  # Normalize so sum = K

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    Args:
        x: input images (B, C, H, W)
        y: input labels (B,)
        alpha: Beta distribution parameter
        mode: 'mixup' or 'cutmix'

    Returns:
        (x_mixed, (y_a, y_b, lam))
        - x_mixed: mixed images
        - y_a: original labels
        - y_b: shuffled labels (y[perm])
        - lam: actual mixing ratio based on mixed area
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices
    indices = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[indices]

    if mode == "mixup":
        # Linear interpolation
        x_mixed = lam * x + (1 - lam) * x[indices]
    elif mode == "cutmix":
        # Cut and paste a rectangular region
        B, C, H, W = x.shape

        # Generate random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random box center
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)

        # Box boundaries (clip to image borders)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)

        # Create mixed image
        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]

        # Recompute lam based on actual area
        actual_area = (y2 - y1) * (x2 - x1)
        total_area = H * W
        lam = actual_area / total_area
    else:
        raise ValueError(f"Unknown mix mode: {mode}")

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Args:
        criterion: loss function
        logits: model output (B, K)
        targets: tuple (y_a, y_b, lam) from mix_batch

    Returns:
        scalar loss
    """
    if not isinstance(targets, tuple) or len(targets) != 3:
        raise ValueError("targets must be (y_a, y_b, lam) from mix_batch")

    y_a, y_b, lam = targets
    loss_a = criterion(logits, y_a)
    loss_b = criterion(logits, y_b)

    return lam * loss_a + (1 - lam) * loss_b


# ============== SELF-CHECK FUNCTIONS ==============

def check_focal_equals_ce():
    """Verify focal loss with gamma=0 equals cross-entropy."""
    print("Checking focal loss with gamma=0 equals CE...")

    logits = torch.randn(32, 9)
    target = torch.randint(0, 9, (32,))

    ce_loss = F.cross_entropy(logits, target)
    focal_loss = FocalLoss(gamma=0.0)
    f_loss = focal_loss(logits, target)

    diff = abs(ce_loss.item() - f_loss.item())
    print(f"  CE loss: {ce_loss.item():.6f}")
    print(f"  Focal(gamma=0) loss: {f_loss.item():.6f}")
    print(f"  Difference: {diff:.10f}")
    print(f"  {'PASS' if diff < 1e-6 else 'FAIL'}")
    return diff < 1e-6


def check_cutmix_area():
    """Verify CutMix lambda matches the actual mixed area."""
    print("Checking CutMix lambda matches actual area...")

    B, C, H, W = 4, 3, 224, 224
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, 9, (B,))

    x_mixed, (y_a, y_b, lam) = mix_batch(x, y, alpha=1.0, mode="cutmix")

    # Count how many pixels are from index (mixed region)
    diff_mask = (x_mixed != x).any(dim=1)  # (B, H, W)
    actual_mixed_pixels = diff_mask.sum().item()
    total_pixels = H * W
    actual_lam = actual_mixed_pixels / total_pixels

    print(f"  Expected lam: {lam:.4f}")
    print(f"  Actual lam: {actual_lam:.4f}")
    print(f"  Difference: {abs(lam - actual_lam):.6f}")
    print(f"  {'PASS' if abs(lam - actual_lam) < 0.01 else 'FAIL'}")
    return abs(lam - actual_lam) < 0.01
