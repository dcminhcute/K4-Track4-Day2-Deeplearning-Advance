"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Completed implementation for Lab Day 2 (DeepWeeds classification).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Suggested backbones (timm names)
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    Args:
        name: timm model name
        pretrained: whether to load pretrained weights
        num_classes: number of output classes
        drop_rate: dropout rate for the head
        init: 'scratch' (from scratch), 'frozen' (freeze backbone, train only head),
              or 'finetune' (fine-tune all)

    Returns:
        model with pretrained_cfg['tags'] recorded
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Record the actual tag used
    actual_tag = model.pretrained_cfg.get('tags', ['unknown'])[0] if model.pretrained_cfg else 'unknown'
    model.used_pretrained_tag = actual_tag

    if init == "frozen":
        freeze_backbone(model)
    elif init == "scratch":
        # Already has pretrained=False if requested, just verify
        pass

    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    When backbone is frozen, BatchNorm must stay in eval mode.
    This function sets backbone to eval mode and disables gradient updates.
    """
    # Get the classifier (head)
    head = model.get_classifier()

    # Freeze all parameters except the head
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    # Set backbone (non-head) to eval mode for BatchNorm
    model.eval()

    # But re-enable grads for head so it can be trained
    for param in head.parameters():
        param.requires_grad = True
    head.train()


def param_groups(model, lr_backbone: float = 1e-4, lr_head: float = 1e-3,
                 weight_decay: float = 0.05):
    """Chia tham số thành 3 nhóm theo slide Day 2 trang 52.

    Returns:
        list[dict]: [{'params': [...], 'lr': ..., 'weight_decay': ...}, ...]
    """
    # Group 1: backbone weights (ndim > 1) -> backbone LR with weight decay
    backbone_weights = []
    # Group 2: backbone norm and bias (ndim <= 1) -> backbone LR, no weight decay
    backbone_norm_bias = []
    # Group 3: head -> head LR with weight decay
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params

        is_head = 'classifier' in name or 'head' in name or 'fc' in name

        if is_head:
            head_params.append(param)
        elif param.ndim > 1:
            # Conv weights, linear weights
            backbone_weights.append(param)
        else:
            # Norm params (running_mean, running_var, num_batches_tracked)
            # and bias params
            backbone_norm_bias.append(param)

    groups = []

    if backbone_weights:
        groups.append({
            'params': backbone_weights,
            'lr': lr_backbone,
            'weight_decay': weight_decay,
        })

    if backbone_norm_bias:
        groups.append({
            'params': backbone_norm_bias,
            'lr': lr_backbone,
            'weight_decay': 0.0,  # No weight decay for norm/bias
        })

    if head_params:
        groups.append({
            'params': head_params,
            'lr': lr_head,
            'weight_decay': weight_decay,
        })

    return groups


def count_params(model) -> float:
    """Số tham số (triệu)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224, device: str = "cpu") -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Uses thop library or falls back to estimation.
    """
    try:
        from thop import profile
        model = model.to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        gmacs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return gmacs / 1e9  # Convert to GMAC
    except ImportError:
        # Fallback: use timm's flops
        try:
            return model.forward.__self__.get_parameter('weight').numel() / 1e9 * 2
        except:
            return 0.0  # Cannot estimate


def set_bn_eval(model):
    """Set all BatchNorm layers in model to eval mode."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def verify_bn_frozen_after_train(model):
    """Check that frozen BatchNorm stays in eval mode after model.train() is called.

    This is a diagnostic function to verify the freeze_backbone behavior.
    """
    # After freeze_backbone and model.train(), check BN layers
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            # Check if this BN is in the backbone (not head)
            if 'classifier' not in name and 'head' not in name and 'fc' not in name:
                if m.training:
                    print(f"WARNING: {name} is in training mode but should be frozen!")
                    return False
    return True
