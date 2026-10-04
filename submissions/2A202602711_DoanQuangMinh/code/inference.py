"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Completed implementation for Lab Day 2 (DeepWeeds classification).
Supports:
  - TTA: horizontal flip, multi-crop, multi-scale
  - Aggregation: probability averaging vs logit averaging
  - Model ensembling
  - Temperature scaling (optimizing NLL on val set)
  - Conv-BatchNorm fusion for faster inference
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable, Tuple
from scipy.optimize import minimize_scalar


def predict_logits(model, loader, device, view: Optional[Callable] = None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    Args:
        model: nn.Module ở chế độ eval
        loader: DataLoader
        device: torch device
        view: hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang)

    Returns:
        (filenames, y_true, logits[N, 9])
    """
    model.eval()

    all_filenames = []
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)

            # Áp dụng view transform nếu có
            if view is not None:
                images = view(images)

            # Forward với autocast
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled="cuda" in str(device)):
                logits = model(images)

            all_filenames.extend(filenames)
            all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
            all_logits.append(logits.cpu().numpy())

    filenames = all_filenames
    y_true = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)

    return filenames, y_true, logits


def view_identity(x):
    """View transform gốc (không thay đổi)."""
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    if isinstance(x, np.ndarray):
        return np.flip(x, axis=-1).copy()
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`.

    Args:
        x: (N, C, H, W) batch ảnh
        crop: kích thước crop

    Returns:
        list of (N, C, crop, crop) tensors
    """
    N, C, H, W = x.shape
    if H < crop or W < crop:
        return [F.interpolate(x, size=(crop, crop), mode='bilinear', align_corners=False)]

    positions = [
        (0, 0),                              # top-left
        (W - crop, 0),                       # top-right
        (0, H - crop),                       # bottom-left
        (W - crop, H - crop),                # bottom-right
        ((W - crop) // 2, (H - crop) // 2),  # center
    ]

    crops = []
    for x1, y1 in positions:
        crop_img = x[:, :, y1:y1 + crop, x1:x1 + crop]
        crops.append(crop_img)

    return crops


def views_multiscale(x, sizes: List[int]):
    """Resize batch về từng kích thước trong `sizes`.

    Args:
        x: (N, C, H, W) batch ảnh
        sizes: list of target sizes

    Returns:
        list of tensors
    """
    results = []
    for size in sizes:
        resized = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        results.append(resized)
    return results


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán.

    Args:
        logits_per_view: list of logits arrays, each (N, 9)
        space: "prob" (softmax rồi trung bình) hoặc "logit" (trung bình rồi softmax)

    Returns:
        probs (N, 9) đã chuẩn hoá
    """
    if space == "prob":
        all_probs = [F.softmax(torch.tensor(lg), dim=1).numpy() for lg in logits_per_view]
        probs = np.mean(all_probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        probs = F.softmax(torch.tensor(avg_logits), dim=1).numpy()
    else:
        raise ValueError(f"Unknown space: {space}")

    # Chuẩn hoá để tổng xác suất mỗi dòng bằng đúng 1
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình.

    Args:
        list_of_probs: list of probability arrays, each (N, 9)

    Returns:
        probs (N, 9) đã chuẩn hoá
    """
    avg_probs = np.mean(list_of_probs, axis=0)
    avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)
    return avg_probs


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T).

    Args:
        val_logits: (N, 9) raw logits
        val_labels: (N,) class labels

    Returns:
        optimal temperature T
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.tensor(val_logits, dtype=torch.float32, device=device)
    labels = torch.tensor(val_labels, dtype=torch.long, device=device)

    def nll(T):
        if T <= 0:
            return 1e9
        scaled_logits = logits / T
        log_probs = F.log_softmax(scaled_logits, dim=1)
        loss = -log_probs[torch.arange(len(labels), device=device), labels].mean().item()
        return loss

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method='bounded')
    return float(result.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T).

    Args:
        logits: (N, 9) raw logits
        T: temperature

    Returns:
        probs (N, 9)
    """
    probs = F.softmax(torch.tensor(logits) / T, dim=1).numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d + BatchNorm2d thành một Conv2d mới."""
    bn.eval()
    conv.eval()

    device = conv.weight.device
    bn_weight = bn.weight.to(device)
    bn_bias = bn.bias.to(device)
    bn_mean = bn.running_mean.to(device)
    bn_var = bn.running_var.to(device)
    bn_eps = bn.eps

    std = torch.sqrt(bn_var + bn_eps)
    gamma_div_std = bn_weight / std

    # Fused Conv
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device)

    # w' = gamma * w / sqrt(var + eps)
    fused_conv.weight = nn.Parameter(
        conv.weight * gamma_div_std.view(-1, 1, 1, 1)
    )

    # b' = beta + gamma * (b - mean) / sqrt(var + eps)
    if conv.bias is not None:
        conv_bias = conv.bias.to(device)
    else:
        conv_bias = torch.zeros(conv.out_channels, device=device)

    fused_conv.bias = nn.Parameter(
        bn_bias + gamma_div_std * (conv_bias - bn_mean)
    )

    return fused_conv


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận.

    Args:
        model: nn.Module cần fuse

    Returns:
        model đã được fuse
    """
    model.eval()
    fused_count = 0
    max_error = 0.0

    # Dò tìm các khối Conv2d + BatchNorm2d trong tuần tự
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            # Check sequential or nested submodules
            if isinstance(child, nn.Sequential):
                i = 0
                while i < len(child) - 1:
                    m1 = child[i]
                    m2 = child[i + 1]
                    if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                        fused_conv = _fuse_conv_bn(m1, m2)
                        child[i] = fused_conv
                        child[i + 1] = nn.Identity()
                        fused_count += 1
                        i += 2
                    else:
                        i += 1

    print(f"Fused {fused_count} Conv-BN pairs")
    return model
