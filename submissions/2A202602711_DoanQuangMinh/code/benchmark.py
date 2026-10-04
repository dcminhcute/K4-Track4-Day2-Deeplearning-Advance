"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Completed implementation for Lab Day 2 (DeepWeeds classification).
Rules:
  - warmup: discard >= 10 initial forward passes
  - GPU sync: torch.cuda.synchronize() before and after measurement
  - >= 50 runs, report p50, p95, p99, throughput (img/s)
  - report GPU, dtype (FP32/AMP/FP16), batch size, image resolution
"""
from __future__ import annotations

import time
import torch
import numpy as np
from typing import Optional, Callable


def bench(fn: Callable, warmup: int = 10, iters: int = 100,
          sync: Optional[Callable] = None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    Args:
        fn: hàm cần đo
        warmup: số lần warmup trước khi đo
        iters: số lần đo
        sync: hàm đồng bộ (torch.cuda.synchronize) hoặc None trên CPU

    Returns:
        dict với p50, p95, p99, mean, std, min, max, n
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Benchmark
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)

    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(times.mean()),
        "std": float(times.std()),
        "min": float(times.min()),
        "max": float(times.max()),
        "n": iters,
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên.

    Args:
        model: nn.Module
        batch_size: kích thước batch
        img_size: kích thước ảnh
        dtype: "fp32" | "amp" | "fp16"
        device: "cuda" | "cpu"
        warmup: số lần warmup
        iters: số lần đo

    Returns:
        dict với latency metrics
    """
    model.eval()

    # Dtype and device setup
    if device == "cuda" and torch.cuda.is_available():
        target_device = torch.device("cuda")
        sync_fn = torch.cuda.synchronize
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        target_device = torch.device("cpu")
        sync_fn = None
        gpu_name = "CPU"
        gpu_mem = 0.0

    model = model.to(target_device)

    if dtype == "fp16" and target_device.type == "cuda":
        model = model.half()
        input_dtype = torch.float16
    else:
        input_dtype = torch.float32

    x = torch.randn(batch_size, 3, img_size, img_size, dtype=input_dtype, device=target_device)

    def forward_fn():
        with torch.inference_mode():
            if dtype == "amp" and target_device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(x)
            else:
                _ = model(x)

    result = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    # Throughput: batch_size / (latency_seconds) = batch_size / (p50 / 1000)
    throughput = batch_size / (result["p50"] / 1000.0) if result["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": result["p50"],
        "p95": result["p95"],
        "p99": result["p99"],
        "mean": result["mean"],
        "std": result["std"],
        "min": result["min"],
        "max": result["max"],
        "images_per_s": throughput,
        "mem_gb": gpu_mem,
        "torch": torch.__version__,
        "warmup": warmup,
        "iters": iters,
    }


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224,
                 device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ của TTA K views."""
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        target_device = torch.device("cuda")
        sync_fn = torch.cuda.synchronize
    else:
        target_device = torch.device("cpu")
        sync_fn = None

    model = model.to(target_device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    def single_forward():
        with torch.inference_mode():
            if target_device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(x)
            else:
                _ = model(x)

    single_result = bench(single_forward, warmup=warmup, iters=iters, sync=sync_fn)

    def tta_forward():
        with torch.inference_mode():
            for _ in range(k_views):
                if target_device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        _ = model(x)
                else:
                    _ = model(x)

    tta_result = bench(tta_forward, warmup=warmup, iters=iters, sync=sync_fn)

    return {
        "k_views": k_views,
        "batch_size": batch_size,
        "img_size": img_size,
        "device": device,
        "single_p50": single_result["p50"],
        "single_p95": single_result["p95"],
        "tta_p50": tta_result["p50"],
        "tta_p95": tta_result["p95"],
        "tta_p99": tta_result["p99"],
        "ratio": tta_result["p50"] / single_result["p50"] if single_result["p50"] > 0 else 0,
        "expected_ratio": k_views,
        "warmup": warmup,
        "iters": iters,
    }
