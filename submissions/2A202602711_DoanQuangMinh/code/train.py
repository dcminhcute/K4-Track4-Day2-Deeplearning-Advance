"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Completed implementation for Lab Day 2 (DeepWeeds classification).
Uses a single run(cfg) function for all experiments.
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Add repo root to path for eval imports
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))
from eval import save_predictions, compute_metrics

# Import from code folder
_code_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_code_dir))
from dataset import load_split, check_split, build_transforms, make_loader, CLASS_NAMES
from model import build_model, param_groups, count_params, count_gmacs, freeze_backbone, set_bn_eval
from losses import build_criterion, mix_batch, mixed_loss, class_weights


@dataclass
class Config:
    # --- identification ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- model ---
    backbone: str = "resnet50"
    init: str = "finetune"  # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- data / augmentation ---
    img_size: int = 224
    aug: str = "basic"  # basic | color | trivial | randaug
    sampler: str | None = None  # None | balanced
    mix: str | None = None  # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- optimization (baseline recipe, GUIDE.md section 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 0
    # --- paths ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"  # config.json, history.csv, checkpoint, logits per run
    pred_dir: str = "predictions"  # prediction files in eval.py format
    # --- only enabled at Step 4 (final): save test predictions ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Output directory for a run: <out_dir>/<exp_id>/seed<k>/."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Standard path for prediction files: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Fix all random sources: random, numpy, torch (CPU and CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential Moving Average: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow with model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        # Also handle BN buffers
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name:
                self.shadow[name] = buf.data.clone()

    def update(self, model):
        """Update EMA after optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

        # Also update BN buffers
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name:
                if name in self.shadow:
                    new_average = self.decay * self.shadow[name] + (1 - self.decay) * buf.data
                    self.shadow[name] = new_average.clone()

    def apply(self, model):
        """Apply EMA weights to model for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

        # Also apply to BN buffers
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buf.data.clone()
                buf.data = self.shadow[name].clone()

    def restore(self, model):
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name].clone()

        for name, buf in model.named_buffers():
            if name in self.backup:
                buf.data = self.backup[name].clone()

        self.backup = {}


def build_optimizer(model, cfg: Config):
    """AdamW with 3 parameter groups."""
    groups = param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup then cosine decay."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps > 0 else 1.0
        else:
            if total_steps == warmup_steps:
                return 0.0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: Optional[EMA] = None) -> dict:
    """Train one epoch. Returns dict with train_loss, lr."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Apply mixup/cutmix if configured
        use_mix = cfg.mix in ("mixup", "cutmix") and cfg.mix_alpha > 0

        if use_mix:
            images, (y_a, y_b, lam) = mix_batch(images, labels, cfg.mix_alpha, mode=cfg.mix)

        with autocast('cuda', enabled=cfg.amp):
            logits = model(images)

            if use_mix:
                loss = mixed_loss(criterion, logits, (y_a, y_b, lam))
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        if ema is not None:
            ema.update(model)

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        "train_loss": total_loss / num_batches,
        "lr": optimizer.param_groups[-1]['lr'],  # Head LR
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Run model on loader at eval mode. Returns (filenames, y_true, logits, loss)."""
    model.eval()

    all_filenames = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    num_batches = 0

    for images, labels, filenames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=False):
            logits = model(images)
            loss = criterion(logits, labels)

        all_filenames.extend(filenames)
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1

    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)

    return all_filenames, all_labels, all_logits, total_loss / num_batches


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Plot training curves: loss and macro-F1 over epochs."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_f1 = [h['val_macro_f1'] for h in history]
    val_acc = [h['val_top1'] for h in history]
    lr_history = [h['lr'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Train/Val Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train')
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Val Macro-F1
    axes[0, 1].plot(epochs, val_f1, 'g-', label='Macro-F1')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Macro-F1')
    axes[0, 1].set_title('Validation Macro-F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Val Top-1 Accuracy
    axes[1, 0].plot(epochs, val_acc, 'm-', label='Top-1 Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Top-1 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(epochs, lr_history, 'c-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def run(cfg: Config) -> dict:
    """Train one configuration and save all outputs.

    Returns dict with summary: best_epoch, val_macro_f1, time_per_epoch, etc.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {cfg.exp_id}")
    print(f"Backbone: {cfg.backbone}, init: {cfg.init}, seed: {cfg.seed}")
    print(f"{'='*60}\n")

    # 1. Set seed and create directories
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(rd / "config.json", 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Load data and verify split
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    print(f"\nData loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Only check split if images_dir exists
    images_path = Path(cfg.images_dir)
    if images_path.exists() and len(list(images_path.glob('*.jpg'))) > 0:
        split_stats = check_split(train_df, val_df, test_df, cfg.images_dir)
        print(f"Split verification: PASS")
    else:
        print(f"WARNING: Images directory not found or empty, skipping split check")
        print(f"Expected at: {cfg.images_dir}")
        split_stats = None

    # 3. Build loaders
    train_transform = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(
        train_df, cfg.images_dir, train_transform,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )
    val_loader = make_loader(
        val_df, cfg.images_dir, val_transform,
        batch_size=cfg.batch_size, train=False, sampler=None,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )

    # Test loader only if needed
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = make_loader(
            test_df, cfg.images_dir, val_transform,
            batch_size=cfg.batch_size, train=False, sampler=None,
            num_workers=cfg.num_workers, seed=cfg.seed,
        )

    # 4. Build model
    model = build_model(
        cfg.backbone, pretrained=(cfg.init != "scratch"),
        num_classes=9, drop_rate=cfg.drop_rate, init=cfg.init,
    )
    model = model.to(device)

    # Log model info
    n_params = count_params(model)
    print(f"Model: {cfg.backbone}")
    print(f"Parameters: {n_params:.2f}M")
    print(f"Tag: {model.pretrained_cfg.get('url', 'unknown')}")

    # 5. Build criterion
    if cfg.loss == "ce_weighted" and cfg.class_weight_beta is not None:
        class_counts = train_df['Label'].value_counts().sort_index().values
        weights = class_weights(class_counts, beta=cfg.class_weight_beta).to(device)
        criterion = build_criterion(cfg.loss, weight=weights)
    else:
        criterion = build_criterion(
            cfg.loss,
            smoothing=cfg.label_smoothing,
            gamma=cfg.focal_gamma,
        )

    # 6. Build optimizer, scheduler, scaler, EMA
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = GradScaler('cuda', enabled=cfg.amp)

    ema = None
    if cfg.ema_decay is not None:
        ema = EMA(model, cfg.ema_decay)
        print(f"Using EMA with decay={cfg.ema_decay}")

    # 7. Training loop
    history = []
    best_f1 = -1.0
    best_epoch = 0

    print(f"\nTraining for {cfg.epochs} epochs...")
    epoch_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_time = time.time()

        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )

        # Evaluate (use EMA weights if available)
        if ema is not None:
            ema.apply(model)
            set_bn_eval(model)

        filenames, y_true, logits, val_loss = evaluate(model, val_loader, criterion, device)
        probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        y_pred = probs.argmax(axis=1)

        metrics = compute_metrics(y_true, y_pred, probs)
        val_f1 = metrics['macro_f1']
        val_acc = metrics['top1']

        if ema is not None:
            ema.restore(model)

        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': train_stats['train_loss'],
            'val_loss': val_loss,
            'val_macro_f1': val_f1,
            'val_top1': val_acc,
            'lr': train_stats['lr'],
        })

        epoch_time = time.time() - epoch_time
        print(f"Epoch {epoch:2d}/{cfg.epochs} | "
              f"train_loss: {train_stats['train_loss']:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_f1: {val_f1:.4f} | "
              f"val_acc: {val_acc:.4f} | "
              f"lr: {train_stats['lr']:.2e} | "
              f"time: {epoch_time:.1f}s")

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, rd / "best_checkpoint.pt")
            print(f"  -> New best F1: {best_f1:.4f}")

    total_time = time.time() - epoch_start
    time_per_epoch = total_time / cfg.epochs

    print(f"\nTraining complete!")
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Total time: {total_time:.1f}s ({time_per_epoch:.1f}s/epoch)")

    # 8. Load best checkpoint and save predictions
    checkpoint = torch.load(rd / "best_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Val predictions
    filenames, y_true, logits, _ = evaluate(model, val_loader, criterion, device)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()

    # Save val predictions using eval.save_predictions
    pred_dir = Path(cfg.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_pred_path = pred_path(cfg, "val")
    save_predictions(str(val_pred_path), filenames, y_true, probs)
    print(f"Saved val predictions to {val_pred_path}")

    # Test predictions (only if enabled)
    if cfg.save_test_predictions and test_loader is not None:
        filenames, y_true, logits, _ = evaluate(model, test_loader, criterion, device)
        probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        test_pred_path = pred_path(cfg, "test")
        save_predictions(str(test_pred_path), filenames, y_true, probs)
        print(f"Saved test predictions to {test_pred_path}")

    # 9. Save history and plot curves
    history_df = pd.DataFrame(history)
    history_df.to_csv(rd / "history.csv", index=False)

    plot_path = rd / f"{cfg.exp_id}.png"
    plot_curves(history, plot_path, f"{cfg.exp_id} - {cfg.backbone}")
    print(f"Saved training curves to {plot_path}")

    # Return summary
    return {
        'exp_id': cfg.exp_id,
        'seed': cfg.seed,
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'best_val_acc': val_acc,
        'time_per_epoch': time_per_epoch,
        'n_params': n_params,
        'history': history,
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Parse ['seed=1', 'loss=focal', 'ema_decay=None'] into a dict.

    Converts types according to Config field types.
    """
    overrides = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid override: {pair} (expected key=value)")
        key, value = pair.split('=', 1)

        # Convert None
        if value.lower() == 'none':
            converted = None
        elif value.lower() == 'true':
            converted = True
        elif value.lower() == 'false':
            converted = False
        else:
            # Try int
            try:
                converted = int(value)
            except ValueError:
                # Try float
                try:
                    converted = float(value)
                except ValueError:
                    # Keep as string
                    converted = value

        overrides[key] = converted

    return overrides


def main():
    """CLI entry point: python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    parser = argparse.ArgumentParser(description="Train DeepWeeds model")
    parser.add_argument('--set', nargs='+', default=[],
                        help='Config overrides in key=value format')
    args = parser.parse_args()

    # Parse overrides
    overrides = parse_overrides(args.set)

    # Create config with overrides
    cfg = Config()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")

    # Run
    result = run(cfg)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"exp_id: {result['exp_id']}")
    print(f"seed: {result['seed']}")
    print(f"best_epoch: {result['best_epoch']}")
    print(f"best_val_f1: {result['best_val_f1']:.4f}")
    print(f"best_val_acc: {result['best_val_acc']:.4f}")
    print(f"time_per_epoch: {result['time_per_epoch']:.1f}s")
    print(f"n_params: {result['n_params']:.2f}M")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
