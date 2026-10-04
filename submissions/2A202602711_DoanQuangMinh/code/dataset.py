"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Completed implementation for Lab Day 2 (DeepWeeds classification).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Each file has columns `Filename, Label, Species`. Returns three DataFrames.
    Does NOT edit, filter, or re-split the data.
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Returns dict of statistics."""
    images_dir = Path(images_dir)

    # 1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total = n_train + n_val + n_test

    # Per-class counts
    def class_counts(df):
        return df['Label'].value_counts().sort_index()

    train_class = class_counts(train_df)
    val_class = class_counts(val_df)
    test_class = class_counts(test_df)

    # 2. Giao của từng cặp tập theo Filename phải RỖNG
    train_files = set(train_df['Filename'])
    val_files = set(val_df['Filename'])
    test_files = set(test_df['Filename'])

    train_val_overlap = len(train_files & val_files)
    train_test_overlap = len(train_files & test_files)
    val_test_overlap = len(val_files & test_files)

    # 3. Hợp ba tập phải bằng đúng 17,509 ảnh
    all_files = train_files | val_files | test_files
    union_count = len(all_files)

    # 4. Mọi Filename đều tồn tại trong images_dir
    missing_files = []
    for f in all_files:
        if not (images_dir / f).exists():
            missing_files.append(f)
        if len(missing_files) > 10:  # Stop after 10 missing
            break

    # Print results
    print("=" * 60)
    print("SPLIT VERIFICATION RESULTS")
    print("=" * 60)
    print(f"\n1. Image counts per set:")
    print(f"   Train: {n_train} ({n_train/total*100:.1f}%)")
    print(f"   Val:   {n_val} ({n_val/total*100:.1f}%)")
    print(f"   Test:  {n_test} ({n_test/total*100:.1f}%)")
    print(f"   Total: {total}")

    print(f"\n   Per-class in TRAIN:")
    for label, count in train_class.items():
        print(f"      {CLASS_NAMES[label]}: {count}")
    print(f"\n   Per-class in VAL:")
    for label, count in val_class.items():
        print(f"      {CLASS_NAMES[label]}: {count}")
    print(f"\n   Per-class in TEST:")
    for label, count in test_class.items():
        print(f"      {CLASS_NAMES[label]}: {count}")

    print(f"\n2. Pairwise intersections (should all be 0):")
    print(f"   train ∩ val:   {train_val_overlap}")
    print(f"   train ∩ test:  {train_test_overlap}")
    print(f"   val ∩ test:    {val_test_overlap}")

    print(f"\n3. Union count: {union_count} (expected: 17509)")
    print(f"   {'PASS' if union_count == 17509 else 'FAIL'}")

    print(f"\n4. File existence check:")
    if len(missing_files) == 0:
        print(f"   All {union_count} files exist - PASS")
    else:
        print(f"   Missing {len(missing_files)} files (showing first 10):")
        for f in missing_files[:10]:
            print(f"      {f}")

    print("=" * 60)

    # Assert checks
    assert train_val_overlap == 0, f"train ∩ val not empty: {train_val_overlap}"
    assert train_test_overlap == 0, f"train ∩ test not empty: {train_test_overlap}"
    assert val_test_overlap == 0, f"val ∩ test not empty: {val_test_overlap}"
    assert union_count == 17509, f"Union count {union_count} != 17509"
    assert len(missing_files) == 0, f"Missing {len(missing_files)} files"

    return {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total},
        "per_class": {
            "train": train_class.to_dict(),
            "val": val_class.to_dict(),
            "test": test_class.to_dict(),
        },
        "overlap": {
            "train_val": train_val_overlap,
            "train_test": train_test_overlap,
            "val_test": val_test_overlap,
        },
        "union": union_count,
        "missing_files": len(missing_files),
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. aug options: 'basic', 'color', 'trivial', 'randaug'."""
    if train:
        # Training transforms
        if aug == "basic":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "color":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "trivial":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "randaug":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            raise ValueError(f"Unknown augmentation: {aug}")
    else:
        # Validation/test transforms
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),  # resize to 256 for 224 crop
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transform


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ images_dir theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row['Filename']
        label = int(row['Label'])

        # Load image
        from PIL import Image
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 0,
                seed: int = 0):
    """Tạo DataLoader với optional balanced sampler."""

    dataset = DeepWeedsDataset(df, images_dir, transform)

    # Worker init function to seed workers
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Sampler
    if train and sampler == "balanced":
        # Compute weights for balanced sampling
        labels = df['Label'].values
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        weights = 1.0 / class_counts[labels]
        sampler_obj = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    else:
        sampler_obj = None
        shuffle = train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        worker_init_fn=worker_init_fn,
    )

    return loader
