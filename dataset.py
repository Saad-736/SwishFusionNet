"""
dataset.py
----------
Food-101 loading utilities with optional class subsetting.

Uses a plain Dataset wrapper (not Subset subclass) to avoid PyTorch's
internal __getitems__ check on Subset subclasses.
"""

import random
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Food101


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Plain Dataset wrapper (no Subset subclassing -> no __getitems__ requirement)
# ---------------------------------------------------------------------------
class FoodSubset(Dataset):
    def __init__(self, base_dataset: Food101, indices: List[int],
                 label_map: dict, class_names: List[str]):
        self.base = base_dataset
        self.indices = indices
        self.label_map = label_map
        self.classes = class_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, lbl = self.base[self.indices[idx]]
        return img, self.label_map[lbl]


def _pick_subset_classes(num_classes: int, seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    all_idx = list(range(101))
    rng.shuffle(all_idx)
    return sorted(all_idx[:num_classes])


def _build_subset(dataset: Food101, keep_classes: List[int]) -> Tuple[FoodSubset, dict]:
    keep_set = set(keep_classes)
    new_label_map = {old: new for new, old in enumerate(keep_classes)}
    selected_indices = [i for i, lbl in enumerate(dataset._labels) if lbl in keep_set]
    class_names = [dataset.classes[c] for c in keep_classes]

    subset = FoodSubset(dataset, selected_indices, new_label_map, class_names)
    info = {
        "kept_class_indices": keep_classes,
        "kept_class_names": class_names,
        "num_classes": len(keep_classes),
        "num_samples": len(selected_indices),
    }
    return subset, info


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def get_food101_loaders(
    root: str = "./data",
    img_size: int = 224,
    batch_size: int = 64,
    num_classes: int = 20,
    num_workers: int = 2,
    seed: int = 42,
):
    """Returns: train_loader, test_loader, info_dict"""
    assert 1 <= num_classes <= 101, "num_classes must be between 1 and 101"
    Path(root).mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(img_size)
    train_full = Food101(root=root, split="train", transform=train_tf, download=True)
    test_full  = Food101(root=root, split="test",  transform=eval_tf,  download=True)

    if num_classes == 101:
        train_ds, test_ds = train_full, test_full
        info = {
            "num_classes": 101,
            "num_train_samples": len(train_full),
            "num_test_samples": len(test_full),
            "kept_class_names": train_full.classes,
        }
    else:
        keep = _pick_subset_classes(num_classes, seed=seed)
        train_ds, info_tr = _build_subset(train_full, keep)
        test_ds,  info_te = _build_subset(test_full,  keep)
        info = {
            "num_classes": num_classes,
            "num_train_samples": info_tr["num_samples"],
            "num_test_samples":  info_te["num_samples"],
            "kept_class_names":  info_tr["kept_class_names"],
        }

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=g, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, info


if __name__ == "__main__":
    train_loader, test_loader, info = get_food101_loaders(num_classes=5)
    print(info)
    x, y = next(iter(train_loader))
    print("Batch:", x.shape, y.shape, "labels:", y[:5].tolist())
