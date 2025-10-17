# src/datasets/pacs.py
from __future__ import annotations
from typing import Dict, List
import os, json, random
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms

DOMAINS = ["Art", "Cartoon", "Photo", "Sketch"]


# -----------------------------
# Transforms
# -----------------------------
def _tfms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# -----------------------------
# Dataset wrapper (adds domain_id)
# -----------------------------
class PACSDomainDataset(datasets.ImageFolder):
    """A single PACS domain folder with domain_id exposed."""
    def __init__(self, root, domain_name, transform=None):
        super().__init__(os.path.join(root, domain_name), transform=transform)
        self.domain_name = domain_name
        self.domain_id = DOMAINS.index(domain_name)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, self.domain_id


# -----------------------------
# Split utils
# -----------------------------
@dataclass
class SplitCfg:
    root: str                 # e.g., data/PACS
    sources: List[str]        # e.g., ["Art","Cartoon","Photo"]
    target: str               # e.g., "Sketch"
    val_ratio: float = 0.2
    seed: int = 42
    split_dir: str | None = None  # defaults to data/splits

    @property
    def path(self) -> str:
        sd = self.split_dir or os.path.join(os.path.dirname(self.root), "splits")
        os.makedirs(sd, exist_ok=True)
        # name encodes which domain is target to avoid collisions
        name = f"pacs_splits_target-{self.target}_seed-{self.seed}_val-{int(self.val_ratio*100)}.json"
        return os.path.join(sd, name)


def _make_or_load_splits(cfg: SplitCfg, img_size: int) -> Dict[str, Dict[str, List[int]]]:
    """
    Returns a dict like:
      {
        "Art": {"train": [..idx..], "val": [..idx..]},
        "Cartoon": {...},
        "Photo": {...},
        "Sketch": {"train": [], "val": []}   # target has no train/val; kept for completeness
      }
    Persisted on disk for reproducibility.
    """
    path = cfg.path
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # create deterministic splits per SOURCE domain
    random.seed(cfg.seed)
    splits = {}
    for dn in DOMAINS:
        ds = PACSDomainDataset(cfg.root, dn, transform=_tfms(img_size, train=False))
        idxs = list(range(len(ds)))
        if dn in cfg.sources:
            random.shuffle(idxs)
            k = int(len(idxs) * (1.0 - cfg.val_ratio))
            train_idx, val_idx = idxs[:k], idxs[k:]
        else:
            train_idx, val_idx = [], []  # target: no training/validation indices
        splits[dn] = {"train": train_idx, "val": val_idx}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    return splits


# -----------------------------
# Public builder
# -----------------------------
def build_loaders(root: str,
                  sources: List[str],
                  target: str,
                  img_size: int = 224,
                  batch_size: int = 32,
                  num_workers: int = 4,
                  val_ratio: float = 0.2,
                  seed: int = 42) -> Dict[str, Dict[str, DataLoader]]:
    """
    Creates train/val for sources with a saved deterministic split,
    test loader for the held-out target, and per-domain eval loaders.
    """
    assert target in DOMAINS, f"Unknown target {target}"
    for s in sources:
        assert s in DOMAINS and s != target, f"Bad source {s}"

    # create / load splits
    split_cfg = SplitCfg(root=root, sources=sources, target=target,
                         val_ratio=val_ratio, seed=seed)
    splits = _make_or_load_splits(split_cfg, img_size)

    # build datasets
    train_subsets, val_subsets = [], []
    for dn in sources:
        full_ds_train = PACSDomainDataset(root, dn, transform=_tfms(img_size, train=True))
        full_ds_val   = PACSDomainDataset(root, dn, transform=_tfms(img_size, train=False))
        train_subsets.append(Subset(full_ds_train, splits[dn]["train"]))
        val_subsets.append(Subset(full_ds_val,   splits[dn]["val"]))

    train_ds = ConcatDataset(train_subsets)
    val_ds   = ConcatDataset(val_subsets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # target test set (entire domain)
    test_loader  = DataLoader(
        PACSDomainDataset(root, target, transform=_tfms(img_size, train=False)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # per-domain evaluation (full domains, eval transforms)
    per_domain_eval = {
        dn: DataLoader(PACSDomainDataset(root, dn, transform=_tfms(img_size, train=False)),
                       batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
        for dn in DOMAINS
    }

    return {
        "train": {"sources_merged": train_loader},
        "val":   {"sources_merged": val_loader},
        "test":  {"target": test_loader, "per_domain": per_domain_eval},
        "meta":  {"splits_file": split_cfg.path}
    }
