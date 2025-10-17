# src/utils/config.py
import yaml, os, random
import numpy as np
import torch

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_from_cfg(cfg):
    return torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("cuda", True) else "cpu")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
