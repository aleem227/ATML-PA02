# src/methods/erm.py
from typing import Dict
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from src.losses.irm_penalty import cross_entropy

class ERM:
    def __init__(self, backbone, head, cfg, device):
        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.device = device
        self.cfg = cfg
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.opt = AdamW([p for p in params if p.requires_grad],
                         lr=cfg["training"]["lr"], weight_decay=cfg["training"]["wd"])
        self.scaler = GradScaler(enabled=cfg["training"].get("amp", True))
        self.num_classes = cfg["data"]["num_classes"]

    def _forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, d = batch  # d is domain_id (unused in ERM)
        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg["training"].get("amp", True)):
            logits = self._forward(x)
            loss = cross_entropy(logits, y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt); self.scaler.update()
        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate_loader(self, loader, split_name):
        import torch.nn.functional as F
        self.backbone.eval(); self.head.eval()
        total, correct, running_loss = 0, 0, 0.0
        for x, y, d in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self._forward(x)
            running_loss += cross_entropy(logits, y).item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        self.backbone.train(); self.head.train()
        return {"split": split_name, "loss": running_loss/total, "acc": correct/total}

    def state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.opt.load_state_dict(state["opt"])
