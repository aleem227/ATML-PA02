# src/methods/irm.py
from typing import Dict
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from src.losses.irm_penalty import cross_entropy, irmv1_penalty

class IRM:
    """
    Minimal IRM-v1: ERM loss + λ * sum_i ||∂R_i/∂s|_{s=1}||^2
    We approximate domain-wise penalties inside each batch by splitting on domain_id.
    """
    def __init__(self, backbone, head, cfg, device):
        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.device = device
        self.cfg = cfg
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.opt = AdamW([p for p in params if p.requires_grad],
                         lr=cfg["training"]["lr"], weight_decay=cfg["training"]["wd"])
        self.scaler = GradScaler(enabled=cfg["training"].get("amp", True))
        self.lambda_irm = cfg["method"]["lambda"]
        self.warmup_epochs = cfg["method"].get("penalty_warmup", 0)
        self.epoch = 0

    def _forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def set_epoch(self, e: int):
        self.epoch = e

    def training_step(self, batch, batch_idx):
        x, y, d = batch  # d: domain ids in the mixed batch
        x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)
        self.opt.zero_grad(set_to_none=True)

        with autocast(enabled=self.cfg["training"].get("amp", True)):
            logits = self._forward(x)
            erm_loss = cross_entropy(logits, y)

            # compute per-domain penalty by masking
            domains = d.unique().tolist()
            penalty = 0.0
            for dom in domains:
                m = (d == dom)
                if m.sum() < 2:  # skip tiny subsets for stability
                    continue
                penalty = penalty + irmv1_penalty(logits[m], y[m])

            penalty = penalty / max(1, len(domains))
            coef = 0.0 if self.epoch < self.warmup_epochs else self.lambda_irm
            loss = erm_loss + coef * penalty

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt); self.scaler.update()
        return {
            "loss": loss.item(),
            "erm": erm_loss.item(),
            "irm_penalty": penalty.item() if isinstance(penalty, torch.Tensor) else float(penalty),
            "coef": coef,
        }

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
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.opt.load_state_dict(state["opt"])
        self.epoch = state.get("epoch", 0)
