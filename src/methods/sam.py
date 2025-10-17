# src/methods/sam.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from src.losses.irm_penalty import cross_entropy

class SAMMethod:
    """
    Sharpness-Aware Minimization (SAM) applied to ERM:
    For each batch:
      1) compute gradients (nominal), perturb weights in gradient direction by rho,
      2) compute loss on perturbed weights, backprop, then restore.
    We log a simple sharpness proxy = (perturbed_loss - nominal_loss) on the batch.
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
        self.rho = float(cfg["method"].get("rho", 0.05))

    def _forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def _params(self):
        for m in (self.backbone, self.head):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def training_step(self, batch, batch_idx):
        x, y, d = batch
        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad(set_to_none=True)

        # 1) first pass: nominal gradients
        with autocast(enabled=self.cfg["training"].get("amp", True)):
            logits = self._forward(x)
            loss = cross_entropy(logits, y)
        self.scaler.scale(loss).backward()

        # compute epsilon norm (gradient norm in parameter space)
        grads = [p.grad for p in self._params() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2) if grads else torch.tensor(0., device=self.device)

        # perturb weights: w <- w + rho * grad / ||grad||
        epsilons = []
        for p in self._params():
            if p.grad is None: 
                epsilons.append(None)
            else:
                e = self.rho * p.grad.detach() / (grad_norm + 1e-12)
                epsilons.append(e)

    
        with torch.no_grad():
            for p, e in zip(self._params(), epsilons):
                if e is not None:
                    p.add_(e)

        # 2) second pass: gradient at perturbed point
        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg["training"].get("amp", True)):
            logits2 = self._forward(x)
            loss_perturbed = cross_entropy(logits2, y)
        self.scaler.scale(loss_perturbed).backward()
        self.scaler.step(self.opt); self.scaler.update()

        # restore weights
        with torch.no_grad():
            for p, e in zip(self._params(), epsilons):
                if e is not None:
                    p.sub_(e)

        # sharpness proxy for logging
        sharpness = (loss_perturbed.detach() - loss.detach())
        return {
            "loss": float(loss_perturbed.detach().item()),
            "erm": float(loss.detach().item()),
            "irm_penalty": 0.0,  # unify keys
            "coef": 0.0,
            "sharpness": float(sharpness.item())
        }

    @torch.no_grad()
    def evaluate_loader(self, loader, split_name):
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

    def load_state_dict(self, state):
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.opt.load_state_dict(state["opt"])
