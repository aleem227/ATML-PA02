# src/methods/groupdro.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from src.losses.irm_penalty import cross_entropy

class GroupDRO:
    """
    Group DRO with domains as groups.
    At each step: compute per-domain losses, upweight the largest loss via
    exponentiated-gradient updates on q (group weights).
    Objective: sum_i q_i * R_i  (minimax over groups).
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

        self.eta = float(cfg["method"].get("eta", 0.1))  # step for weight updates
        # initialize group weights uniformly over 3 sources (Art/Cartoon/Photo)
        self.domain_names = cfg["data"]["sources"]
        self.num_groups = len(self.domain_names)
        self.q = torch.ones(self.num_groups, device=device) / self.num_groups

    def _forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def _domain_indices(self, d):
        # map batch domain ids to [0..num_groups-1] within sources order
        # d are global domain ids; we convert to local indices by name
        # we do it via a name->id map in train loop, but here assume d already local
        return d

    def training_step(self, batch, batch_idx):
        x, y, d = batch  # d: global domain ids, but our train loader mixes only sources
        x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)

        # Map global ids to local group indices based on order in cfg["data"]["sources"]
        # Build a lookup tensor: global_id -> local_index or -1 for target
        # (global ids: Art=0, Cartoon=1, Photo=2, Sketch=3 in our dataset code)
        name_to_global = {"Art":0, "Cartoon":1, "Photo":2, "Sketch":3}
        mapping = torch.full((4,), -1, dtype=torch.long, device=self.device)
        for li, name in enumerate(self.domain_names):
            mapping[name_to_global[name]] = li
        local = mapping[d]

        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg["training"].get("amp", True)):
            logits = self._forward(x)
            losses = []
            group_loss_vec = torch.zeros(self.num_groups, device=self.device)
            for gi in range(self.num_groups):
                m = (local == gi)
                if m.any():
                    li = cross_entropy(logits[m], y[m])
                    group_loss_vec[gi] = li
                    losses.append(li)
            # Group DRO objective
            loss = (self.q * group_loss_vec).sum()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt); self.scaler.update()

        # exponentiated-gradient update on q using detached group losses
        with torch.no_grad():
            # only update using groups that appeared in batch
            mask = (group_loss_vec > 0)
            if mask.any():
                g = torch.zeros_like(self.q)
                g[mask] = group_loss_vec[mask]
                self.q = self.q * torch.exp(self.eta * g)
                self.q = self.q / self.q.sum()

        # return lightweight stats
        stats = {"loss": float(loss.detach().item())}
        for i, name in enumerate(self.domain_names):
            stats[f"loss_{name}"] = float(group_loss_vec[i].item())
            stats[f"w_{name}"] = float(self.q[i].item())
        # For unified logger keys:
        stats["irm_penalty"] = 0.0
        stats["coef"] = 0.0
        return stats

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
            "q": self.q.detach().cpu().numpy().tolist(),
            "domain_names": self.domain_names,
        }

    def load_state_dict(self, state):
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.opt.load_state_dict(state["opt"])
        if "q" in state:
            import torch
            self.q = torch.tensor(state["q"], device=self.device, dtype=torch.float)
        self.domain_names = state.get("domain_names", self.domain_names)

    # used by trainer to log weights
    def get_group_weights(self):
        return {name: float(self.q[i].item()) for i, name in enumerate(self.domain_names)}
