# scripts/eval_compare.py
import torch
from src.utils.config import load_config, set_seed, device_from_cfg
from src.datasets.pacs import build_loaders
from src.models.backbones import build_backbone, LinearHead
from src.methods.erm import ERM
from src.methods.irm import IRM

# ==== USER INPUTS ====
CONFIG_PATH = "configs/pacs_irm.yaml"
ERM_CKPT = "runs/erm_best.pt"       # or irm_lam0_seed1_wu5_best.pt
IRM_CKPT = "runs/irm_best.pt"  # change to your best IRM λ
# ======================

def load_model(method_name, ckpt_path, cfg, device):
    backbone, feat_dim = build_backbone(cfg["model"]["backbone"],
                                        pretrained=False,  # we load weights anyway
                                        freeze=cfg["model"].get("freeze_backbone", False))
    head = LinearHead(in_dim=feat_dim, num_classes=cfg["data"]["num_classes"])
    if method_name.lower() == "irm":
        model = IRM(backbone, head, cfg, device)
    else:
        model = ERM(backbone, head, cfg, device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.backbone.eval()
    model.head.eval()
    return model

def main():
    cfg = load_config(CONFIG_PATH)
    device = device_from_cfg(cfg)
    set_seed(cfg["training"]["seed"])

    loaders = build_loaders(
        root=cfg["data"]["root"],
        sources=cfg["data"]["sources"],
        target=cfg["data"]["target"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["training"]["seed"]
    )

    print(f"\nDevice: {device}\n")

    # Load models
    erm_model = load_model("erm", ERM_CKPT, cfg, device)
    irm_model = load_model("irm", IRM_CKPT, cfg, device)

    # Evaluate each
    results = []
    for dn, ld in loaders["test"]["per_domain"].items():
        erm_acc = erm_model.evaluate_loader(ld, f"ERM-{dn}")["acc"]
        irm_acc = irm_model.evaluate_loader(ld, f"IRM-{dn}")["acc"]
        results.append((dn, erm_acc, irm_acc))

    # Print pretty table
    print("\nPer-domain accuracy comparison:")
    print(f"{'Domain':10s} | {'ERM':>8s} | {'IRM':>8s}")
    print("-"*32)
    for dn, erm_acc, irm_acc in results:
        print(f"{dn:10s} | {erm_acc:8.3f} | {irm_acc:8.3f}")

    # Compute averages (sources + target)
    source_acc_erm = [r[1] for r in results if r[0] != cfg["data"]["target"]]
    source_acc_irm = [r[2] for r in results if r[0] != cfg["data"]["target"]]
    print("-"*32)
    print(f"{'Source avg':10s} | {sum(source_acc_erm)/len(source_acc_erm):8.3f} | {sum(source_acc_irm)/len(source_acc_irm):8.3f}")
    tgt = cfg["data"]["target"]
    tgt_erm = [r[1] for r in results if r[0]==tgt][0]
    tgt_irm = [r[2] for r in results if r[0]==tgt][0]
    print(f"{'Target ('+tgt+')':10s} | {tgt_erm:8.3f} | {tgt_irm:8.3f}")

if __name__ == "__main__":
    main()
