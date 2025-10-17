# scripts/eval_compare_methods.py
import os, json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config, set_seed, device_from_cfg
from src.datasets.pacs import build_loaders
from src.models.backbones import build_backbone, LinearHead
from src.methods.erm import ERM
from src.methods.irm import IRM
from src.methods.groupdro import GroupDRO
from src.methods.sam import SAMMethod

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "runs" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

# ---- fill these with your filenames (or keep defaults) ----
ERM_CKPT  = PROJECT / "runs" / "erm_best.pt"
IRM_CKPT  = PROJECT / "runs" / "irm_lam1_seed42_wu5_best.pt"  # you picked λ=1
GDRO_CKPT = PROJECT / "runs" / "groupdro_best.pt"
SAM_CKPT  = PROJECT / "runs" / "sam_best.pt"
CFG_PATH  = PROJECT / "configs" / "pacs_erm.yaml"  # model/data params shared
# -----------------------------------------------------------

def load_model(kind, ckpt, cfg, device):
    bb, dim = build_backbone(cfg["model"]["backbone"], pretrained=False, freeze=cfg["model"].get("freeze_backbone", False))
    head = LinearHead(dim, cfg["data"]["num_classes"])
    if kind=="erm": m = ERM(bb, head, cfg, device)
    elif kind=="irm":
        if "method" not in cfg: cfg["method"]={"name":"irm","lambda":1.0,"penalty_warmup":5}
        if "lambda" not in cfg["method"]: cfg["method"]["lambda"]=1.0
        m = IRM(bb, head, cfg, device)
    elif kind=="groupdro": m = GroupDRO(bb, head, cfg, device)
    elif kind=="sam": m = SAMMethod(bb, head, cfg, device)
    else: raise ValueError(kind)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state)
    m.backbone.eval(); m.head.eval()
    return m

def eval_method(name, model, loaders, cfg):
    per_dom = {}
    for dn in ["Art","Cartoon","Photo","Sketch"]:
        r = model.evaluate_loader(loaders["test"]["per_domain"][dn], dn)
        per_dom[dn] = r["acc"]
    src_avg = sum(per_dom[d] for d in cfg["data"]["sources"]) / len(cfg["data"]["sources"])
    worst_src = min(per_dom[d] for d in cfg["data"]["sources"])
    tgt = per_dom[cfg["data"]["target"]]
    return {"name":name, "per_domain":per_dom, "source_avg":src_avg, "worst_source":worst_src, "target":tgt}

def barplot_target(results):
    names = [r["name"] for r in results]
    vals  = [r["target"] for r in results]
    plt.figure()
    plt.bar(names, vals)
    plt.ylabel("Target (Sketch) accuracy")
    plt.title("Target accuracy by method")
    plt.ylim(0,1)
    plt.savefig(OUT / "methods_target_bar.png", dpi=150, bbox_inches="tight")

def barplot_worst(results):
    names = [r["name"] for r in results]
    vals  = [r["worst_source"] for r in results]
    plt.figure()
    plt.bar(names, vals)
    plt.ylabel("Worst-source (Art/Cartoon/Photo) accuracy")
    plt.title("Worst-source accuracy by method")
    plt.ylim(0,1)
    plt.savefig(OUT / "methods_source_worst_bar.png", dpi=150, bbox_inches="tight")

def main():
    cfg = load_config(str(CFG_PATH))
    device = device_from_cfg(cfg); set_seed(cfg["training"]["seed"])
    loaders = build_loaders(root=cfg["data"]["root"], sources=cfg["data"]["sources"],
                            target=cfg["data"]["target"], img_size=cfg["data"]["img_size"],
                            batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"],
                            seed=cfg["training"]["seed"])

    models = [
        ("ERM",    ERM_CKPT),
        ("IRM(λ=1)", IRM_CKPT),
        ("GroupDRO", GDRO_CKPT),
        ("SAM",    SAM_CKPT),
    ]

    results = []
    for name, ckpt in models:
        if not Path(ckpt).exists():
            print(f"[warn] missing {name} checkpoint at {ckpt}, skipping")
            continue
        kind = "erm" if name.startswith("ERM") else ("irm" if name.startswith("IRM") else ("groupdro" if "GroupDRO" in name else "sam"))
        r = eval_method(name, load_model(kind, ckpt, cfg, device), loaders, cfg)
        results.append(r)

    if not results:
        print("No results to aggregate.")
        return

    # table
    rows = []
    for r in results:
        row = {"method": r["name"], "target": r["target"], "source_avg": r["source_avg"], "worst_source": r["worst_source"]}
        row.update({f"acc_{k}": v for k,v in r["per_domain"].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "compare_methods.csv", index=False)
    with open(OUT / "compare_methods.md","w") as f:
        f.write("| method | target | source_avg | worst_source | Art | Cartoon | Photo | Sketch |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _,r in df.iterrows():
            f.write(f"| {r['method']} | {r['target']:.3f} | {r['source_avg']:.3f} | {r['worst_source']:.3f} | {r['acc_Art']:.3f} | {r['acc_Cartoon']:.3f} | {r['acc_Photo']:.3f} | {r['acc_Sketch']:.3f} |\n")

    barplot_target(results); barplot_worst(results)
    print("Saved tables/plots to", OUT)

if __name__ == "__main__":
    main()
