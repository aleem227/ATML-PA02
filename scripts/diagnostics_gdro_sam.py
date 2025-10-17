# scripts/diagnostics_gdro_sam.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
OUT  = PROJECT / "runs" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

def plot_groupdro_curves():
    # Find the most recent groupdro_metrics.csv
    cands = sorted(RUNS.glob("groupdro*_metrics.csv"))
    if not cands: 
        print("[warn] no GroupDRO metrics csv found"); return
    csvp = cands[-1]
    df = pd.read_csv(csvp)
    # Worst-source curve was logged each epoch
    if "worst_source_acc" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["worst_source_acc"], label="GroupDRO")
        plt.xlabel("epoch"); plt.ylabel("worst-source accuracy")
        plt.title("Worst-source accuracy across epochs (GroupDRO)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.ylim(0,1)
        plt.savefig(OUT / "groupdro_worst_source_curve.png", dpi=150, bbox_inches="tight")
    # weights
    for k in ["w_Art","w_Cartoon","w_Photo"]:
        if k not in df.columns: 
            print("[warn] no group weight columns found"); return
    plt.figure()
    plt.plot(df["epoch"], df["w_Art"], label="Art")
    plt.plot(df["epoch"], df["w_Cartoon"], label="Cartoon")
    plt.plot(df["epoch"], df["w_Photo"], label="Photo")
    plt.xlabel("epoch"); plt.ylabel("group weight q")
    plt.title("Group weights (q) during training (GroupDRO)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(OUT / "groupdro_group_weights.png", dpi=150, bbox_inches="tight")

def plot_sam_sharpness():
    cands = sorted(RUNS.glob("sam*_metrics.csv"))
    if not cands: 
        print("[warn] no SAM metrics csv found"); return
    csvp = cands[-1]
    df = pd.read_csv(csvp)
    if "sharpness" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["sharpness"])
        plt.xlabel("epoch"); plt.ylabel("sharpness proxy (perturbed - nominal loss)")
        plt.title("SAM sharpness proxy across epochs")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUT / "sam_sharpness_curve.png", dpi=150, bbox_inches="tight")

def main():
    plot_groupdro_curves()
    plot_sam_sharpness()
    print("Saved diagnostics to", OUT)

if __name__ == "__main__":
    main()
