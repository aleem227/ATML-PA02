# scripts/plot_groupdro_metrics.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_groupdro_csv(runs_dir: Path) -> Path:
    cands = sorted(runs_dir.glob("groupdro*_metrics.csv"))
    if not cands:
        raise FileNotFoundError(f"No GroupDRO metrics found under: {runs_dir}")
    return cands[-1]

def plot_target(df: pd.DataFrame, out: Path):
    if not {"epoch", "target_acc"}.issubset(df.columns):
        print("[warn] target plot skipped (columns missing)")
        return
    plt.figure()
    plt.plot(df["epoch"], df["target_acc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Target (Sketch) accuracy")
    plt.title("GroupDRO: Target accuracy over epochs")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    fp = out / "groupdro_target_curve.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fp}")

def plot_worst_source(df: pd.DataFrame, out: Path):
    if not {"epoch", "worst_source_acc"}.issubset(df.columns):
        print("[warn] worst-source plot skipped (columns missing)")
        return
    plt.figure()
    plt.plot(df["epoch"], df["worst_source_acc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-source accuracy (min of Art/Cartoon/Photo)")
    plt.title("GroupDRO: Worst-source accuracy over epochs")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    fp = out / "groupdro_worst_source_curve.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fp}")

def plot_group_weights(df: pd.DataFrame, out: Path):
    cols = ["w_Art", "w_Cartoon", "w_Photo"]
    if not {"epoch", *cols}.issubset(df.columns):
        print("[warn] group-weights plot skipped (columns missing)")
        return
    plt.figure()
    for c in cols:
        plt.plot(df["epoch"], df[c], label=c.replace("w_", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Group weight q")
    plt.title("GroupDRO: Domain weight trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fp = out / "groupdro_group_weights.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fp}")

# scripts/plot_groupdro_metrics_combined.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path to groupdro *_metrics.csv")
    ap.add_argument("--outdir", type=str, default="runs/summary", help="Output directory")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else sorted((project / "runs").glob("groupdro*_metrics.csv"))[-1]
    out_dir = project / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError("CSV must contain 'epoch' column")

    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)

    # --- Combined plot: Target vs Worst-source ---
    plt.figure(figsize=(7,5))
    plt.plot(df["epoch"], df["target_acc"], marker="o", label="Target (Sketch)")
    if "worst_source_acc" in df.columns:
        plt.plot(df["epoch"], df["worst_source_acc"], marker="s", label="Worst Source (min Art/Cartoon/Photo)")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("GroupDRO: Target vs Worst-Source Accuracy Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.5, 1.0)   # Focused y-axis range for readability
    plt.tight_layout()

    fp = out_dir / "groupdro_target_vs_worstsource.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fp}")

if __name__ == "__main__":
    main()

