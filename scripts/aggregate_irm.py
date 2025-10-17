# scripts/aggregate_irm.py
import os, json, glob, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
OUT  = PROJECT / "runs" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

def load_runs():
    infos = []
    for p in RUNS.glob("irm_lam*_seed*_run_info.json"):
        with open(p, "r") as f:
            info = json.load(f)
        # parse lambda/seed from filename as backup
        stem = p.stem
        # expects irm_lam{λ}_seed{seed}_...
        parts = stem.split("_")
        lam = None; seed = None
        for t in parts:
            if t.startswith("lam"):
                lam = float(t.replace("lam", ""))
            if t.startswith("seed"):
                seed = int(t.replace("seed", ""))
        info["lambda_from_name"] = lam
        info["seed_from_name"] = seed
        infos.append(info)
    return infos

def summarize(infos):
    # collect per-run summary
    rows = []
    for info in infos:
        lam = info.get("lambda_from_name", info.get("lambda"))
        seed = info.get("seed_from_name", info.get("seed"))
        best_tgt = info.get("best_target_acc", np.nan)
        metrics_csv = info.get("metrics_csv")
        # read final epoch values:
        try:
            df = pd.read_csv(metrics_csv)
            final_pen = df["irm_penalty_avg"].dropna().values[-1] if "irm_penalty_avg" in df.columns else np.nan
            best_tgt_over_epochs = df["target_acc"].max() if "target_acc" in df.columns else best_tgt
        except Exception:
            final_pen = np.nan
            best_tgt_over_epochs = best_tgt
        rows.append({
            "lambda": lam,
            "seed": seed,
            "best_target_acc": best_tgt_over_epochs,
            "final_irm_penalty": final_pen
        })
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "irm_per_run.csv", index=False)

    # aggregate over seeds
    agg = per_run.groupby("lambda").agg(
        mean_target_acc=("best_target_acc","mean"),
        std_target_acc=("best_target_acc","std"),
        mean_final_pen=("final_irm_penalty","mean"),
        std_final_pen=("final_irm_penalty","std"),
        runs=("best_target_acc","count")
    ).reset_index().sort_values("lambda")
    agg.to_csv(OUT / "irm_by_lambda.csv", index=False)
    return per_run, agg

def plot_agg(agg: pd.DataFrame):
    # Plot A: lambda vs best target acc
    plt.figure()
    x = agg["lambda"].values
    y = agg["mean_target_acc"].values
    yerr = agg["std_target_acc"].fillna(0).values
    plt.errorbar(x, y, yerr=yerr, fmt='-o')
    plt.xlabel("IRM penalty λ")
    plt.ylabel("Best target accuracy (Sketch)")
    plt.title("IRM λ vs Target Acc (mean±std over seeds)")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT / "irm_lambda_vs_target_acc.png", dpi=150, bbox_inches="tight")

    # Plot B: lambda vs final IRM penalty
    plt.figure()
    y2 = agg["mean_final_pen"].values
    y2err = agg["std_final_pen"].fillna(0).values
    plt.errorbar(x, y2, yerr=y2err, fmt='-o')
    plt.xlabel("IRM penalty λ")
    plt.ylabel("Final IRM penalty (epoch-average)")
    plt.title("IRM λ vs Final Penalty (mean±std over seeds)")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT / "irm_lambda_vs_final_penalty.png", dpi=150, bbox_inches="tight")

def main():
    infos = load_runs()
    if not infos:
        print("No IRM run_info files found in runs/. Did you run the sweep?")
        return
    per_run, agg = summarize(infos)
    # Save pretty markdown table too
    with open(OUT / "irm_by_lambda.md", "w") as f:
        f.write("| lambda | target_acc mean | target_acc std | final_pen mean | final_pen std | runs |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for _, r in agg.iterrows():
            f.write(f"| {r['lambda']:.3g} | {r['mean_target_acc']:.3f} | {r['std_target_acc'] if not np.isnan(r['std_target_acc']) else 0:.3f} | "
                    f"{r['mean_final_pen'] if not np.isnan(r['mean_final_pen']) else 0:.3e} | {r['std_final_pen'] if not np.isnan(r['std_final_pen']) else 0:.1e} | {int(r['runs'])} |\n")
    plot_agg(agg)
    print(f"Saved:\n- {OUT/'irm_per_run.csv'}\n- {OUT/'irm_by_lambda.csv'}\n- {OUT/'irm_by_lambda.md'}\n- plots in {OUT}")

if __name__ == "__main__":
    main()
#python scripts\sweep_irm.py
