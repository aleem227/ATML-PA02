# scripts/sweep_irm.py
import itertools, subprocess, sys, os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG  = PROJECT / "configs" / "pacs_irm.yaml"

# edit your grid here:
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]   # 0.0 = ERM baseline
SEEDS   = [1]
WARMUP  = 5  # fixed for now

def run_job(lmbda, seed):
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", str(CONFIG),
        "--lambda", str(lmbda),
        "--seed", str(seed),
        "--warmup", str(WARMUP),
        "--name_suffix", "sweep"
    ]
    os.environ["EPOCHS_OVERRIDE"] = "10"
    print(">>>", " ".join(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(PROJECT), env=env, check=True)

def main():
    for lmbda, seed in itertools.product(LAMBDAS, SEEDS):
        run_job(lmbda, seed)

if __name__ == "__main__":
    main()
