import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

datasets = ["ECG200", "GunPoint"]  # extend as needed
n_kernels = 10000

def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

for ds in datasets:
    # sktime ROCKET
    run([
        "python", "experiments/run_sktime_baselines.py",
        "--dataset", ds,
        "--results_dir", str(RESULTS_DIR),
    ])

    # original ROCKET
    run([
        "python", "experiments/run_rocket_original.py",
        "--dataset", ds,
        "--n_kernels", str(n_kernels),
        "--results_dir", str(RESULTS_DIR),
    ])

# optionally, call a comparison script/notebook-export here
print("All runs finished.")