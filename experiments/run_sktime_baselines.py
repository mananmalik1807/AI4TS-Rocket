# experiments/run_sktime_baselines.py

import os
import sys
from pathlib import Path

# Make src/ importable when running this as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from experiments_sktime import run_sktime_rocket_experiment


if __name__ == "__main__":
    datasets = ["ItalyPowerDemand", "ECG200", "GunPoint", "ElectricDevices"]

    results = []
    for name in datasets:
        print(f"Running ROCKET (sktime) on {name} ...")
        res = run_sktime_rocket_experiment(
            name,
            num_kernels=10000,
            random_state=42,
            project_root=PROJECT_ROOT,
        )
        print(f"  -> accuracy = {res['accuracy']:.4f}, saved to {res['results_path']}")
        results.append(res)

    print("Done.")