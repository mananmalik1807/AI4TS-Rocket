# paper-accurate ROCKET for one dataset.
# experiments/run_rocket_original.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from experiments_rocket_original import run_original_rocket_experiment


if __name__ == "__main__":
    datasets = ["ECG200", "GunPoint"]

    for name in datasets:
        print(f"Running original ROCKET on {name} ...")
        res = run_original_rocket_experiment(
            dataset_name=name,
            num_kernels=10000,
            random_state=42,
            project_root=PROJECT_ROOT,
        )
        print(res)

    print("Done.")