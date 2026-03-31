# experiments/run_all_rocket.py

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # same list for sktime
    datasets = ["ECG200", "GunPoint", "ItalyPowerDemand", "ElectricDevices"]

    # sktime ROCKET baselines: always 10k kernels (controlled inside the script)
    for ds in datasets:
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_sktime_baselines.py"),
                "--dataset",
                ds,
                "--results_dir",
                "experiments/results",
            ]
        )

    # original ROCKET: 10k for the first three, 2k for ElectricDevices
    n_kernels_map = {
        "ECG200": 10000,
        "GunPoint": 10000,
        "ItalyPowerDemand": 10000,
        "ElectricDevices": 2000,
    }

    for ds in datasets:
        n_kernels = n_kernels_map[ds]
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_rocket_original.py"),
                "--dataset",
                ds,
                "--n_kernels",
                str(n_kernels),
                "--results_dir",
                "experiments/results",
            ]
        )

    print("All ROCKET runs completed.")


if __name__ == "__main__":
    main()