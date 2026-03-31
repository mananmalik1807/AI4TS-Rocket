# experiments/run_sktime_baselines.py

from pathlib import Path
import argparse
import sys

# add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from experiments_sktime import run_sktime_rocket_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (e.g. ECG200, GunPoint) or 'all' for a fixed list.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Directory where result CSVs will be written.",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir

    if args.dataset.lower() == "all":
        datasets = ["ECG200", "GunPoint", "ItalyPowerDemand", "ElectricDevices"]
    else:
        datasets = [args.dataset]

    for name in datasets:
        print(f"Running sktime ROCKET baseline on {name} ...")
        res = run_sktime_rocket_experiment(
            dataset_name=name,
            num_kernels=10000,
            random_state=42,
            project_root=PROJECT_ROOT,
        )
        # ensure results go to args.results_dir if that differs
        out_path = results_dir / f"{name.lower()}_rocket_sktime.csv"
        default_path = PROJECT_ROOT / res["results_path"]
        if out_path != default_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            default_path.rename(out_path)
        print(res)

    print("Done.")


if __name__ == "__main__":
    main()