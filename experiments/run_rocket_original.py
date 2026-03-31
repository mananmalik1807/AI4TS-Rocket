# experiments/run_rocket_original.py

from pathlib import Path
import argparse
import sys

# add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from experiments_rocket_original import run_original_rocket_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (e.g. ECG200, GunPoint) or 'all' for a fixed list.",
    )
    parser.add_argument(
        "--n_kernels",
        type=int,
        default=10000,
        help="Number of random convolution kernels.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for kernel generation and classifier.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Directory where result CSVs will be written (must match experiment code).",
    )
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        datasets = ["ECG200", "GunPoint"]
    else:
        datasets = [args.dataset]

    for name in datasets:
        print(f"Running original ROCKET on {name} ...")
        res = run_original_rocket_experiment(
            dataset_name=name,
            num_kernels=args.n_kernels,
            random_state=args.random_state,
            project_root=PROJECT_ROOT,
        )
        # Note: run_original_rocket_experiment already writes CSV under project_root/experiments/results.
        # If args.results_dir differs, move/rename here as needed.
        print(res)

    print("Done.")


if __name__ == "__main__":
    main()