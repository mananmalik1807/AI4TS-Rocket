# ROCKET Reproduction and Extensions

This repository reproduces and analyses the ROCKET method for time series classification, and compares the original ROCKET implementation with the sktime ROCKET transform on selected UCR datasets.

---

## Repository structure

```text
.
├── data/
│   └── ItalyPowerDemand/
│       ├── ItalyPowerDemand_TRAIN.ts
│       └── ItalyPowerDemand_TEST.ts
│
├── experiments/
│   ├── results/
│   │   ├── ecg200_rocket_original.csv
│   │   ├── ecg200_rocket_sktime.csv
│   │   ├── electricdevices_rocket_original.csv
│   │   ├── electricdevices_rocket_sktime.csv
│   │   ├── gunpoint_rocket_original.csv
│   │   ├── gunpoint_rocket_sktime.csv
│   │   ├── italypowerdemand_rocket_original.csv
│   │   ├── italypowerdemand_rocket_sktime.csv
│   │   ├── rocket_sktime_vs_original_raw.csv
│   │   └── rocket_sktime_vs_original_relative.csv
│   │
│   ├── run_all_rocket.py
│   ├── run_rocket_original.py
│   └── run_sktime_baselines.py
│
├── notebooks/
│   ├── reproduction/
│   │   ├── 00_sktime_rocket_baseline_summary.ipynb
│   │   ├── 01_reproduce_rocket_italypower.ipynb
│   │   ├── 02_reproduce_rocket_ecg200.ipynb
│   │   ├── 03_reproduce_rocket_gunpoint.ipynb
│   │   ├── 04_reproduce_rocket_electricdevices.ipynb
│   │   └── 05_compare_rocket_sktime_vs_original_ecg200.ipynb
│   └── improvements/
│       └── (reserved for future improvement experiments)
│
├── src/
│   ├── __pycache__/
│   ├── rocket_original/
│   │   ├── __init__.py
│   │   └── ...  # original ROCKET implementation
│   ├── experiments_rocket_original.py
│   └── experiments_sktime.py
│
├── README.md
└── environment.yml  (optional)
```

### Data

- `data/ItalyPowerDemand/` contains the ItalyPowerDemand dataset files used in the reproduction notebooks.  
- Additional UCR datasets (e.g. ECG200, GunPoint, ElectricDevices) are downloaded automatically via sktime or loaded from local paths, depending on how you configure `experiments_sktime.py` and `experiments_rocket_original.py`.

### Experiments scripts

The `experiments/` folder contains small driver scripts that call into `src/` and write CSV outputs under `experiments/results/`.

- `experiments/run_sktime_baselines.py`  
  Runs the sktime ROCKET + ridge baselines on the configured UCR datasets, using helper functions from `src/experiments_sktime.py` and saving one CSV per dataset (e.g. `ecg200_rocket_sktime.csv`).

- `experiments/run_rocket_original.py`  
  Runs the original ROCKET implementation (the paper-faithful transform) on the same datasets, using `src/rocket_original/` and `src/experiments_rocket_original.py`, and saves matching CSVs (e.g. `ecg200_rocket_original.csv`).

- `experiments/run_all_rocket.py`  
  Convenience wrapper that calls both of the above so that all required `*_rocket_sktime.csv` and `*_rocket_original.csv` files are present under `experiments/results/`.

The `experiments/results/` directory then contains per-dataset results and two aggregated comparison files:

- `rocket_sktime_vs_original_raw.csv`: one row per (dataset, implementation) with accuracy and runtime.  
- `rocket_sktime_vs_original_relative.csv`: relative metrics such as accuracy differences and time ratios between original ROCKET and sktime ROCKET.

### Source code

The `src/` directory contains the reusable experiment code used by both the scripts and the notebooks.

```text
src/
├── __pycache__/
├── rocket_original/
│   ├── __init__.py
│   └── ...  # original ROCKET implementation (kernels + transform)
├── experiments_rocket_original.py
└── experiments_sktime.py
```

- `src/rocket_original/`  
  Reimplementation of the original ROCKET transform, including random kernel generation, the transform itself, and integration with a linear classifier as described in the paper.

- `src/experiments_sktime.py`  
  Helper functions to run sktime ROCKET experiments (e.g. loading a UCR dataset, applying `sktime.transformations.panel.rocket.Rocket`, fitting a ridge classifier, timing, and writing a `*_rocket_sktime.csv` result file).

- `src/experiments_rocket_original.py`  
  Matching helpers for the original ROCKET implementation, used by `run_rocket_original.py` and the reproduction notebooks to generate `*_rocket_original.csv` files.

### Notebooks

Reproduction and analysis are organised under `notebooks/`.

- `notebooks/reproduction/00_sktime_rocket_baseline_summary.ipynb`  
  Summarises sktime ROCKET baselines by loading all `*_rocket_sktime.csv` files from `experiments/results/` and computing aggregate statistics.

- `notebooks/reproduction/01_reproduce_rocket_italypower.ipynb`  
  Runs and analyses ROCKET on the ItalyPowerDemand dataset.

- `notebooks/reproduction/02_reproduce_rocket_ecg200.ipynb`  
  Runs and analyses ROCKET on ECG200.

- `notebooks/reproduction/03_reproduce_rocket_gunpoint.ipynb`  
  Runs and analyses ROCKET on GunPoint.

- `notebooks/reproduction/04_reproduce_rocket_electricdevices.ipynb`  
  Runs and analyses ROCKET on ElectricDevices.

- `notebooks/reproduction/05_compare_rocket_sktime_vs_original_ecg200.ipynb`  
  Loads the per-dataset CSVs, builds `comparison_df` and `rel_df` (raw and relative comparison tables), and writes `rocket_sktime_vs_original_raw.csv` and `rocket_sktime_vs_original_relative.csv` under `experiments/results/`.

- `notebooks/improvements/`  
  Reserved for experiments that extend or improve on the original ROCKET setup (e.g. varying number of kernels, MiniRocket/MultiRocket baselines, alternative classifiers).

---

## Environment setup

All commands below assume you are in the repository root.

### Create and activate the environment

Create a conda environment named `myenv` and install Python:

```bash
conda create -n myenv python=3.11
conda activate myenv
```

If you use the provided `environment.yml`, create the environment with:

```bash
conda env create -n myenv -f environment.yml
conda activate myenv
```

Otherwise, install the core dependencies manually (minimal example):

```bash
pip install numpy pandas scikit-learn sktime jupyter
```

Make sure `myenv` is activated whenever you run experiments or notebooks.

---

## Running experiments from the command line

These commands regenerate all CSV results under `experiments/results/`.[file:27]

### 1. Run sktime ROCKET baselines

```bash
conda activate myenv
python experiments/run_sktime_baselines.py
```

This will create files such as:

- `experiments/results/ecg200_rocket_sktime.csv`  
- `experiments/results/gunpoint_rocket_sktime.csv`  
- `experiments/results/italypowerdemand_rocket_sktime.csv`  
- `experiments/results/electricdevices_rocket_sktime.csv`  

Each CSV typically includes dataset name, number of kernels, train/test sizes, accuracy, and timing columns.[file:27]

### 2. Run original ROCKET implementation

```bash
conda activate myenv
python experiments/run_rocket_original.py
```

This will create the corresponding original-implementation files:

- `experiments/results/ecg200_rocket_original.csv`  
- `experiments/results/gunpoint_rocket_original.csv`  
- `experiments/results/italypowerdemand_rocket_original.csv`  
- `experiments/results/electricdevices_rocket_original.csv`

### 3. Run all experiments in one go (recommended)

```bash
conda activate myenv
python experiments/run_all_rocket.py
```

This wrapper ensures that all `*_rocket_sktime.csv` and `*_rocket_original.csv` files exist so that the analysis notebooks can be executed without manual steps.

---

## Running the notebooks

After running the experiment scripts at least once, you can reproduce all analysis and plots from the notebooks.

1. Start Jupyter/code:

   ```bash
   conda activate myenv
   jupyter lab
   ```

2. Open notebooks under `notebooks/reproduction/` in the following order if you want the full story:
   - `00_sktime_rocket_baseline_summary.ipynb`  
   - `01_reproduce_rocket_italypower.ipynb`  
   - `02_reproduce_rocket_ecg200.ipynb`  
   - `03_reproduce_rocket_gunpoint.ipynb`  
   - `04_reproduce_rocket_electricdevices.ipynb`  
   - `05_compare_rocket_sktime_vs_original_ecg200.ipynb`  

3. The last notebook writes:
   - `experiments/results/rocket_sktime_vs_original_raw.csv`  
   - `experiments/results/rocket_sktime_vs_original_relative.csv`  

These aggregated CSVs are used in the project report to discuss accuracy–runtime trade-offs between the original ROCKET implementation and the sktime ROCKET baseline.

---

## Extending the project

To add new experiments or improvements:

- Add new helper functions under `src/` (e.g. new experiment runners or additional transforms).  
- Add new driver scripts under `experiments/` if you want reproducible command-line entry points.
- Place new analysis notebooks under `notebooks/improvements/` and follow the same pattern of reading and writing CSVs under `experiments/results/`.