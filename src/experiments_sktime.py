# reusable experiment function
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import Rocket

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score


def run_sktime_rocket_experiment(
    dataset_name: str,
    num_kernels: int = 10000,
    random_state: int = 42,
    project_root: str | Path | None = None,
):
    """Run ROCKET (sktime) + ridge on a single UCR dataset and return metrics.

    Timing:
    - total_time_sec: from start of data loading through prediction.
    - fit_predict_time_sec: ROCKET transform + classifier fit + predict only.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root)

    # start end-to-end timer (data loading + preprocessing + ROCKET + classifier)
    t_total_start = time.time()

    # 1. load dataset
    X_train_df, y_train = load_UCR_UEA_dataset(
        name=dataset_name,
        split="train",
        return_X_y=True,
    )
    X_test_df, y_test = load_UCR_UEA_dataset(
        name=dataset_name,
        split="test",
        return_X_y=True,
    )

    # 2. convert panel to NumPy (univariate case)
    X_train = np.vstack(
        X_train_df.iloc[:, 0].apply(lambda s: s.to_numpy()).to_numpy()
    )
    X_test = np.vstack(
        X_test_df.iloc[:, 0].apply(lambda s: s.to_numpy()).to_numpy()
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 3. normalise
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    def to_panel(X):
        return pd.DataFrame({0: [pd.Series(row) for row in X]})

    X_train_panel = to_panel(X_train_norm)
    X_test_panel = to_panel(X_test_norm)

    # 4. ROCKET transform + classifier timing (fit + predict only)
    t_fit_start = time.time()

    rocket = Rocket(num_kernels=num_kernels, random_state=random_state)
    rocket.fit(X_train_panel)
    X_train_features = rocket.transform(X_train_panel)
    X_test_features = rocket.transform(X_test_panel)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)

    fit_predict_time_sec = time.time() - t_fit_start
    total_time_sec = time.time() - t_total_start

    # 5. save CSV
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name.lower()}_rocket_sktime.csv"

    import csv

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "n_kernels",
                "train_size",
                "test_size",
                "accuracy",
                "fit_predict_time_sec",
                "total_time_sec",
                "random_state",
            ]
        )
        writer.writerow(
            [
                dataset_name,
                num_kernels,
                X_train.shape[0],
                X_test.shape[0],
                acc,
                fit_predict_time_sec,
                total_time_sec,
                random_state,
            ]
        )

    return {
        "dataset": dataset_name,
        "n_kernels": num_kernels,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "accuracy": acc,
        "fit_predict_time_sec": fit_predict_time_sec,
        "total_time_sec": total_time_sec,
        "random_state": random_state,
        "results_path": out_path.relative_to(project_root),
    }