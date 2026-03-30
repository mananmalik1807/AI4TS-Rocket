from pathlib import Path
import time

import numpy as np
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

@dataclass
class RocketKernel:
    weights: np.ndarray  # shape (kernel_length,)
    bias: float
    dilation: int
    use_padding: bool
    
    

def generate_random_kernels(num_kernels: int, series_length: int, rng: np.random.RandomState):
    kernels = []

    for _ in range(num_kernels):
        # length in {7, 9, 11}
        kernel_length = int(rng.choice([7, 9, 11]))

        # weights ~ N(0, 1), then mean-centered
        weights = rng.normal(loc=0.0, scale=1.0, size=kernel_length)
        weights = weights - weights.mean()

        # bias ~ U(-1, 1)
        bias = float(rng.uniform(-1.0, 1.0))

        # dilation: d = 2^x, x ~ U(0, A), A = log2((L_in - 1) / (L_k - 1))
        # clamp A >= 0 to avoid weird cases on very short series
        A = np.log2(max((series_length - 1) / (kernel_length - 1), 1.0))
        x = rng.uniform(0.0, A)
        dilation = int(round(2 ** x))
        dilation = max(dilation, 1)

        # padding: use or not with prob 0.5
        use_padding = bool(rng.rand() < 0.5)

        kernels.append(RocketKernel(weights=weights, bias=bias, dilation=dilation, use_padding=use_padding))

    return kernels

def apply_kernel_to_series(series: np.ndarray, kernel: RocketKernel) -> np.ndarray:
    """
    Compute the feature map for one kernel on one 1D series.
    """
    w = kernel.weights
    b = kernel.bias
    d = kernel.dilation
    L_in = len(series)
    L_k = len(w)

    if kernel.use_padding:
        # amount of padding so that the "effective" kernel can be centered everywhere
        # effective length = (L_k - 1) * d + 1
        eff_len = (L_k - 1) * d + 1
        pad = (eff_len - 1) // 2
        padded = np.pad(series, pad_width=pad, mode="constant", constant_values=0.0)
        L_eff = len(padded)
    else:
        padded = series
        L_eff = L_in

    # compute how many positions the kernel can slide over
    # last index where the kernel's last weight still lies inside
    last_start = L_eff - 1 - (L_k - 1) * d
    if last_start < 0:
        # degenerate case: kernel longer than series; just one dot product over available
        return np.array([], dtype=float)

    positions = last_start + 1
    feature_map = np.empty(positions, dtype=float)

    for i in range(positions):
        # indices: i, i+d, i+2d, ..., i+(L_k-1)*d
        idx = i + d * np.arange(L_k)
        vals = padded[idx]
        feature_map[i] = np.dot(vals, w) + b

    return feature_map

def rocket_transform(X_norm: np.ndarray, kernels: list[RocketKernel]) -> np.ndarray:
    """
    X_norm: shape (n_samples, series_length)
    Returns X_features: shape (n_samples, 2 * num_kernels) with [ppv, max] for each kernel.
    """
    n_samples, series_length = X_norm.shape
    num_kernels = len(kernels)
    X_features = np.zeros((n_samples, 2 * num_kernels), dtype=float)

    for i in range(n_samples):
        series = X_norm[i]
        feat_vector = []

        for k in kernels:
            fm = apply_kernel_to_series(series, k)
            if fm.size == 0:
                # if degenerate, use zeros
                max_val = 0.0
                ppv = 0.0
            else:
                max_val = fm.max()
                ppv = (fm > 0).mean()

            feat_vector.extend([ppv, max_val])

        X_features[i, :] = np.array(feat_vector, dtype=float)

    return X_features

def run_original_rocket_experiment(dataset_name: str,
                                   num_kernels: int = 10000,
                                   random_state: int = 42,
                                   project_root: str | Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root)

    # 1. load dataset (same as sktime version)
    X_train_df, y_train = load_UCR_UEA_dataset(
        name=dataset_name,
        split="train",
        return_X_y=True
    )
    X_test_df, y_test = load_UCR_UEA_dataset(
        name=dataset_name,
        split="test",
        return_X_y=True
    )

    # 2. convert panel to NumPy (univariate)
    X_train = np.vstack(X_train_df.iloc[:, 0].apply(lambda s: s.to_numpy()).to_numpy())
    X_test = np.vstack(X_test_df.iloc[:, 0].apply(lambda s: s.to_numpy()).to_numpy())
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 3. normalise
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 4. ROCKET-style kernels + transform
    rng = np.random.RandomState(random_state)
    series_length = X_train_norm.shape[1]
    kernels = generate_random_kernels(num_kernels=num_kernels,
                                      series_length=series_length,
                                      rng=rng)

    t0 = time.time()
    X_train_features = rocket_transform(X_train_norm, kernels)
    X_test_features = rocket_transform(X_test_norm, kernels)

    # 5. classifier on ROCKET features
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)
    fit_predict_time_sec = time.time() - t0

    # 6. save CSV
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name.lower()}_rocket_original.csv"

    df = pd.DataFrame([{
        "dataset": dataset_name,
        "n_kernels": num_kernels,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "accuracy": acc,
        "fit_predict_time_sec": fit_predict_time_sec,
        "random_state": random_state,
    }])
    df.to_csv(out_path, index=False)

    return {
        "dataset": dataset_name,
        "n_kernels": num_kernels,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "accuracy": acc,
        "fit_predict_time_sec": fit_predict_time_sec,
        "random_state": random_state,
        "results_path": out_path.relative_to(project_root),
    }