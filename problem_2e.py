import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.random_projection import SparseRandomProjection



def load_matrix_try_both(path: Path) -> np.ndarray:
    """Load numeric matrix from txt; try comma first, then whitespace."""
    try:
        return np.loadtxt(path, delimiter=",", dtype=float)
    except Exception:
        return np.loadtxt(path, dtype=float)


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Append a column of ones (bias term)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def qr_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve min ||Ax - b||_2 using QR. If R is singular, fall back to lstsq.
    (Keeps 'using QR' per spec but robust to low-rank cases after projection.)
    """
    Q, R = np.linalg.qr(A, mode="reduced")
    Qtb = Q.T @ b
    try:
        return np.linalg.solve(R, Qtb)  # fast path
    except np.linalg.LinAlgError:
        # fallback: least-squares on the triangular system
        x, *_ = np.linalg.lstsq(R, Qtb, rcond=None)
        return x


def classify(scores: np.ndarray) -> np.ndarray:
    """C(y) = +1 if y>=0 else -1."""
    return np.where(scores >= 0, 1, -1).astype(int)


def misclass_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - float(np.mean(y_true == y_pred))



def run_sparse_rp_qr_2e(data_dir: Path, runs: int = 500) -> pd.DataFrame:
    # Load data
    A = load_matrix_try_both(data_dir / "train.txt")
    b = load_matrix_try_both(data_dir / "train_values.txt").astype(int).reshape(-1)

    Aval = load_matrix_try_both(data_dir / "validate.txt")
    bval = load_matrix_try_both(data_dir / "validate_values.txt").astype(int).reshape(-1)

    assert A.shape[1] == 30 and Aval.shape[1] == 30, f"Expected 30 features, got {A.shape[1]}, {Aval.shape[1]}"
    assert A.shape[0] == b.shape[0] and Aval.shape[0] == bval.shape[0], "Row count mismatch train/val vs labels"

    dims = [5, 10, 20]
    rows = []

    print("\n=== Problem 2(e): Sparse Random Projection + QR Least Squares ===")
    for d in dims:
        tr_errs, va_errs, times = [], [], []

        for r in range(runs):
            # independent RP each run
            rp = SparseRandomProjection(n_components=d, dense_output=True, random_state=10000 + r)

            t0 = time.perf_counter()

            rp.fit(A)                 # choose the random components
            Aproj = rp.transform(A)
            Avalproj = rp.transform(Aval)

            # add bias
            Aproj_i = add_intercept(Aproj)
            Avalproj_i = add_intercept(Avalproj)

            # QR LS
            w = qr_least_squares(Aproj_i, b)

            # predictions
            ytr = classify(Aproj_i @ w)
            yva = classify(Avalproj_i @ w)

            tr_err = misclass_rate(b, ytr)
            va_err = misclass_rate(bval, yva)

            elapsed = time.perf_counter() - t0

            tr_errs.append(tr_err)
            va_errs.append(va_err)
            times.append(elapsed)

            rows.append({
                "method": "linear_sparseRP",
                "d": d,
                "run": r + 1,
                "train_err": tr_err,
                "val_err": va_err,
                "time_sec_total": elapsed
            })

        # summary printout per d
        tr_mean = float(np.mean(tr_errs)); tr_std = float(np.std(tr_errs, ddof=1))
        va_mean = float(np.mean(va_errs)); va_std = float(np.std(va_errs, ddof=1))
        tm_mean = float(np.mean(times));   tm_std = float(np.std(times, ddof=1))

        print(f"\n-- d = {d} --")
        print(f"Train misclassification: {tr_mean*100:.2f}% ± {tr_std*100:.2f}%")
        print(f"Valid misclassification: {va_mean*100:.2f}% ± {va_std*100:.2f}%")
        print(f"Avg time over {runs} runs (projection + QR + preds): {tm_mean:.6f}s ± {tm_std:.6f}s")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Problem 2(e): Sparse RP -> QR Least Squares (500 runs)")
    ap.add_argument("--data-dir", type=str, default=".", help="Folder containing train/validate files")
    ap.add_argument("--runs", type=int, default=500, help="Number of independent runs (default 500)")
    args = ap.parse_args()

    out = run_sparse_rp_qr_2e(Path(args.data_dir), runs=args.runs)
   

