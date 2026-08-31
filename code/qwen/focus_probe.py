"""focus_probe.py — Logistic regression probe on Qwen2-Audio encoder vectors.

Predicts prosodic focus label (1 vs 2) from the 1280-dim mean-pooled audio
encoder vector, evaluated with leave-one-out cross-validation.

Usage (from repo root):
    # Basic LOO probe on speaker0 f-files
    python code/qwen/focus_probe.py --encoder-csv data/output/qwen/qwen_encoder_<ts>.csv

    # With cross-speaker held-out test (speaker1 and speaker2 ns-files)
    python code/qwen/focus_probe.py \\
        --encoder-csv  data/output/qwen/qwen_encoder_SP0_<ts>.csv \\
        --test-csvs    data/output/qwen/qwen_encoder_SP1_<ts>.csv \\
                       data/output/qwen/qwen_encoder_SP2_<ts>.csv

    # Disable PCA (raw 1280-dim)
    python code/qwen/focus_probe.py --encoder-csv ... --pca 0
"""

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOCUS_COL   = "focus"
FEAT_PREFIX = "d"      # columns d0, d1, …, d1279


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoder_csv(path):
    df = pd.read_csv(path)
    # Keep only rows with a valid focus label (1 or 2)
    df = df[df[FOCUS_COL].isin([1, 2])].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith(FEAT_PREFIX) and c[1:].isdigit()]
    X = df[feat_cols].values.astype(np.float32)
    y = (df[FOCUS_COL].values == 2).astype(int)   # 0 = focus-1, 1 = focus-2
    return df, X, y


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def make_pipeline(pca_components):
    steps = [("scaler", StandardScaler())]
    if pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    steps.append(("clf", LogisticRegression(max_iter=1000, random_state=42)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Leave-one-out CV
# ---------------------------------------------------------------------------

def run_loo(X, y, pca_components):
    loo  = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        pipe = make_pipeline(pca_components)
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[0, 1]
    correct = int((preds == y).sum())
    return preds, probs, correct


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom   = 1 + z ** 2 / n
    centre  = (p + z ** 2 / (2 * n)) / denom
    margin  = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _emit(text, log_fh=None):
    print(text)
    if log_fh is not None:
        log_fh.write(text + "\n")


def print_results(header, correct, n, preds, y, log_fh=None):
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    _emit(f"\n{'─' * 54}", log_fh)
    _emit(f"  {header}", log_fh)
    _emit(f"{'─' * 54}", log_fh)
    _emit(f"  N        : {n}  (focus-1: {(y==0).sum()}, focus-2: {(y==1).sum()})", log_fh)
    _emit(f"  Correct  : {correct}", log_fh)
    _emit(f"  Accuracy : {acc:.1%}  (chance 50.0%)", log_fh)
    _emit(f"  95% CI   : [{lo:.1%}, {hi:.1%}]  (Wilson)", log_fh)
    _emit(f"  Confusion matrix  (rows = true, cols = predicted):", log_fh)
    _emit(f"               pred-0  pred-1", log_fh)
    _emit(f"    true-0  :    {cm[0,0]:4d}    {cm[0,1]:4d}   (focus-1)", log_fh)
    _emit(f"    true-1  :    {cm[1,0]:4d}    {cm[1,1]:4d}   (focus-2)", log_fh)
    _emit(f"{'─' * 54}", log_fh)


# ---------------------------------------------------------------------------
# 2-D PCA scatter
# ---------------------------------------------------------------------------

def pca_scatter(X, y, df, output_path, log_fh=None):
    scaler = StandardScaler()
    pca    = PCA(n_components=2, random_state=42)
    X2     = pca.fit_transform(scaler.fit_transform(X))

    fig, ax = plt.subplots(figsize=(7, 5))
    palette     = {0: "#3182bd", 1: "#e6550d"}
    label_names = {0: "focus-1", 1: "focus-2"}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=palette[cls], label=label_names[cls],
            alpha=0.75, edgecolors="white", linewidths=0.4, s=60,
        )

    # Light annotation with file_id + item index
    for i in range(len(y)):
        ax.annotate(
            f"{df['file_id'].iloc[i]}[{df['item_idx'].iloc[i]}]",
            (X2[i, 0], X2[i, 1]),
            fontsize=4.5, alpha=0.45, ha="center", va="bottom",
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title(
        "PCA of Qwen2-Audio encoder vectors\ncoloured by prosodic focus label"
    )
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    _emit(f"  Scatter → {output_path}", log_fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LOO logistic regression probe on Qwen2-Audio encoder vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--encoder-csv", required=True, metavar="CSV",
        help="Encoder CSV for the main (train) speaker — speaker0 f-files",
    )
    parser.add_argument(
        "--test-csvs", nargs="*", default=[], metavar="CSV",
        help="Encoder CSVs for held-out speakers (cross-speaker test; no CV needed)",
    )
    parser.add_argument(
        "--pca", type=int, default=50, metavar="N",
        help="PCA components before logistic regression (0 = disabled, default: 50)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join("data", "output", "qwen"),
        help="Directory for plots and output (default: data/output/qwen)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts           = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"focus_probe_results_{ts}.txt")
    scatter_path = os.path.join(args.output_dir, f"focus_probe_pca_scatter_{ts}.png")
    pca_desc     = f"PCA({args.pca}) → " if args.pca > 0 else ""

    with open(results_path, "w", encoding="utf-8") as log:

        _emit(
            f"focus_probe.py — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"encoder CSV  : {args.encoder_csv}\n"
            f"pipeline     : {pca_desc}LogisticRegression\n"
            f"test CSVs    : {args.test_csvs or 'none'}",
            log,
        )

        # -------------------------------------------------------------------
        # Load main dataset
        # -------------------------------------------------------------------
        _emit(f"\nLoading encoder CSV : {args.encoder_csv}", log)
        df, X, y = load_encoder_csv(args.encoder_csv)
        n = len(y)
        _emit(
            f"  Items loaded       : {n}  "
            f"(focus-1: {(y==0).sum()}, focus-2: {(y==1).sum()})",
            log,
        )

        if n < 20:
            _emit(
                f"\nWARNING: Only {n} items found. For a meaningful probe you need encoder "
                "vectors for all f1-f10 files (~100 items). Run:\n\n"
                "  python code/qwen/run_qwen.py --skip-inference "
                "--files f1 f2 f3 f4 f5 f6 f7 f8 f9 f10\n\n"
                "then re-run this script on the newly generated encoder CSV.",
                log,
            )

        # -------------------------------------------------------------------
        # LOO cross-validation
        # -------------------------------------------------------------------
        _emit(f"\nRunning LOO-CV  [{pca_desc}LogisticRegression] …", log)
        preds_loo, _, correct_loo = run_loo(X, y, args.pca)
        print_results(
            f"LOO-CV — main speaker (f-files) — {pca_desc}LR",
            correct_loo, n, preds_loo, y,
            log_fh=log,
        )

        # -------------------------------------------------------------------
        # 2-D PCA scatter plot
        # -------------------------------------------------------------------
        pca_scatter(X, y, df, scatter_path, log_fh=log)

        # -------------------------------------------------------------------
        # Cross-speaker held-out test
        # -------------------------------------------------------------------
        if args.test_csvs:
            _emit(
                f"\nCross-speaker test — training full model on {n} main-speaker items …",
                log,
            )
            pipe = make_pipeline(args.pca)
            pipe.fit(X, y)

            for csv_path in args.test_csvs:
                _emit(f"\n  Loading test CSV: {csv_path}", log)
                df_t, X_t, y_t = load_encoder_csv(csv_path)
                if len(y_t) == 0:
                    _emit("  (no valid focus-labeled items — skipping)", log)
                    continue
                preds_t   = pipe.predict(X_t)
                correct_t = int((preds_t == y_t).sum())
                print_results(
                    f"Cross-speaker — {os.path.basename(csv_path)}",
                    correct_t, len(y_t), preds_t, y_t,
                    log_fh=log,
                )
        else:
            _emit(
                "\n  (no --test-csvs provided; skipping cross-speaker test)\n"
                "  To generate encoder vectors for speaker1 / speaker2:\n"
                "    python code/qwen/run_qwen.py --skip-inference "
                "--speaker speaker1 --files ns1 ns2 ns3\n"
                "    python code/qwen/run_qwen.py --skip-inference "
                "--speaker speaker2 --files ns1 ns2 ns3\n"
                "  then re-run with --test-csvs <those CSVs>",
                log,
            )

    _emit(f"\n  Results → {results_path}")


if __name__ == "__main__":
    main()
