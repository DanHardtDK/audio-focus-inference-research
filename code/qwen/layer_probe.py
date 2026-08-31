"""layer_probe.py — Layer-by-layer logistic regression probe on Qwen2-Audio encoder.

Loads a stacked all-layers encoder CSV produced by

    qwen_inspect.py --all-layers

and trains a separate LOO logistic regression probe on each layer.
Optionally runs cross-speaker held-out tests on the same set of layers.
Saves a profile plot (accuracy vs. layer depth), a results CSV, and a text summary.

Usage (from repo root, using the .venv environment):
    python code/qwen/layer_probe.py \\
        --layers-csv data/output/qwen/qwen_encoder_layers_speaker0_f1-..._<ts>.csv \\
        --test-layers-csvs \\
            data/output/qwen/qwen_encoder_layers_speaker0_ns1-ns2-ns3_<ts>.csv \\
            data/output/qwen/qwen_encoder_layers_speaker1_ns1-ns2-ns3_<ts>.csv \\
            data/output/qwen/qwen_encoder_layers_speaker2_ns1-ns2-ns3_<ts>.csv \\
        --pca 50 \\
        --output-dir data/output/qwen
"""

import argparse
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOCUS_COL   = "focus"
TRUE_A_COL  = "true_A"
FEAT_PREFIX = "d"

LABEL_MAP = {"A": 0, "B": 1, "C": 2}   # for inference target


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_layers_csv(path, target="focus"):
    df = pd.read_csv(path)
    if target == "focus":
        df = df[df[FOCUS_COL].isin([1, 2])].reset_index(drop=True)
    else:  # inference
        df = df[df[TRUE_A_COL].isin(["A", "B", "C"])].reset_index(drop=True)
    return df


def get_Xy(df_layer, target="focus"):
    feat_cols = [c for c in df_layer.columns
                 if c.startswith(FEAT_PREFIX) and c[len(FEAT_PREFIX):].isdigit()]
    X = df_layer[feat_cols].values.astype(np.float32)
    if target == "focus":
        y = (df_layer[FOCUS_COL].values == 2).astype(int)
    else:  # inference
        y = np.array([LABEL_MAP[a] for a in df_layer[TRUE_A_COL].values], dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Probe pipeline
# ---------------------------------------------------------------------------

def make_pipeline(pca_components):
    steps = [("scaler", StandardScaler())]
    if pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    steps.append(("clf", LogisticRegression(max_iter=1000, random_state=42)))
    return Pipeline(steps)


def run_loo(X, y, pca_components):
    loo   = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X):
        pipe = make_pipeline(pca_components)
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
    return int((preds == y).sum()), preds


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p      = k / n
    denom  = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-by-layer logistic regression probe on Qwen2-Audio encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--layers-csv", required=True, metavar="CSV",
        help="Stacked all-layers encoder CSV (from qwen_inspect.py --all-layers); "
             "must have a 'layer' column with integer values 0-32",
    )
    parser.add_argument(
        "--test-layers-csvs", nargs="*", default=[], metavar="CSV",
        help="Stacked all-layers CSVs for held-out speakers (cross-speaker test)",
    )
    parser.add_argument(
        "--pca", type=int, default=50, metavar="N",
        help="PCA components before logistic regression (0 = disabled, default: 50)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join("data", "output", "qwen"),
        help="Output directory (default: data/output/qwen)",
    )
    parser.add_argument(
        "--target", choices=["focus", "inference"], default="focus",
        help="Probe target: 'focus' = binary focus-position (default, Task 1); "
             "'inference' = 3-class true_A label A/B/C (Task 3)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts           = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tgt_tag      = args.target                                   # "focus" or "inference"
    results_path = os.path.join(args.output_dir, f"layer_probe_{tgt_tag}_results_{ts}.txt")
    plot_path    = os.path.join(args.output_dir, f"layer_probe_{tgt_tag}_profile_{ts}.png")
    csv_out_path = os.path.join(args.output_dir, f"layer_probe_{tgt_tag}_data_{ts}.csv")

    pca_desc = f"PCA({args.pca}) -> " if args.pca > 0 else ""

    df_train = load_layers_csv(args.layers_csv, target=args.target)
    layers   = sorted(df_train["layer"].unique().tolist())
    n_layers = len(layers)

    test_dfs = [(p, load_layers_csv(p, target=args.target)) for p in args.test_layers_csvs]

    # ── Sanity check: warn if any test CSV contains feature vectors that are
    #   byte-identical to training vectors (data leakage). This catches the
    #   speaker0 ns1-3 case where clips are aliases of f-file clips.
    train_layer0 = df_train[df_train["layer"] == layers[0]]
    Xtr0, _      = get_Xy(train_layer0, target=args.target)
    train_keys   = {Xtr0[i].tobytes() for i in range(len(Xtr0))}
    for p, df_test in test_dfs:
        t0       = df_test[df_test["layer"] == layers[0]]
        Xt0, _   = get_Xy(t0, target=args.target)
        dupes    = sum(1 for i in range(len(Xt0)) if Xt0[i].tobytes() in train_keys)
        if dupes > 0:
            print(f"  ⚠  WARNING: test CSV {os.path.basename(p)} has "
                  f"{dupes}/{len(Xt0)} items with feature vectors identical to "
                  f"training items (possible duplicate clips / data leakage).")

    loo_accs = []
    loo_lo   = []
    loo_hi   = []
    test_results = {p: {"accs": [], "lo": [], "hi": []} for p, _ in test_dfs}

    # Build short display labels for test CSVs
    # qwen_encoder_layers_{speaker}_{files}_{ts}.csv -> "{speaker}_{files}"
    header_test_cols = []
    for p, _ in test_dfs:
        base  = os.path.basename(p)
        parts = base.replace("qwen_encoder_layers_", "").rsplit("_", 2)
        label = "_".join(parts[:2]) if len(parts) >= 3 else base
        header_test_cols.append((p, label))

    print(f"\nlayer_probe.py")
    print(f"  train CSV : {args.layers_csv}")
    print(f"  layers    : {n_layers}  (indices {layers[0]}-{layers[-1]})")
    print(f"  pipeline  : {pca_desc}LogisticRegression")
    for p, _ in test_dfs:
        print(f"  test CSV  : {p}")
    print(f"\nProbing {n_layers} layers ...\n")

    csv_rows = []

    with open(results_path, "w", encoding="utf-8") as log:
        log.write(
            f"layer_probe.py -- {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"train CSV : {args.layers_csv}\n"
            f"pipeline  : {pca_desc}LogisticRegression\n\n"
        )

        # Header
        header = f"{'Lyr':>4}  {'LOO-acc':>8}  {'95%-CI':>17}  {'N':>4}"
        for _, col_label in header_test_cols:
            header += f"  {col_label[:28]:>28}"
        sep = "-" * (len(header) + 4)
        log.write(header + "\n" + sep + "\n")
        print(header)
        print(sep)

        for layer in layers:
            # ── LOO-CV on training set ──────────────────────────────────
            subset   = df_train[df_train["layer"] == layer]
            X, y     = get_Xy(subset, target=args.target)
            n        = len(y)
            correct, _ = run_loo(X, y, args.pca)
            acc      = correct / n
            lo, hi   = wilson_ci(correct, n)

            loo_accs.append(acc)
            loo_lo.append(lo)
            loo_hi.append(hi)

            row = {
                "layer":   layer,
                "loo_acc": round(acc, 4),
                "loo_lo":  round(lo, 4),
                "loo_hi":  round(hi, 4),
                "n_train": n,
            }

            # ── cross-speaker test ──────────────────────────────────────
            pipe_full = make_pipeline(args.pca)
            pipe_full.fit(X, y)

            line = f"{layer:>4}  {acc:>8.1%}  [{lo:.1%}, {hi:.1%}]  {n:>4}"

            for (p, df_test), (_, col_label) in zip(test_dfs, header_test_cols):
                t_sub    = df_test[df_test["layer"] == layer]
                X_t, y_t = get_Xy(t_sub, target=args.target)
                if len(y_t) == 0:
                    test_results[p]["accs"].append(np.nan)
                    test_results[p]["lo"].append(np.nan)
                    test_results[p]["hi"].append(np.nan)
                    row[f"test_{col_label[:20]}"] = np.nan
                    line += f"  {'n/a':>28}"
                else:
                    preds_t   = pipe_full.predict(X_t)
                    correct_t = int((preds_t == y_t).sum())
                    acc_t     = correct_t / len(y_t)
                    lo_t, hi_t = wilson_ci(correct_t, len(y_t))
                    test_results[p]["accs"].append(acc_t)
                    test_results[p]["lo"].append(lo_t)
                    test_results[p]["hi"].append(hi_t)
                    row[f"test_{col_label[:20]}"] = round(acc_t, 4)
                    line += f"  {acc_t:.1%} [{lo_t:.1%},{hi_t:.1%}]"

            log.write(line + "\n")
            print(line)
            csv_rows.append(row)

    # --- Results CSV ---
    pd.DataFrame(csv_rows).to_csv(csv_out_path, index=False)
    print(f"\n\u2713 Results CSV  : {csv_out_path}")
    print(f"\u2713 Results TXT  : {results_path}")

    # --- Accuracy profile plot ---
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        layers, loo_accs, "o-", color="#3182bd",
        label="LOO-CV (train speaker, f-files)",
        linewidth=2, markersize=5, zorder=3,
    )
    ax.fill_between(layers, loo_lo, loo_hi, alpha=0.15, color="#3182bd")

    test_palette = ["#e6550d", "#31a354", "#756bb1", "#636363"]
    for (p, _df_t), (_, col_label), color in zip(test_dfs, header_test_cols, test_palette):
        accs = test_results[p]["accs"]
        los  = test_results[p]["lo"]
        his  = test_results[p]["hi"]
        accs_arr = np.array([a for a in accs], dtype=float)
        lo_arr   = np.array([v for v in los],  dtype=float)
        hi_arr   = np.array([v for v in his],  dtype=float)
        ax.plot(
            layers, accs_arr, "s--", color=color,
            label=f"Cross-speaker: {col_label}",
            linewidth=1.5, markersize=4,
        )
        ax.fill_between(layers, lo_arr, hi_arr, alpha=0.10, color=color)

    chance_level = 0.5 if args.target == "focus" else 1 / 3
    chance_label = "Chance (50%)" if args.target == "focus" else "Chance (33%)"
    ax.axhline(chance_level, color="gray", linestyle=":", linewidth=1.2,
               label=chance_label, zorder=1)

    ax.set_xlabel(
        "Encoder layer  (0 = conv/embedding output,  1-32 = transformer layers)",
        fontsize=10,
    )
    target_label = (
        "prosodic focus probe (focus=1 vs 2)" if args.target == "focus"
        else "inference answer probe (A / B / C)"
    )
    ax.set_ylabel("Probe accuracy", fontsize=10)
    ax.set_title(
        f"Qwen2-Audio encoder \u2014 {target_label}\n"
        f"{pca_desc}LogisticRegression,  LOO-CV (train) + held-out test",
        fontsize=11,
    )
    y_floor = 0.3 if args.target == "focus" else 0.15
    ax.set_ylim(y_floor, 1.05)
    ax.set_xlim(-0.5, layers[-1] + 0.5)
    ax.set_xticks(layers[::2])
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\u2713 Profile plot  : {plot_path}")


if __name__ == "__main__":
    main()
