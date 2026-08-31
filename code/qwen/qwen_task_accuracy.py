"""qwen_task_accuracy.py - Compute Qwen2-Audio's own A/B/C task accuracy."""
import glob
import math
import os
import pandas as pd

LOGITS_DIR = "data/output/qwen"
PATTERN    = "qwen_logits_*.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def main():
    files = sorted(glob.glob(os.path.join(LOGITS_DIR, PATTERN)))
    if not files:
        print("No logits CSVs found.")
        return

    print(f"\nQwen2-Audio task accuracy (argmax of p_A/p_B/p_C vs true_A)\n")
    print(f"{'Set':<55}  {'N':>4}  {'Acc':>7}  {'95% CI (Wilson)':>18}")
    print("-" * 92)

    for f in files:
        df = pd.read_csv(f)
        n = len(df)
        k = int(df["correct"].sum())
        acc = k / n
        lo, hi = wilson(k, n)
        label = os.path.basename(f).replace("qwen_logits_", "").rsplit("_", 1)[0]
        # strip trailing date if present
        label = label.rsplit("_", 1)[0] if label[-8:].isdigit() else label
        print(f"{label:<55}  {n:>4}  {acc:>6.1%}  [{lo:>5.1%},{hi:>6.1%}]")

    print("\nChance baseline: 33.3% (3-way A/B/C task)")


if __name__ == "__main__":
    main()
