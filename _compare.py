import glob, os, re
import pandas as pd
from pathlib import Path

VARIANTS = {
    "baseline_4bit": "data/output/qwen",
    "audio_fp16":    "data/output/qwen2_audio_fp16",
    "omni_fp16":     "data/output/qwen2_5_omni_fp16",
}
SPEAKERS = [0, 1, 2]

def newest(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ---------------- TASK 4 ----------------
t4_rows = []
for vname, vdir in VARIANTS.items():
    for sp in SPEAKERS:
        pat = f"{vdir}/Task4/master_inference_*_SP*speaker{sp}*.csv"
        f = newest(pat)
        if not f:
            t4_rows.append([vname, sp, "MISSING", "", "", "", "", "", "", "", "", "", ""])
            continue
        df = pd.read_csv(f)
        N = len(df)
        acc = df["inf_correct"].mean() if "inf_correct" in df else float("nan")
        # per-class accuracy
        per_class = {}
        if "true_A" in df:
            for c in ["A", "B", "C"]:
                sub = df[df["true_A"] == c]
                per_class[c] = (len(sub), sub["inf_correct"].mean() if len(sub) else float("nan"))
        # prediction distribution: look for a model_pred / pred col
        pred_col = None
        for c in ["model_pred", "pred", "model_answer", "prediction", "model_inference"]:
            if c in df.columns:
                pred_col = c; break
        dist = {"A":0,"B":0,"C":0,"empty":0,"other":0}
        if pred_col:
            for v in df[pred_col].astype(str):
                v2 = v.strip().upper()
                if v2 in ("","NAN","NONE"):
                    dist["empty"] += 1
                elif v2 and v2[0] in "ABC" and (len(v2)==1 or not v2[1].isalpha()):
                    dist[v2[0]] += 1
                else:
                    # try find first A/B/C token
                    m = re.search(r"\b([ABC])\b", v2)
                    if m: dist[m.group(1)] += 1
                    else: dist["other"] += 1
        valid_frac = (dist["A"]+dist["B"]+dist["C"]) / N if N and pred_col else float("nan")
        t4_rows.append([vname, sp, N, f"{acc:.3f}",
                        f"A={per_class.get('A',(0,0))[1]:.2f}(n{per_class.get('A',(0,0))[0]})",
                        f"B={per_class.get('B',(0,0))[1]:.2f}(n{per_class.get('B',(0,0))[0]})",
                        f"C={per_class.get('C',(0,0))[1]:.2f}(n{per_class.get('C',(0,0))[0]})",
                        dist["A"], dist["B"], dist["C"], dist["empty"], dist["other"],
                        f"{valid_frac:.2f}"])

t4_df = pd.DataFrame(t4_rows, columns=["variant","spk","N","acc","accA","accB","accC","pA","pB","pC","pEmpty","pOther","valid"])
print("=== TASK 4: Inference accuracy ===")
print(t4_df.to_string(index=False))

# ---------------- TASK 2 ----------------
UPPER = re.compile(r"\b[A-Z]{2,}\b")
t2_rows = []
for vname, vdir in VARIANTS.items():
    for sp in SPEAKERS:
        pat = f"{vdir}/Task2/master_transcription_*_SP*speaker{sp}*.csv"
        f = newest(pat)
        if not f:
            t2_rows.append([vname, sp, "MISSING", "", "", ""])
            continue
        df = pd.read_csv(f)
        N = len(df)
        tc = df["trans_correct"].mean() if "trans_correct" in df else float("nan")
        has_upper = df["model_S1"].astype(str).apply(lambda s: bool(UPPER.search(s))).mean() if "model_S1" in df else float("nan")
        t2_rows.append([vname, sp, N, f"{tc:.3f}", f"{has_upper:.3f}", os.path.basename(f)])

t2_df = pd.DataFrame(t2_rows, columns=["variant","spk","N","trans_correct","frac_has_UPPER_token","file"])
print("\n=== TASK 2: Transcription ===")
print(t2_df.to_string(index=False))

# ---------------- TASK 1 / 3 PROBES ----------------
ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)%\s+\[\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\]\s+(\d+)")

def parse_peak(path):
    best = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = ROW_RE.match(line)
            if m:
                layer = int(m.group(1)); acc = float(m.group(2))
                ci_lo = float(m.group(3)); ci_hi = float(m.group(4)); n = int(m.group(5))
                if best is None or acc > best[1]:
                    best = (layer, acc, ci_lo, ci_hi, n)
    return best

probe_rows = []
for vname, vdir in VARIANTS.items():
    for task in ["Task1", "Task3"]:
        pat = f"{vdir}/{task}/layer_probe_*_results_*.txt"
        f = newest(pat)
        if not f:
            probe_rows.append([vname, task, "MISSING", "", "", "", "", ""])
            continue
        peak = parse_peak(f)
        if peak is None:
            probe_rows.append([vname, task, os.path.basename(f), "no rows parsed", "", "", "", ""])
        else:
            l, acc, lo, hi, n = peak
            probe_rows.append([vname, task, os.path.basename(f), l, f"{acc:.1f}%", f"[{lo:.1f}%,{hi:.1f}%]", n, ""])

p_df = pd.DataFrame(probe_rows, columns=["variant","task","file","peak_layer","peak_acc","ci","n",""])
print("\n=== TASK 1 / TASK 3: Peak LOO probe accuracy ===")
print(p_df.to_string(index=False))
