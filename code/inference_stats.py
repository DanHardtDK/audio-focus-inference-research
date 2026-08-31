"""
Statistical analysis of the Inference Task results reported in
"Can Audio LLMs Understand Spoken Language? An Inference Test Based on
Alternative Semantics" (Tables 2, 6, 7, 8, 9, 10, and Table 5 human survey).

All input numbers are transcribed directly from the paper's tables (percentages
and cell sizes N). Because only aggregated accuracy percentages are available
(not raw item-by-item correctness), counts of "correct" trials are reconstructed
as round(pct/100 * N). This is exact for FS=0 (zero-shot) cells, where each of
the N items is tested exactly once. For FS=2 / FS=5 cells, the paper's
fold-averaging procedure means the effective number of independent trials is
not simply N (see paper Sec 4.1.1); we still use N as a conservative nominal
sample size, which should be read as an approximation - flagged throughout.

Tests used:
- Exact binomial test (two-sided) vs. the 50% text-only baseline for each cell.
- Two-proportion z-test (statsmodels) for model-vs-model / condition-vs-condition
  comparisons where both N's are reasonably large; Fisher's exact test is used
  as a robustness check whenever any cell N <= 24.
- Wilson score confidence intervals for all proportions.
- Cohen's h as the effect size for proportion differences.
- A grouped-binomial GLM (logistic regression on aggregated counts) for the
  2x2x2 (Focus x Logic x Alternative) design in Table 10, both per-model and
  pooled with Model as a covariate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

def wilson_ci(x, n, alpha=0.05):
    lo, hi = proportion_confint(x, n, alpha=alpha, method="wilson")
    return lo, hi

def binom_vs_baseline(x, n, p0=0.5, label=""):
    res = stats.binomtest(x, n, p0, alternative="two-sided")
    lo, hi = wilson_ci(x, n)
    return {
        "label": label, "x": x, "n": n, "pct": 100 * x / n,
        "baseline_pct": 100 * p0, "p_value": res.pvalue,
        "ci_low": 100 * lo, "ci_high": 100 * hi,
        "sig_0.05": res.pvalue < 0.05,
    }

def two_prop_test(x1, n1, x2, n2, label=""):
    z, p = proportions_ztest([x1, x2], [n1, n2])
    # Fisher exact as robustness check
    table = [[x1, n1 - x1], [x2, n2 - x2]]
    odds, p_fisher = stats.fisher_exact(table)
    h = cohens_h(x1 / n1, x2 / n2)
    return {
        "label": label, "p1_pct": 100 * x1 / n1, "p2_pct": 100 * x2 / n2,
        "n1": n1, "n2": n2, "z": z, "p_ztest": p, "p_fisher": p_fisher,
        "cohens_h": h,
    }

def pct_to_count(pct, n):
    return int(round(pct / 100 * n))

# ---------------------------------------------------------------------------
# TABLE 2: Inference accuracy (%) by model, few-shot setting (FS), speaker
# Speaker 0: N=128; Speakers 1 & 2: N=24. FS=5 only run for Speaker 0.
# ---------------------------------------------------------------------------
table2 = [
    # model, fs, speaker, pct, n
    ("gemini-3.1-pro-preview", 0, 0, 87.5, 128),
    ("gemini-3.1-pro-preview", 0, 1, 91.7, 24),
    ("gemini-3.1-pro-preview", 0, 2, 54.2, 24),
    ("gemini-3.1-pro-preview", 2, 0, 82.5, 128),
    ("gemini-3.1-pro-preview", 2, 1, 90.3, 24),
    ("gemini-3.1-pro-preview", 2, 2, 70.8, 24),
    ("gemini-3.1-pro-preview", 5, 0, 90.0, 128),

    ("gemini-2.5-flash", 0, 0, 68.8, 128),
    ("gemini-2.5-flash", 0, 1, 58.3, 24),
    ("gemini-2.5-flash", 0, 2, 41.7, 24),
    ("gemini-2.5-flash", 2, 0, 53.6, 128),
    ("gemini-2.5-flash", 2, 1, 54.5, 24),
    ("gemini-2.5-flash", 2, 2, 47.2, 24),
    ("gemini-2.5-flash", 5, 0, 60.0, 128),

    ("gpt-4o-audio-preview", 0, 0, 46.1, 128),
    ("gpt-4o-audio-preview", 0, 1, 20.8, 24),
    ("gpt-4o-audio-preview", 0, 2, 41.7, 24),
    ("gpt-4o-audio-preview", 2, 0, 42.3, 128),
    ("gpt-4o-audio-preview", 2, 1, 41.7, 24),
    ("gpt-4o-audio-preview", 2, 2, 36.1, 24),
    ("gpt-4o-audio-preview", 5, 0, 49.2, 128),

    ("gpt-audio", 0, 0, 39.1, 128),
    ("gpt-audio", 0, 1, 29.2, 24),
    ("gpt-audio", 0, 2, 37.5, 24),
    ("gpt-audio", 2, 0, 39.1, 128),
    ("gpt-audio", 2, 1, 38.9, 24),
    ("gpt-audio", 2, 2, 34.7, 24),
    ("gpt-audio", 5, 0, 45.0, 128),
]
df2 = pd.DataFrame(table2, columns=["model", "fs", "speaker", "pct", "n"])
df2["x"] = df2.apply(lambda r: pct_to_count(r.pct, r.n), axis=1)

print("=" * 90)
print("1. BINOMIAL TESTS vs. 50% TEXT-ONLY BASELINE (Table 2 cells)")
print("=" * 90)
rows = []
for _, r in df2.iterrows():
    res = binom_vs_baseline(r.x, r.n, 0.5, label=f"{r.model} FS={r.fs} Sp{r.speaker}")
    res["fewshot_caveat"] = "approx (folded)" if r.fs > 0 else "exact"
    rows.append(res)
bt = pd.DataFrame(rows)
bt_display = bt[["label", "x", "n", "pct", "p_value", "sig_0.05", "fewshot_caveat"]]
print(bt_display.to_string(index=False))

n_sig = bt["sig_0.05"].sum()
print(f"\n{n_sig} of {len(bt)} cells significantly above 50% baseline (alpha=.05, uncorrected).")
print("With Bonferroni correction for", len(bt), "tests, alpha_corrected =", 0.05/len(bt))
bt["sig_bonf"] = bt["p_value"] < (0.05/len(bt))
print(bt[bt["sig_bonf"]][["label","pct","p_value"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 2. MODEL-VS-MODEL COMPARISONS (Speaker 0, FS=0 -- the largest, cleanest cell)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. MODEL-VS-MODEL COMPARISONS (Speaker 0, FS=0, N=128 each)")
print("=" * 90)

def get_xn(model, fs, speaker):
    r = df2[(df2.model == model) & (df2.fs == fs) & (df2.speaker == speaker)].iloc[0]
    return int(r.x), int(r.n)

models_fs0_sp0 = {
    "gemini-3.1-pro-preview": get_xn("gemini-3.1-pro-preview", 0, 0),
    "gemini-2.5-flash": get_xn("gemini-2.5-flash", 0, 0),
    "gpt-4o-audio-preview": get_xn("gpt-4o-audio-preview", 0, 0),
    "gpt-audio": get_xn("gpt-audio", 0, 0),
}
import itertools
pair_rows = []
for (m1, (x1, n1)), (m2, (x2, n2)) in itertools.combinations(models_fs0_sp0.items(), 2):
    res = two_prop_test(x1, n1, x2, n2, label=f"{m1} vs {m2}")
    pair_rows.append(res)
pw = pd.DataFrame(pair_rows)
print(pw.to_string(index=False))

# Pooled Gemini vs pooled GPT (Speaker 0, FS=0)
gx = models_fs0_sp0["gemini-3.1-pro-preview"][0] + models_fs0_sp0["gemini-2.5-flash"][0]
gn = models_fs0_sp0["gemini-3.1-pro-preview"][1] + models_fs0_sp0["gemini-2.5-flash"][1]
px = models_fs0_sp0["gpt-4o-audio-preview"][0] + models_fs0_sp0["gpt-audio"][0]
pn = models_fs0_sp0["gpt-4o-audio-preview"][1] + models_fs0_sp0["gpt-audio"][1]
pooled = two_prop_test(gx, gn, px, pn, label="Pooled Gemini vs Pooled GPT (Sp0, FS0)")
print("\nPooled family comparison:")
print(pd.DataFrame([pooled]).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. FEW-SHOT EFFECT WITHIN MODEL (Speaker 0: FS0 vs FS2 vs FS5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. FEW-SHOT EFFECT WITHIN MODEL (Speaker 0). CAUTION: FS>0 N is approximate")
print("   (fold-averaging means trials aren't fully independent - see docstring).")
print("=" * 90)
fs_rows = []
for model in df2.model.unique():
    x0, n0 = get_xn(model, 0, 0)
    x2, n2 = get_xn(model, 2, 0)
    x5, n5 = get_xn(model, 5, 0)
    fs_rows.append(two_prop_test(x0, n0, x2, n2, label=f"{model}: FS0 vs FS2"))
    fs_rows.append(two_prop_test(x0, n0, x5, n5, label=f"{model}: FS0 vs FS5"))
    fs_rows.append(two_prop_test(x2, n2, x5, n5, label=f"{model}: FS2 vs FS5"))
    # Cochran-Armitage-style trend via logistic regression on FS as ordinal predictor
    fsvals = np.array([0, 2, 5])
    xs = np.array([x0, x2, x5]); ns = np.array([n0, n2, n5])
    succ = xs; fail = ns - xs
    glm = sm.GLM(np.column_stack([succ, fail]), sm.add_constant(fsvals), family=sm.families.Binomial())
    fit = glm.fit()
    fs_rows.append({"label": f"{model}: trend across FS (GLM slope)", "p1_pct": None, "p2_pct": None,
                     "n1": None, "n2": None, "z": fit.tvalues[1], "p_ztest": fit.pvalues[1],
                     "p_fisher": None, "cohens_h": fit.params[1]})
fsw = pd.DataFrame(fs_rows)
print(fsw.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. SPEAKER EFFECT WITHIN MODEL (FS=0: Speaker0 vs 1 vs 2) - small N, use Fisher
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. SPEAKER EFFECT WITHIN MODEL (FS=0). Speakers 1,2 have N=24 -> Fisher exact")
print("=" * 90)
sp_rows = []
for model in df2.model.unique():
    x0, n0 = get_xn(model, 0, 0)
    x1, n1 = get_xn(model, 0, 1)
    x2, n2 = get_xn(model, 0, 2)
    sp_rows.append(two_prop_test(x0, n0, x1, n1, label=f"{model}: Sp0 vs Sp1"))
    sp_rows.append(two_prop_test(x0, n0, x2, n2, label=f"{model}: Sp0 vs Sp2"))
    sp_rows.append(two_prop_test(x1, n1, x2, n2, label=f"{model}: Sp1 vs Sp2"))
spw = pd.DataFrame(sp_rows)
print(spw[["label","p1_pct","p2_pct","n1","n2","p_fisher","cohens_h"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. FACTORIAL DESIGN: Table 10 (Foc x Log x Alt, N=16/cell, FS=0)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("5. FACTORIAL (Foc x Log x Alt) GROUPED-BINOMIAL GLM -- Table 10, N=16/cell")
print("=" * 90)
table10 = [
    # Foc, Log, Alt, G3.1, G2.5, G4o, Gpa (inference accuracy %)
    (1, "POS", 1, 100, 88, 44, 31),
    (1, "POS", 2, 94, 50, 31, 50),
    (1, "NEG", 1, 100, 75, 75, 38),
    (1, "NEG", 2, 88, 75, 6, 12),
    (2, "POS", 1, 69, 25, 62, 56),
    (2, "POS", 2, 100, 88, 62, 62),
    (2, "NEG", 1, 56, 69, 25, 25),
    (2, "NEG", 2, 94, 81, 62, 38),
]
long_rows = []
model_names = ["Gemini-3.1", "Gemini-2.5", "GPT-4o", "GPT-audio"]
for foc, log, alt, *accs in table10:
    for m, acc in zip(model_names, accs):
        n = 16
        x = pct_to_count(acc, n)
        long_rows.append({"Foc": foc, "Log": log, "Alt": alt, "Model": m, "x": x, "n": n})
t10 = pd.DataFrame(long_rows)
t10["fail"] = t10.n - t10.x
t10["Foc"] = t10.Foc.astype(str)
t10["Alt"] = t10.Alt.astype(str)

print("\n-- Pooled model (Model as covariate) --")
glm_pooled = smf.glm("x + fail ~ C(Foc) + C(Log) + C(Alt) + C(Model)",
                      data=t10, family=sm.families.Binomial())
fit_pooled = glm_pooled.fit()
print(fit_pooled.summary2().tables[1])

print("\n-- Per-model GLMs (Foc, Log, Alt main effects; +interactions if data allow) --")
for m in model_names:
    sub = t10[t10.Model == m]
    glm_m = smf.glm("x + fail ~ C(Foc) + C(Log) + C(Alt)", data=sub, family=sm.families.Binomial())
    fit_m = glm_m.fit()
    print(f"\nModel: {m}")
    print(fit_m.summary2().tables[1])

# ---------------------------------------------------------------------------
# 6. HUMAN SURVEY vs BEST MODEL (Table 5 vs Table 2)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("6. HUMAN SURVEY vs MODEL (approximate; survey N reconstructed from means)")
print("   Overall: 12 respondents x 24 items = 288 respondent-item trials (NOT independent")
print("   within respondent -- treat as approximate / conservative).")
print("=" * 90)
human_overall_x = pct_to_count(67.7, 288)
human_overall_n = 288
gem_x, gem_n = get_xn("gemini-3.1-pro-preview", 0, 0)  # best model, Sp0 FS0 as reference
res_survey = two_prop_test(human_overall_x, human_overall_n, gem_x, gem_n,
                            label="Human overall vs gemini-3.1-pro (Sp0,FS0)")
print(pd.DataFrame([res_survey]).to_string(index=False))

# per-speaker survey vs matched model/speaker cell
print("\nPer-speaker (survey n reconstructed from mean x 24 items):")
survey_speakers = {0: (86.1, 3*24), 1: (63.5, 4*24), 2: (60.0, 5*24)}
for sp, (pct, n) in survey_speakers.items():
    x = pct_to_count(pct, n)
    gx, gn = get_xn("gemini-3.1-pro-preview", 0, sp)
    res = two_prop_test(x, n, gx, gn, label=f"Human Sp{sp} vs gemini-3.1-pro Sp{sp}")
    print(pd.DataFrame([res]).to_string(index=False))

print("\nDone.")
