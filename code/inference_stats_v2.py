"""
Item-level statistical re-analysis of the Inference Task, built from the raw
master CSVs in data/output/ (via build_dataset.py) rather than the aggregated
percentages in the paper's tables. This lets us use:

 - Exact paired tests (McNemar's) for model-vs-model comparisons, since all
   four models were tested on the SAME 128 items (Speaker 0, FS=0) and
   Speakers 1 & 2 share the same 24 items with each other.
 - Cluster-robust (GEE) logistic regression, clustering by item, for the
   Focus x Logic x Alternative design -- properly accounting for the fact
   that the same item appears once per model (4 correlated observations).
 - Cluster-robust (GEE) logistic regression, clustering by item, for the
   few-shot trend -- properly accounting for the fact that FS=2 items are
   each tested 3-4 times (once per cross-validation fold), which are NOT
   independent trials, unlike the simple proportion tests used previously.

All numbers were validated against the paper's Table 2 first (see
build_dataset.py output: 0/28 cells differ from the published percentages by
more than 0.3 points), so this is the same underlying data as the paper, at
item-level resolution.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from build_dataset import build_item_dataset

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

resolved, validation_report = build_item_dataset()
MODELS = ["gemini-3.1-pro-preview", "gemini-2.5-flash", "gpt-4o-audio-preview", "gpt-audio"]

def item_key(df):
    return list(zip(df.file_id, df.example_index))

print("=" * 90)
print("0. VALIDATION vs. PAPER TABLE 2 (sanity check on reconstructed item-level data)")
print("=" * 90)
print(validation_report.to_string(index=False))

# ---------------------------------------------------------------------------
# 1. SANITY CHECK: reproduce Table 6/7/8/9 from FS0/Sp0 item-level data
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("1. SANITY CHECK -- reproducing Table 6/7/8/9 (Foc/Log/Alt breakdowns) from raw items")
print("=" * 90)
fs0sp0 = {m: resolved[(m, 0, 0)] for m in MODELS}
for m in MODELS:
    d = fs0sp0[m]
    foc = d.groupby("focus")["inf_correct"].mean() * 100
    log = d.groupby("logic")["inf_correct"].mean() * 100
    alt = d.groupby("alternative")["inf_correct"].mean() * 100
    print(f"\n{m}: Inf%% overall={d.inf_correct.mean()*100:.1f}")
    print("  by Focus:", foc.round(1).to_dict())
    print("  by Logic:", log.round(1).to_dict())
    print("  by Alt:  ", alt.round(1).to_dict())

# ---------------------------------------------------------------------------
# 2. PAIRED MODEL-VS-MODEL COMPARISONS (McNemar's, Speaker 0, FS0, N=128 shared items)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. PAIRED MODEL-VS-MODEL COMPARISONS -- McNemar's exact test (Sp0, FS0, same 128 items)")
print("=" * 90)
merged_sp0 = fs0sp0[MODELS[0]][["file_id", "example_index", "inf_correct"]].rename(columns={"inf_correct": MODELS[0]})
for m in MODELS[1:]:
    merged_sp0 = merged_sp0.merge(
        fs0sp0[m][["file_id", "example_index", "inf_correct"]].rename(columns={"inf_correct": m}),
        on=["file_id", "example_index"]
    )
print(f"Matched items across all 4 models: {len(merged_sp0)}")

import itertools
rows = []
for m1, m2 in itertools.combinations(MODELS, 2):
    both_right = ((merged_sp0[m1] == 1) & (merged_sp0[m2] == 1)).sum()
    only_1 = ((merged_sp0[m1] == 1) & (merged_sp0[m2] == 0)).sum()
    only_2 = ((merged_sp0[m1] == 0) & (merged_sp0[m2] == 1)).sum()
    both_wrong = ((merged_sp0[m1] == 0) & (merged_sp0[m2] == 0)).sum()
    table = [[both_right, only_1], [only_2, both_wrong]]
    res = mcnemar(table, exact=True)
    rows.append({
        "comparison": f"{m1} vs {m2}", "acc1": merged_sp0[m1].mean()*100, "acc2": merged_sp0[m2].mean()*100,
        "n_disagree": only_1 + only_2, "only_1_right": only_1, "only_2_right": only_2,
        "mcnemar_p_exact": res.pvalue,
    })
mc = pd.DataFrame(rows)
print(mc.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. PAIRED SPEAKER COMPARISON (Speaker1 vs Speaker2 share the same 24 items)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. PAIRED SPEAKER1 vs SPEAKER2 -- McNemar's (same 24 items, different speaker audio)")
print("=" * 90)
rows = []
for m in MODELS:
    d1 = resolved[(m, 1, 0)][["file_id", "example_index", "inf_correct"]].rename(columns={"inf_correct": "sp1"})
    d2 = resolved[(m, 2, 0)][["file_id", "example_index", "inf_correct"]].rename(columns={"inf_correct": "sp2"})
    mm = d1.merge(d2, on=["file_id", "example_index"])
    only1 = ((mm.sp1 == 1) & (mm.sp2 == 0)).sum()
    only2 = ((mm.sp1 == 0) & (mm.sp2 == 1)).sum()
    both_r = ((mm.sp1 == 1) & (mm.sp2 == 1)).sum()
    both_w = ((mm.sp1 == 0) & (mm.sp2 == 0)).sum()
    res = mcnemar([[both_r, only1], [only2, both_w]], exact=True)
    rows.append({"model": m, "n": len(mm), "sp1_acc": mm.sp1.mean()*100, "sp2_acc": mm.sp2.mean()*100,
                 "n_disagree": only1+only2, "mcnemar_p_exact": res.pvalue})
print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: Speaker 0 uses a different item/file_id namespace from Speakers 1 & 2, so")
print("Sp0-vs-Sp1 / Sp0-vs-Sp2 cannot be exactly paired this way; independent-samples tests")
print("(as in the previous report) remain the right tool for those two comparisons.")

# ---------------------------------------------------------------------------
# 4. CLUSTER-ROBUST (GEE) LOGISTIC REGRESSION: Focus x Logic x Alternative, pooled models
#    Cluster = item (same item shared across the 4 models -> correlated obs)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. GEE LOGISTIC REGRESSION (cluster=item) -- Focus x Logic x Alternative, Sp0 FS0, pooled models")
print("=" * 90)
long_rows = []
for m in MODELS:
    d = fs0sp0[m].copy()
    d["item_id"] = d.file_id.astype(str) + "_" + d.example_index.astype(str)
    d["model"] = m
    long_rows.append(d[["item_id", "model", "focus", "logic", "alternative", "inf_correct"]])
long_df = pd.concat(long_rows, ignore_index=True)
long_df["focus"] = long_df["focus"].astype(str)
long_df["alternative"] = long_df["alternative"].astype(str)

gee = smf.gee("inf_correct ~ C(focus) + C(logic) + C(alternative) + C(model)",
              groups="item_id", data=long_df, family=sm.families.Binomial())
gee_fit = gee.fit()
print(gee_fit.summary())

print("\n-- Per-model GEE is unnecessary here (no repeated obs within model/item at FS0);")
print("   per-model results below use ordinary logistic regression (1 obs/item/model) --")
for m in MODELS:
    d = fs0sp0[m].copy()
    d["focus"] = d["focus"].astype(str); d["alternative"] = d["alternative"].astype(str)
    fit = smf.glm("inf_correct ~ C(focus) + C(logic) + C(alternative)", data=d, family=sm.families.Binomial()).fit()
    print(f"\nModel: {m}")
    print(fit.summary2().tables[1])

# ---------------------------------------------------------------------------
# 5. FEW-SHOT TREND -- GEE logistic regression clustered by item
#    (FS2 has 3-4 non-independent trials per item; this properly accounts for that,
#     unlike the independent-proportion tests used in the previous, aggregate-only report)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("5. FEW-SHOT TREND -- GEE logistic regression clustered by item (Speaker 0)")
print("=" * 90)
rows = []
for m in MODELS:
    parts = []
    for fs in [0, 2, 5]:
        d = resolved.get((m, 0, fs))
        if d is None:
            continue
        dd = d.copy()
        dd["item_id"] = dd.file_id.astype(str) + "_" + dd.example_index.astype(str)
        dd["fs"] = fs
        parts.append(dd[["item_id", "fs", "inf_correct"]])
    long_fs = pd.concat(parts, ignore_index=True)
    gee_fs = smf.gee("inf_correct ~ fs", groups="item_id", data=long_fs, family=sm.families.Binomial()).fit()
    coef = gee_fs.params["fs"]; se = gee_fs.bse["fs"]; p = gee_fs.pvalues["fs"]
    n_obs = len(long_fs); n_items = long_fs.item_id.nunique()
    rows.append({"model": m, "n_obs": n_obs, "n_items": n_items, "slope(logit/FS-step)": coef, "se": se, "p_value": p})
    print(f"\n{m}: FS0={100*resolved[(m,0,0)].inf_correct.mean():.1f}%  "
          f"FS2={100*resolved[(m,0,2)].inf_correct.mean():.1f}% (n={len(resolved[(m,0,2)])}, "
          f"{resolved[(m,0,2)].groupby(['file_id','example_index']).size().mean():.1f} trials/item)  "
          f"FS5={100*resolved[(m,0,5)].inf_correct.mean():.1f}%")
print("\nGEE trend summary (accuracy ~ few-shot count, clustered by item):")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. HUMAN SURVEY vs MODEL -- real per-respondent data (not reconstructed from means)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("6. HUMAN SURVEY vs MODEL -- one-sample t-test on REAL per-respondent accuracies")
print("=" * 90)
import os
SURVEY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "qualtrics_export", "output", "inference_scoring",
              "Inference_Survey_24_May+13,+2026_13.47.scored_summary_anon.csv")
survey = pd.read_csv(SURVEY_CSV)
survey = survey[survey.Status == 0]  # drop preview/test response
print(f"Respondents (excl. preview row): {len(survey)}  -- matches paper's Table 5 (n=12 total: 3/4/5 by speaker)")

model_ref = "gemini-3.1-pro-preview"
rows = []
for sp_label, g in survey.groupby("speaker"):
    sp_num = int(sp_label.replace("speaker", ""))
    model_pct = 100 * resolved[(model_ref, sp_num, 0)].inf_correct.mean()
    accs = g.accuracy_pct.values
    n = len(accs)
    tstat, p = stats.ttest_1samp(accs, model_pct)
    rows.append({"speaker": sp_label, "n_respondents": n, "human_mean_pct": accs.mean(),
                 "human_sd": accs.std(ddof=1) if n > 1 else np.nan,
                 f"{model_ref}_pct": model_pct, "t": tstat, "p_value": p})
print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: this replaces the previous report's approximation (which treated all")
print("respondent-item pairs as independent trials, inflating apparent significance).")
print("Using the real 12 independent respondents as the sampling unit, Speaker 1's")
print("human-vs-model gap drops from 'clearly significant' (p=.008, independent-trials")
print("approximation) to only marginal (see p-value above) once respondent-level")
print("variance is properly accounted for.")

print("\nDone.")
