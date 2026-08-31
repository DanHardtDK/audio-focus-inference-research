import pandas as pd
from pathlib import Path

task4 = [
    "data/output/qwen/Task4/master_inference_audio_qwen_Qwen-Qwen2-Audio-7B-Instruct_SPspeaker0_FS0_20260520_095418.csv",
    "data/output/qwen2_audio_fp16/Task4/master_inference_audio_qwen_Qwen-Qwen2-Audio-7B-Instruct_SPspeaker0_FS0_20260521_180950.csv",
    "data/output/qwen2_5_omni_fp16/Task4/master_inference_audio_qwen_Qwen-Qwen2.5-Omni-7B_SPspeaker0_FS0_20260521_185400.csv",
]

print("="*80)
print("TASK 4 ANALYSIS")
print("="*80)
for p in task4:
    print(f"\n--- {p} ---")
    df = pd.read_csv(p)
    print(f"Total rows N = {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "inf_correct" in df.columns:
        print(f"Overall accuracy (mean inf_correct) = {df['inf_correct'].mean():.4f}")
    if "true_A" in df.columns and "inf_correct" in df.columns:
        print("Breakdown by true_A:")
        for cls in ["A","B","C"]:
            sub = df[df["true_A"]==cls]
            if len(sub):
                print(f"  true_A={cls}: count={len(sub)}, acc={sub['inf_correct'].mean():.4f}")
            else:
                print(f"  true_A={cls}: count=0")
    if "model_A" in df.columns:
        vc = df["model_A"].astype(str).str.strip()
        valid = vc.isin(["A","B","C"])
        print("model_A prediction distribution:")
        for cls in ["A","B","C"]:
            print(f"  predicted {cls}: {(vc==cls).sum()}")
        empty = vc.isin(["","nan","None"]).sum()
        other = (~valid & ~vc.isin(["","nan","None"])).sum()
        print(f"  empty/nan: {empty}")
        print(f"  other: {other}")
        print(f"  fraction valid (A/B/C): {valid.mean():.4f}")

task2 = [
    "data/output/qwen/Task2/master_transcription_audio_qwen_Qwen-Qwen2-Audio-7B-Instruct_SPspeaker0_FS0_20260520_110525.csv",
    "data/output/qwen/Task2/master_transcription_audio_qwen_Qwen-Qwen2-Audio-7B-Instruct_SPspeaker0_FS0_20260520_111510.csv",
    "data/output/qwen2_audio_fp16/Task2/master_transcription_audio_qwen_Qwen-Qwen2-Audio-7B-Instruct_SPspeaker0_FS0_20260521_180718.csv",
    "data/output/qwen2_5_omni_fp16/Task2/master_transcription_audio_qwen_Qwen-Qwen2.5-Omni-7B_SPspeaker0_FS0_20260521_185209.csv",
]

print("\n" + "="*80)
print("TASK 2 ANALYSIS")
print("="*80)
for p in task2:
    print(f"\n--- {p} ---")
    df = pd.read_csv(p)
    print(f"Total rows N = {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "trans_correct" in df.columns:
        print(f"Mean trans_correct = {df['trans_correct'].mean():.4f}")
    else:
        print("(no trans_correct column)")
