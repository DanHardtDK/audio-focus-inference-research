"""
qwenAudioInput.py — Qwen2-Audio inference runner (per-item clip approach).

Unlike audioInput.py (which builds a multi-item prompt and sends all audio at once),
this script calls the model once per item clip, then writes a CSV in the same format.
This avoids all prompt-parsing issues and is more reliable for Qwen.

Usage (from repo root):
    # Zero-shot
    python code/qwenAudioInput.py f1 f2 f3

    # With focus hint
    python code/qwenAudioInput.py f1 --focus-hint

    # 2-shot cross-validation (marks few-shot items; model still runs on all clips)
    python code/qwenAudioInput.py f1 --fewshot 2 --cv

    # 5-shot CV with hint
    python code/qwenAudioInput.py f1 --fewshot 5 --cv --focus-hint

    # Custom output dir
    python code/qwenAudioInput.py f1 --output-dir data/output/qwen
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import soundfile as sf
import torch

# Shared loader (handles precision + Qwen2-Audio / Qwen2.5-Omni families)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_loader import load_qwen, generate_text_only  # noqa: E402

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
MODEL_NAME  = "Qwen/Qwen2-Audio-7B-Instruct"
BACKEND     = "qwen"
MODE        = "audio"

CSV_COLUMNS = [
    "example_index", "is_few_shot",
    "true_S1", "true_S2", "true_A",
    "inf_correct", "model_A",
    "trans_correct",
    "model_S1", "model_S2", "model_explanation",
    "s1_edit_norm", "s2_edit_norm",
    "focus", "logic", "alternative",
    "file_id", "mode", "backend", "model_name",
    "run_timestamp_utc", "response_id", "cv_fold", "speaker",
]

SYSTEM_BASE = (
    "Your task:\n"
    "1. Listen to S1 and S2 from audio.\n"
    "2. Classify S2 relative to S1:\n"
    "   A = entailed\n"
    "   B = independent\n"
    "   C = contradicted.\n\n"
    "IMPORTANT:\n"
    "- Pay close attention to which word is prosodically focused in S1.\n"
    "- Reply with exactly one character: A, B, or C, followed by a brief explanation."
)

FOCUS_HINT = (
    "\n\n---------------------------------------------------------------\n"
    "FOCUS GUIDANCE\n"
    "---------------------------------------------------------------\n"
    "The classification depends on the focused element in S1, because of\n"
    "the presence of 'only', in the following way: 'Sam only gave TOM\n"
    "oranges' entails that Sam did not give anyone else oranges. On the\n"
    "other hand, 'Sam only gave Tom ORANGES' entails that Sam didn't give\n"
    "anything else to Tom.\n\n"
    "You must follow this logic in determining the inference. You must\n"
    "also refer to this logic in producing the explanation."
)


# ---------------------------------------------------------------
# Model loading (module-level cache)
# ---------------------------------------------------------------
_model     = None
_processor = None
_target_sr = None
_family    = None


def load_model(model_name: str = MODEL_NAME, precision: str = "4bit"):
    global _model, _processor, _target_sr, _family
    if _model is None:
        _model, _processor, _target_sr, _family = load_qwen(model_name, precision)
    return _model, _processor, _target_sr


# ---------------------------------------------------------------
# Single-clip inference
# ---------------------------------------------------------------
def classify_clip(audio_arr: "np.ndarray", focus_hint: bool = False) -> tuple[str, str]:
    """
    Run Qwen on a single audio array.
    Returns (label, explanation) where label is one of A / B / C.
    """
    model, processor, target_sr = _model, _processor, _target_sr

    system_msg = SYSTEM_BASE + (FOCUS_HINT if focus_hint else "")

    conversation = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "clip.wav"},
                {"type": "text",  "text": "Classify S2 relative to S1."},
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text],
        audio=[audio_arr],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = generate_text_only(
            model, _family, **inputs, max_new_tokens=128, do_sample=False
        )
    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Extract label — first A/B/C in output
    m = re.search(r'\b([ABC])\b', raw)
    label = m.group(1) if m else "B"
    # Everything after the label is explanation
    explanation = raw[m.end():].strip().lstrip(".,;: ") if m else raw
    return label, explanation


# ---------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------
def load_audio(path: str, target_sr: int) -> "np.ndarray":
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:          # stereo / multi-channel → mix down to mono
        arr = arr.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError("librosa required for resampling. pip install librosa")
    return arr


# ---------------------------------------------------------------
# CV fold construction (mirrors audioInput.py)
# ---------------------------------------------------------------
def make_cv_folds(total_num: int, fewshot_num: int):
    if fewshot_num <= 0:
        return [(None, [])]
    if total_num % fewshot_num != 0:
        raise ValueError(
            f"--cv requires total_num ({total_num}) divisible by fewshot_num ({fewshot_num})."
        )
    n_folds = total_num // fewshot_num
    return [
        (fold, list(range(fold * fewshot_num, (fold + 1) * fewshot_num)))
        for fold in range(n_folds)
    ]


# ---------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------
def write_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------
# Process one file × one fold
# ---------------------------------------------------------------
def process_file_fold(
    file_id: str,
    clips_dir: str,
    json_dir: str,
    fewshot_indices: list,
    fold_id,
    focus_hint: bool,
    model_name: str,
    run_timestamp: str,
    speaker: str,
    output_dir: str,
    log_f,
) -> list:
    """
    Run inference on all clips for file_id in one fold.
    Returns list of result dicts (one per item).
    """
    json_path = os.path.join(json_dir, file_id + ".json")
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found — skipping {file_id}", file=log_f)
        return []

    with open(json_path) as f:
        items = json.load(f)

    fewshot_set = set(fewshot_indices)
    results = []

    for idx, item in enumerate(items):
        clip_path = os.path.join(clips_dir, f"{file_id}_item{idx}.wav")
        true_A  = item.get("A", "")
        true_S1 = item.get("S1", "")
        true_S2 = item.get("S2", "")
        focus   = item.get("focus", "")
        logic   = item.get("logic", "")
        alt     = item.get("alternative", "")

        print(f"  {file_id}[{idx}] …", end=" ", flush=True)
        print(f"  [{idx}] {true_S1[:50]} / {true_S2[:40]}", file=log_f)

        if not os.path.exists(clip_path):
            print("MISSING CLIP — using default B", flush=True)
            print(f"    MISSING CLIP: {clip_path}", file=log_f)
            model_A      = "B"
            explanation  = "clip not found"
        else:
            audio_arr = load_audio(clip_path, _target_sr)
            model_A, explanation = classify_clip(audio_arr, focus_hint=focus_hint)
            print(f"pred={model_A}  {explanation[:60]}", flush=True)
            print(f"    pred={model_A}  {explanation}", file=log_f)

        inf_correct = int(model_A == true_A) if true_A else ""

        results.append({
            "example_index": idx,
            "is_few_shot":   1 if idx in fewshot_set else 0,
            "true_S1": true_S1,
            "true_S2": true_S2,
            "true_A":  true_A,
            "inf_correct": inf_correct,
            "model_A": model_A,
            "trans_correct": "",
            "model_S1": "",
            "model_S2": "",
            "model_explanation": explanation,
            "s1_edit_norm": "",
            "s2_edit_norm": "",
            "focus": focus,
            "logic": logic,
            "alternative": alt,
            "file_id": file_id,
            "mode": MODE,
            "backend": BACKEND,
            "model_name": model_name,
            "run_timestamp_utc": run_timestamp,
            "response_id": "",
            "cv_fold": str(fold_id) if fold_id is not None else "",
            "speaker": speaker,
        })

    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio per-clip inference")
    parser.add_argument("files", nargs="+", metavar="FILEID",
                        help="File IDs to process (e.g. f1 f2 ns1)")
    parser.add_argument("--model",   default=MODEL_NAME)
    parser.add_argument("--precision", default="4bit",
                        choices=["4bit", "8bit", "fp16", "bf16"],
                        help="Model precision (default: 4bit, matches original runs)")
    parser.add_argument("--fewshot", type=int, default=0, dest="fewshot_num",
                        help="Number of few-shot items per fold (default: 0)")
    parser.add_argument("--cv",      action="store_true",
                        help="Enable n-fold CV rotation of few-shot indices")
    parser.add_argument("--focus-hint", action="store_true", dest="focus_hint",
                        help="Add focus-hint instruction to each clip prompt")
    parser.add_argument("--speaker", default="speaker0")
    parser.add_argument("--clips-dir", default=None,
                        help="Clips directory (default: data/speakers/{speaker}/clips)")
    parser.add_argument("--json-dir",  default="data/stimuli")
    parser.add_argument("--output-dir", default=os.path.join("data", "output", "qwen"))
    args = parser.parse_args()

    clips_dir = args.clips_dir or os.path.join("data", "speakers", args.speaker, "clips")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once before processing any files
    load_model(args.model, args.precision)

    safe_model    = args.model.replace("/", "-").replace("\\", "-")
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    master_rows   = []

    FH_TEXT = "_focusHint" if args.focus_hint else ""
    CV_TEXT = "_CV"        if (args.cv and args.fewshot_num > 0) else ""
    SP_TEXT = f"_SP{args.speaker}" if args.speaker else ""

    for file_id in args.files:
        print(f"\n=== {file_id} ===")

        # Load JSON to know total_num for fold computation
        json_path = os.path.join(args.json_dir, file_id + ".json")
        if not os.path.exists(json_path):
            print(f"  ERROR: {json_path} not found — skipping")
            continue
        with open(json_path) as f:
            items = json.load(f)
        total_num = len(items)

        # Build folds
        if args.cv and args.fewshot_num > 0:
            folds = make_cv_folds(total_num, args.fewshot_num)
        else:
            folds = [(None, list(range(args.fewshot_num)))]

        for (fold_id, fewshot_indices) in folds:
            fold_tag = f"_cv{fold_id}" if fold_id is not None else ""
            run_id   = (
                f"inference_{MODE}_{BACKEND}_{safe_model}"
                f"{SP_TEXT}_FS{args.fewshot_num}{FH_TEXT}{CV_TEXT}{fold_tag}"
                f"_{run_timestamp}"
            )

            log_path = os.path.join(args.output_dir, f"{file_id}_{run_id}.log")
            csv_path = os.path.join(args.output_dir, f"{file_id}_{run_id}.csv")

            with open(log_path, "w", encoding="utf-8") as log_f:
                print(
                    f"=== Log for {file_id} fold={fold_id} fewshot={fewshot_indices} ===",
                    file=log_f,
                )

                rows = process_file_fold(
                    file_id=file_id,
                    clips_dir=clips_dir,
                    json_dir=args.json_dir,
                    fewshot_indices=fewshot_indices,
                    fold_id=fold_id,
                    focus_hint=args.focus_hint,
                    model_name=args.model,
                    run_timestamp=run_timestamp,
                    speaker=args.speaker,
                    output_dir=args.output_dir,
                    log_f=log_f,
                )

                if not rows:
                    print(f"  No results for {file_id}{fold_tag} — skipping CSV write.", file=log_f)
                    continue

                n_test = sum(1 for r in rows if r["is_few_shot"] == 0 and r["inf_correct"] != "")
                n_correct = sum(1 for r in rows if r["is_few_shot"] == 0 and r["inf_correct"] == 1)
                acc = n_correct / n_test if n_test else float("nan")

                print(
                    f"  {file_id}{fold_tag}: {n_correct}/{n_test} correct on test items "
                    f"({acc:.1%})",
                    file=log_f,
                )
                print(
                    f"  {file_id}{fold_tag}: {n_correct}/{n_test} correct ({acc:.1%})"
                )

                write_csv(rows, csv_path)
                print(f"  CSV: {csv_path}", file=log_f)

            master_rows.extend(rows)

    # Master CSV
    if master_rows:
        master_id = (
            f"inference_{MODE}_{BACKEND}_{safe_model}"
            f"{SP_TEXT}_FS{args.fewshot_num}{FH_TEXT}{CV_TEXT}_{run_timestamp}"
        )
        master_path = os.path.join(args.output_dir, f"master_{master_id}.csv")
        write_csv(master_rows, master_path)
        print(f"\nMaster CSV: {master_path}")


if __name__ == "__main__":
    main()
