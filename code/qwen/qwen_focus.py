"""
qwen_focus.py — Qwen2-Audio focus-identification test (Task 2).

Mirrors the 'transcription' task in audioInput.py so results are directly
comparable across models (Gemini, GPT-4o, Qwen2-Audio).

For each S1 clip the model is asked to transcribe the sentence and mark the
prosodically emphasised word with ALL UPPERCASE.  Focus position is then
extracted from the model's transcription using the same focus_position()
logic as audioInput.py, and scored against the ground-truth `focus` field.

Scoring (trans_correct):
  1  — model uppercase matches ground truth (focus 1=person, 2=object)
  0  — mismatch or no uppercase detected

Output columns match the transcription master CSVs produced by audioInput.py
so all models can be compared in one table.

Usage (from repo root):
    python code/qwen/qwen_focus.py --speaker speaker0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime

import soundfile as sf
import torch

# Shared loader (handles precision + Qwen2-Audio / Qwen2.5-Omni families)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from qwen_loader import load_qwen, generate_text_only  # noqa: E402

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
BACKEND    = "qwen"
MODE       = "audio"

# Matches audioInput.py CSV_COLUMNS (+ speaker for Qwen outputs)
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

SYSTEM_PROMPT = (
    "You will hear a spoken sentence. Transcribe it. "
    "Write the one word that is spoken with the most prominent stress in ALL CAPITALS. "
    "All other words must be in normal casing. "
    "Reply with only the transcription — no preamble, no quotes, no other text. "
    "\n\nFormat example: Sam only gave BILL grapes."
)

# ---------------------------------------------------------------
# Focus-position constants (mirrors audioInput.py)
# ---------------------------------------------------------------
FOCUS_NONE   = 0
FOCUS_FIRST  = 1   # second-to-last token is uppercase (person)
FOCUS_SECOND = 2   # last token is uppercase (object)
FOCUS_BOTH   = 3


def focus_position(sentence: str) -> int:
    """
    Return FOCUS_FIRST, FOCUS_SECOND, FOCUS_BOTH, or FOCUS_NONE based on
    which of the last two alphabetic tokens in *sentence* are all-uppercase.
    Mirrors the function of the same name in audioInput.py.
    """
    tokens = re.findall(r"\b\w+\b", sentence)
    if len(tokens) < 2:
        return FOCUS_NONE
    last        = tokens[-1].isupper()
    second_last = tokens[-2].isupper()
    if last and second_last:
        return FOCUS_BOTH
    elif second_last:
        return FOCUS_FIRST
    elif last:
        return FOCUS_SECOND
    return FOCUS_NONE


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein distance normalised by max length (0 = identical)."""
    a, b = s1.lower(), s2.lower()
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / max(m, n)


# ---------------------------------------------------------------
# Model cache
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
# Single-clip transcription query
# ---------------------------------------------------------------
def transcribe_with_focus(audio_arr) -> str:
    """
    Ask the model to transcribe the S1 clip and mark the focused word
    in UPPERCASE.  Returns the raw model response string.
    """
    model, processor, target_sr = _model, _processor, _target_sr

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "clip.wav"},
                {"type": "text",
                 "text": "Transcribe this sentence, marking the emphasised word in UPPERCASE."},
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
            model, _family, **inputs, max_new_tokens=64, do_sample=False
        )
    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return _parse_transcription(raw)


def _parse_transcription(raw: str) -> str:
    """
    Extract the bare transcription sentence from a potentially verbose response.
    Handles patterns like:
      "The original content of this audio is: 'Sam only gave BILL grapes.'"
      "Sam only gave BILL grapes. Sam also gave Sue grapes."
      "S1: Sam only gave BILL grapes."
    Returns only the first sentence (S1).
    """
    # Strip opening preamble up to a colon followed by optional quote
    s = re.sub(r'^[^A-Z]*[A-Z][^:]*:\s*["\']?', '', raw).strip('"\' ')
    # If still looks like preamble (lowercase start, colon present), try simpler strip
    if re.match(r'^[a-z]', s) or not s:
        s = raw
    # Take only the first sentence (stop at first period/question/exclamation)
    m = re.match(r'([^.!?]+[.!?]?)', s)
    s = m.group(1).strip() if m else s.strip()
    return s


# ---------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------
def load_audio(path: str, target_sr: int):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError("librosa required for resampling.")
    return arr


# ---------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------
def write_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-Audio focus-identification test (transcription mode)"
    )
    parser.add_argument("files", nargs="+", metavar="FILEID",
                        help="File IDs to process (e.g. f1 f2 f3)")
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--precision",  default="4bit",
                        choices=["4bit", "8bit", "fp16", "bf16"],
                        help="Model precision (default: 4bit, matches original runs)")
    parser.add_argument("--speaker",    default="speaker0")
    parser.add_argument("--clips-dir",  default=None,
                        help="Clips directory (default: data/speakers/{speaker}/clips)")
    parser.add_argument("--json-dir",   default="data/stimuli")
    parser.add_argument("--output-dir",
                        default=os.path.join("data", "output", "qwen", "Task2"))
    args = parser.parse_args()

    clips_dir = args.clips_dir or os.path.join("data", "speakers", args.speaker, "clips")
    os.makedirs(args.output_dir, exist_ok=True)

    load_model(args.model, args.precision)

    safe_model    = args.model.replace("/", "-").replace("\\", "-")
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id        = (
        f"transcription_{MODE}_{BACKEND}_{safe_model}"
        f"_SP{args.speaker}_FS0_{run_timestamp}"
    )
    master_rows = []

    for file_id in args.files:
        print(f"\n=== {file_id} ===")

        json_path = os.path.join(args.json_dir, file_id + ".json")
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found — skipping")
            continue

        with open(json_path, encoding="utf-8") as f:
            items = json.load(f)

        rows = []
        for idx, item in enumerate(items):
            clip_path = os.path.join(clips_dir, f"{file_id}_item{idx}.wav")
            true_S1   = item.get("S1", "")
            true_S2   = item.get("S2", "")
            true_A    = item.get("A", "")
            focus_val = item.get("focus", "")
            logic     = item.get("logic", "")
            alt       = item.get("alternative", "")

            gold_pos = int(focus_val) if str(focus_val).isdigit() else FOCUS_NONE

            print(f"  [{idx}] {true_S1}", end=" … ", flush=True)

            if not os.path.exists(clip_path):
                print("MISSING CLIP")
                model_S1      = ""
                trans_correct = ""
                s1_edit_norm  = ""
            else:
                audio_arr    = load_audio(clip_path, _target_sr)
                model_S1     = transcribe_with_focus(audio_arr)
                model_pos    = focus_position(model_S1)
                trans_correct = int(
                    model_pos in (FOCUS_FIRST, FOCUS_SECOND)
                    and gold_pos in (FOCUS_FIRST, FOCUS_SECOND)
                    and model_pos == gold_pos
                )
                s1_edit_norm = round(normalized_edit_distance(true_S1, model_S1), 4)
                print(f"'{model_S1}'  trans_correct={trans_correct}")

            rows.append({
                "example_index":    idx,
                "is_few_shot":      0,
                "true_S1":          true_S1,
                "true_S2":          true_S2,
                "true_A":           true_A,
                "inf_correct":      "",
                "model_A":          "",
                "trans_correct":    trans_correct,
                "model_S1":         model_S1,
                "model_S2":         "",
                "model_explanation": "",
                "s1_edit_norm":     s1_edit_norm,
                "s2_edit_norm":     "",
                "focus":            focus_val,
                "logic":            logic,
                "alternative":      alt,
                "file_id":          file_id,
                "mode":             MODE,
                "backend":          BACKEND,
                "model_name":       args.model,
                "run_timestamp_utc": run_timestamp,
                "response_id":      "",
                "cv_fold":          "",
                "speaker":          args.speaker,
            })

        # Per-file CSV + accuracy summary
        file_csv  = os.path.join(args.output_dir, f"{file_id}_{run_id}.csv")
        write_csv(rows, file_csv)
        n_scored  = sum(1 for r in rows if r["trans_correct"] != "")
        n_correct = sum(1 for r in rows if r["trans_correct"] == 1)
        if n_scored:
            print(f"  {file_id}: {n_correct}/{n_scored} correct ({n_correct/n_scored:.1%})")

        master_rows.extend(rows)

    # Master CSV
    if master_rows:
        master_path = os.path.join(args.output_dir, f"master_{run_id}.csv")
        write_csv(master_rows, master_path)
        n_scored  = sum(1 for r in master_rows if r["trans_correct"] != "")
        n_correct = sum(1 for r in master_rows if r["trans_correct"] == 1)
        if n_scored:
            print(f"\nOverall: {n_correct}/{n_scored} correct ({n_correct/n_scored:.1%})")
        print(f"Master CSV: {master_path}")


if __name__ == "__main__":
    main()
