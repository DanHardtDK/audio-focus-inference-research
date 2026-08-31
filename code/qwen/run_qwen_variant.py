"""run_qwen_variant.py — Run all 4 Tasks for one (model, precision) Qwen variant.

Generates the same Task1 / Task2 / Task3 / Task4 folder structure as the
original 4-bit pipeline, but inside a per-variant parent directory so multiple
model/precision variants live side-by-side without overwriting each other.

Tasks:
  Task 1  Layer-by-layer probe on the audio encoder, target = focus  (binary)
  Task 2  Per-clip transcription with focus marked in UPPERCASE       (qwen_focus.py)
  Task 3  Layer-by-layer probe on the audio encoder, target = inference (3-class A/B/C)
  Task 4  Per-clip A/B/C inference                                    (qwenAudioInput.py)

Layout produced (with --variant-tag qwen_fp16):
    data/output/qwen_fp16/
        internals/   (encoder CSVs + logits)   <- input to probes
        Task1/       (focus  probe results)
        Task2/       (transcription / focus identification)
        Task3/       (inference probe results)
        Task4/       (per-clip A/B/C inference)

The original quantised pipeline (run_qwen.py) is untouched.

Usage (from repo root):
    # Unquantised Qwen2-Audio-7B-Instruct, all f-files, speaker0
    python code/qwen/run_qwen_variant.py \\
        --model Qwen/Qwen2-Audio-7B-Instruct \\
        --precision fp16 \\
        --variant-tag qwen2_audio_fp16

    # Qwen2.5-Omni-7B at fp16
    python code/qwen/run_qwen_variant.py \\
        --model Qwen/Qwen2.5-Omni-7B \\
        --precision fp16 \\
        --variant-tag qwen2_5_omni_fp16

    # Add cross-speaker held-out probe data (requires those speaker dirs)
    python code/qwen/run_qwen_variant.py \\
        --model Qwen/Qwen2-Audio-7B-Instruct \\
        --precision fp16 \\
        --variant-tag qwen2_audio_fp16 \\
        --cross-speakers speaker1 speaker2
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Match the original quantised baseline (f1..f13 -> N=128 probe samples)
DEFAULT_F_FILES  = [f"f{i}" for i in range(1, 14)]
DEFAULT_NS_FILES = ["ns1", "ns2", "ns3"]

PYTHON  = sys.executable                     # use the same interpreter as this script
ROOT    = os.path.dirname(os.path.abspath(__file__))
RUNNER  = os.path.join(ROOT, "qwenAudioInput.py")
FOCUS   = os.path.join(ROOT, "qwen_focus.py")
INSPECT = os.path.join(ROOT, "qwen_inspect.py")
PROBE   = os.path.join(ROOT, "layer_probe.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, label):
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*68}")
    rc = subprocess.run(cmd, text=True).returncode
    print(f"\n  [{'done' if rc == 0 else f'FAILED (exit {rc})'}]")
    return rc


def newest(pattern):
    """Return the newest file matching glob pattern (or None)."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Run all 4 Qwen tasks for one (model, precision) variant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",      required=True,
                   help="HuggingFace model id (e.g. Qwen/Qwen2-Audio-7B-Instruct)")
    p.add_argument("--precision",  default="fp16",
                   choices=["4bit", "8bit", "fp16", "bf16"],
                   help="Model precision (default: fp16)")
    p.add_argument("--variant-tag", required=True,
                   help="Folder name for this variant under data/output/ "
                        "(e.g. qwen2_audio_fp16, qwen2_5_omni_fp16)")
    p.add_argument("--speaker",    default="speaker0",
                   help="Training speaker (default: speaker0)")
    p.add_argument("--files",      nargs="*", default=DEFAULT_F_FILES,
                   help="File IDs for the training speaker (default: f1..f13, matches 4-bit baseline)")
    p.add_argument("--cross-speakers", nargs="*", default=[],
                   help="Other speaker dirs to extract held-out probe data for "
                        "(uses --ns-files). E.g. speaker1 speaker2")
    p.add_argument("--ns-files",   nargs="*", default=DEFAULT_NS_FILES,
                   help="File IDs for cross-speaker probe data (default: ns1 ns2 ns3)")
    p.add_argument("--output-root", default=os.path.join("data", "output"),
                   help="Parent directory for variant folders (default: data/output)")
    p.add_argument("--skip-task1", action="store_true", help="Skip focus probe")
    p.add_argument("--skip-task2", action="store_true", help="Skip qwen_focus inference")
    p.add_argument("--skip-task3", action="store_true", help="Skip inference probe")
    p.add_argument("--skip-task4", action="store_true", help="Skip A/B/C inference")
    p.add_argument("--skip-internals", action="store_true",
                   help="Skip encoder/logit extraction (assumes Task1/Task3 inputs already exist)")
    p.add_argument("--focus-hint-too", action="store_true",
                   help="Also run Task 4 with --focus-hint as a separate condition")
    args = p.parse_args()

    base = os.path.join(args.output_root, args.variant_tag)
    dirs = {
        "internals": os.path.join(base, "internals"),
        "Task1":     os.path.join(base, "Task1"),
        "Task2":     os.path.join(base, "Task2"),
        "Task3":     os.path.join(base, "Task3"),
        "Task4":     os.path.join(base, "Task4"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    start = datetime.utcnow()
    print(f"\n{'#'*68}")
    print(f"  Qwen variant runner")
    print(f"  model       : {args.model}")
    print(f"  precision   : {args.precision}")
    print(f"  variant tag : {args.variant_tag}")
    print(f"  speaker     : {args.speaker}")
    print(f"  files       : {' '.join(args.files)}")
    print(f"  output root : {base}")
    print(f"  started     : {start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'#'*68}")

    failed = []

    # -----------------------------------------------------------------------
    # Internals — encoder all-layers + logits (input to Task 1 + Task 3)
    # -----------------------------------------------------------------------
    if not args.skip_internals and not (args.skip_task1 and args.skip_task3):
        file_args = []
        for fid in args.files:
            file_args += ["--file", fid]

        cmd = [
            PYTHON, INSPECT,
            "--model",      args.model,
            "--precision",  args.precision,
            "--speaker",    args.speaker,
            "--wav-dir",    os.path.join("data", "speakers", args.speaker, "clips"),
            "--output-dir", dirs["internals"],
            "--all-layers",
        ] + file_args
        if run(cmd, f"INTERNALS — {args.speaker} f-files (encoder + logits)") != 0:
            failed.append("internals_train")

        # Cross-speaker held-out
        for sp in args.cross_speakers:
            file_args2 = []
            for fid in args.ns_files:
                file_args2 += ["--file", fid]
            cmd2 = [
                PYTHON, INSPECT,
                "--model",      args.model,
                "--precision",  args.precision,
                "--speaker",    sp,
                "--wav-dir",    os.path.join("data", "speakers", sp, "clips"),
                "--output-dir", dirs["internals"],
                "--all-layers",
            ] + file_args2
            if run(cmd2, f"INTERNALS — {sp} ns-files (cross-speaker)") != 0:
                failed.append(f"internals_{sp}")

    # -----------------------------------------------------------------------
    # Task 2 — qwen_focus.py (transcription + focus marking)
    # -----------------------------------------------------------------------
    if not args.skip_task2:
        cmd = [
            PYTHON, FOCUS,
            "--model",      args.model,
            "--precision",  args.precision,
            "--speaker",    args.speaker,
            "--output-dir", dirs["Task2"],
        ] + args.files
        if run(cmd, f"TASK 2 — focus identification ({args.speaker} f-files)") != 0:
            failed.append("Task2")

        # Cross-speaker held-out (uses ns-files) — matches 4-bit baseline layout
        for sp in args.cross_speakers:
            cmd_cs = [
                PYTHON, FOCUS,
                "--model",      args.model,
                "--precision",  args.precision,
                "--speaker",    sp,
                "--output-dir", dirs["Task2"],
            ] + args.ns_files
            if run(cmd_cs, f"TASK 2 — focus identification ({sp} ns-files)") != 0:
                failed.append(f"Task2_{sp}")

    # -----------------------------------------------------------------------
    # Task 4 — qwenAudioInput.py (A/B/C inference, FS0)
    # -----------------------------------------------------------------------
    if not args.skip_task4:
        cmd = [
            PYTHON, RUNNER,
            "--model",      args.model,
            "--precision",  args.precision,
            "--fewshot",    "0",
            "--speaker",    args.speaker,
            "--output-dir", dirs["Task4"],
        ] + args.files
        if run(cmd, f"TASK 4 — A/B/C inference FS0 ({args.speaker} f-files)") != 0:
            failed.append("Task4")

        if args.focus_hint_too:
            cmd_fh = [
                PYTHON, RUNNER,
                "--model",      args.model,
                "--precision",  args.precision,
                "--fewshot",    "0",
                "--focus-hint",
                "--speaker",    args.speaker,
                "--output-dir", dirs["Task4"],
            ] + args.files
            if run(cmd_fh, f"TASK 4 — A/B/C inference FS0+FH ({args.speaker} f-files)") != 0:
                failed.append("Task4_FH")

        # Cross-speaker held-out (uses ns-files) — matches 4-bit baseline layout
        for sp in args.cross_speakers:
            cmd_cs = [
                PYTHON, RUNNER,
                "--model",      args.model,
                "--precision",  args.precision,
                "--fewshot",    "0",
                "--speaker",    sp,
                "--output-dir", dirs["Task4"],
            ] + args.ns_files
            if run(cmd_cs, f"TASK 4 — A/B/C inference FS0 ({sp} ns-files)") != 0:
                failed.append(f"Task4_{sp}")

            if args.focus_hint_too:
                cmd_cs_fh = [
                    PYTHON, RUNNER,
                    "--model",      args.model,
                    "--precision",  args.precision,
                    "--fewshot",    "0",
                    "--focus-hint",
                    "--speaker",    sp,
                    "--output-dir", dirs["Task4"],
                ] + args.ns_files
                if run(cmd_cs_fh, f"TASK 4 — A/B/C inference FS0+FH ({sp} ns-files)") != 0:
                    failed.append(f"Task4_FH_{sp}")

    # -----------------------------------------------------------------------
    # Task 1 + Task 3 — layer_probe.py on the encoder CSVs we just produced
    # -----------------------------------------------------------------------
    train_layers_csv = newest(os.path.join(
        dirs["internals"],
        f"qwen_encoder_layers_{args.speaker}_*.csv",
    ))
    if train_layers_csv is None:
        if not args.skip_task1: failed.append("Task1_no_layers_csv")
        if not args.skip_task3: failed.append("Task3_no_layers_csv")
    else:
        # Collect optional cross-speaker test CSVs
        test_csvs = []
        for sp in args.cross_speakers:
            t = newest(os.path.join(
                dirs["internals"], f"qwen_encoder_layers_{sp}_*.csv"
            ))
            if t:
                test_csvs.append(t)

        if not args.skip_task1:
            cmd = [
                PYTHON, PROBE,
                "--layers-csv", train_layers_csv,
                "--target",     "focus",
                "--pca",        "50",
                "--output-dir", dirs["Task1"],
            ]
            if test_csvs:
                cmd += ["--test-layers-csvs", *test_csvs]
            if run(cmd, "TASK 1 — focus probe") != 0:
                failed.append("Task1")

        if not args.skip_task3:
            cmd = [
                PYTHON, PROBE,
                "--layers-csv", train_layers_csv,
                "--target",     "inference",
                "--pca",        "50",
                "--output-dir", dirs["Task3"],
            ]
            if test_csvs:
                cmd += ["--test-layers-csvs", *test_csvs]
            if run(cmd, "TASK 3 — inference probe") != 0:
                failed.append("Task3")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = (datetime.utcnow() - start).seconds // 60
    print(f"\n{'#'*68}")
    print(f"  DONE — elapsed ~{elapsed} min  (variant: {args.variant_tag})")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"  All steps completed successfully.")
    print(f"  Output: {base}")
    print(f"{'#'*68}\n")


if __name__ == "__main__":
    main()
