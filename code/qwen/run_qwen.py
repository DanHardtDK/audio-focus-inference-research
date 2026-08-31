"""run_qwen.py — Run Qwen2-Audio inference and/or internals extraction.

Inference conditions (matching the paper's experimental design):
  FS0              zero-shot, no focus hint
  FS0_FH           zero-shot, with focus hint
  FS2_CV           2-shot, 5-fold cross-validation
  FS2_CV_FH        2-shot, 5-fold CV, with focus hint
  FS5_CV           5-shot, 2-fold cross-validation
  FS5_CV_FH        5-shot, 2-fold CV, with focus hint

Internals (via qwen_inspect.py):
  logit probabilities  p(A) / p(B) / p(C) at the first generated token
  encoder vectors      mean-pooled 1280-dim audio tower output + cosine heatmap

Files are processed one at a time per condition so that a crash does not lose
all results and VRAM is fully released between runs.

Usage (from repo root):
    # All conditions + internals for all files
    python code/qwen/run_qwen.py

    # Only inference, specific files
    python code/qwen/run_qwen.py --files f1 f2 f3 --skip-inspect

    # Only internals (no new inference)
    python code/qwen/run_qwen.py --skip-inference

    # Skip FS0 conditions (already run)
    python code/qwen/run_qwen.py --skip-fs0

    # Custom output directory
    python code/qwen/run_qwen.py --output-dir data/output/qwen
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_FILE_IDS = [
    "f1", "f2", "f3", "f4", "f5",
    "f6", "f7", "f8", "f9", "f10",
    "ns1", "ns2", "ns3",
]

MODEL   = "Qwen/Qwen2-Audio-7B-Instruct"
PYTHON  = os.path.join("venv312", "Scripts", "python")
RUNNER  = os.path.join("code", "qwen", "qwenAudioInput.py")
INSPECT = os.path.join("code", "qwen", "qwen_inspect.py")

# Conditions: (label, fewshot, use_cv, focus_hint)
CONDITIONS = [
    ("FS0",       0, False, False),
    ("FS0_FH",    0, False, True),
    ("FS2_CV",    2, True,  False),
    ("FS2_CV_FH", 2, True,  True),
    ("FS5_CV",    5, True,  False),
    ("FS5_CV_FH", 5, True,  True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    rc = subprocess.run(cmd, text=True).returncode
    status = "done" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n  [{status}]")
    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2-Audio inference and/or internals extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files", nargs="*", default=ALL_FILE_IDS, metavar="FILEID",
        help="File IDs to process (default: all 13)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join("data", "output", "qwen"),
        help="Output directory for CSVs and logs (default: data/output/qwen)",
    )
    parser.add_argument("--model",   default=MODEL)
    parser.add_argument("--speaker", default="speaker0")
    parser.add_argument(
        "--skip-fs0", action="store_true",
        help="Skip FS0 and FS0_FH conditions (already run)",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip all inference, only run internals",
    )
    parser.add_argument(
        "--skip-inspect", action="store_true",
        help="Skip internals extraction, only run inference",
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Pass --all-layers to qwen_inspect.py to extract per-layer encoder representations",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.utcnow()
    total_files = len(args.files)

    print(f"\n{'#'*60}")
    print(f"  Qwen2-Audio runner — {start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Files     : {', '.join(args.files)}")
    print(f"  Model     : {args.model}")
    print(f"  Speaker   : {args.speaker}")
    print(f"  Output    : {args.output_dir}")
    print(f"{'#'*60}")

    failed = []

    # -----------------------------------------------------------------------
    # 1. Inference — one file at a time per condition
    # -----------------------------------------------------------------------
    if not args.skip_inference:
        conditions = [
            c for c in CONDITIONS
            if not (args.skip_fs0 and c[0].startswith("FS0"))
        ]

        for (label, fewshot, use_cv, focus_hint) in conditions:
            print(f"\n{'─'*60}")
            print(f"  CONDITION: {label}")
            print(f"{'─'*60}")

            for i, file_id in enumerate(args.files, 1):
                cmd = [
                    PYTHON, RUNNER,
                    "--model",      args.model,
                    "--fewshot",    str(fewshot),
                    "--speaker",    args.speaker,
                    "--output-dir", args.output_dir,
                    file_id,
                ]
                if use_cv:
                    cmd.append("--cv")
                if focus_hint:
                    cmd.append("--focus-hint")

                rc = run(cmd, f"[{i}/{total_files}] {label} — {file_id}")
                if rc != 0:
                    failed.append(f"{label}:{file_id}")
    else:
        print("\n[Skipping inference conditions]")

    # -----------------------------------------------------------------------
    # 2. Internals — logits + encoder vectors (qwen_inspect.py)
    # -----------------------------------------------------------------------
    if not args.skip_inspect:
        file_args = []
        for fid in args.files:
            file_args += ["--file", fid]

        cmd = [
            PYTHON, INSPECT,
            "--speaker",    args.speaker,
            "--wav-dir",    os.path.join("data", "speakers", args.speaker, "clips"),
            "--output-dir", args.output_dir,
        ] + file_args
        if args.all_layers:
            cmd.append("--all-layers")

        rc = run(cmd, "Internals — logits + encoder vectors")
        if rc != 0:
            failed.append("INSPECT")
    else:
        print("\n[Skipping internals inspection]")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = (datetime.utcnow() - start).seconds // 60
    print(f"\n{'#'*60}")
    print(f"  DONE — elapsed ~{elapsed} min")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All steps completed successfully.")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
