#!/usr/bin/env python3
"""Convert inference-survey JSON items into a Qualtrics Advanced TXT survey.

Usage:
    python3 code/qualtrics_export/qualtrics_inference_export.py INPUT_JSON OUTPUT_TXT

Example:
    python3 code/qualtrics_export/qualtrics_inference_export.py \
        code/qualtrics_export/input/set2_inference_items.json \
        code/qualtrics_export/output/set2_inference_survey.txt

What it does:
    - Reads a JSON array of inference-survey items.
    - Converts each item into one Qualtrics multiple-choice question.
    - Writes a Qualtrics Advanced TXT file ready for import.
    - Writes a companion CSV mapping each question to its matching audio clip.
    - Prints a preview of the first 3 generated questions.

Answer mapping:
    - A -> Sentence 2 must be true.
    - B -> Sentence 2 might be true.
    - C -> Sentence 2 must be false.

Audio mapping:
    - By default, the script looks for clips in data/clips/set2.
    - If the input file is named like f1.json, question 1 maps to f1_item0.wav,
      question 2 maps to f1_item1.wav, etc.
    - For custom mixed subsets, include item-level metadata like source_json,
      source_item_index, or audio_file if you want automatic audio mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalize_s2
from qualtrics_export_common import (
    build_audio_map_rows,
    load_items,
    sanitize_qualtrics_text,
    write_audio_map,
)


ADVANCED_FORMAT_HEADER = "[[AdvancedFormat]]"
BLOCK_HEADER = "[[Block:Inference Survey]]"
QUESTION_HEADER = "[[Question:MC:SingleAnswer:Vertical]]"
CHOICES_HEADER = "[[Choices]]"

LABEL_MAP = {
    "A": "Sentence 2 must be true.",
    "B": "Sentence 2 might be true.",
    "C": "Sentence 2 must be false.",
}
INFERENCE_CHOICES = [
    "Sentence 2 must be true.",
    "Sentence 2 might be true.",
    "Sentence 2 must be false.",
]


def build_qualtrics_question(item: dict[str, Any], idx: int) -> str:
    """Build one Qualtrics Advanced TXT inference question block from a JSON item."""
    required_fields = ("S1", "S2", "A", "focus", "logic", "alternative")
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise KeyError(f"Item {idx} is missing required fields: {', '.join(missing_fields)}")

    label = sanitize_qualtrics_text(item["A"])
    if label not in LABEL_MAP:
        raise ValueError(f"Unsupported inference label {label!r} in item {idx}")

    s1 = sanitize_qualtrics_text(item.get("S1_normalized", normalize_s1(str(item["S1"]))))
    s2 = sanitize_qualtrics_text(item.get("S2_normalized", normalize_s2(str(item["S2"]))))
    correct_answer = LABEL_MAP[label]

    lines = [
        f"[[ED:item_id:{idx}]]",
        f"[[ED:label:{label}]]",
        f"[[ED:focus:{sanitize_qualtrics_text(item['focus'])}]]",
        f"[[ED:logic:{sanitize_qualtrics_text(item['logic'])}]]",
        f"[[ED:alternative:{sanitize_qualtrics_text(item['alternative'])}]]",
        f"[[ED:correct_answer:{sanitize_qualtrics_text(correct_answer)}]]",
        QUESTION_HEADER,
        "Given Sentence 1, what can we say about Sentence 2?<br><br>",
        f"Sentence 1: {s1}<br>",
        f"Sentence 2: {s2}<br>",
        CHOICES_HEADER,
        *INFERENCE_CHOICES,
    ]
    return "\n".join(lines)


def build_survey(items: list[dict[str, Any]]) -> str:
    """Build the complete Qualtrics Advanced TXT inference survey."""
    question_blocks = [build_qualtrics_question(item, idx) for idx, item in enumerate(items, start=1)]
    return "\n\n".join([ADVANCED_FORMAT_HEADER, BLOCK_HEADER, *question_blocks]) + "\n"


def print_preview(items: list[dict[str, Any]], preview_count: int = 3) -> None:
    """Print a small console preview for the first few inference questions."""
    print(f"Previewing first {min(preview_count, len(items))} question(s):")
    for idx, item in enumerate(items[:preview_count], start=1):
        label = sanitize_qualtrics_text(item["A"])
        print()
        print(f"Question {idx}")
        print(f"  Sentence 1: {sanitize_qualtrics_text(item.get('S1_normalized', normalize_s1(str(item['S1']))))}")
        print(f"  Sentence 2: {sanitize_qualtrics_text(item.get('S2_normalized', normalize_s2(str(item['S2']))))}")
        print(f"  Correct answer: {LABEL_MAP[label]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an inference-survey JSON array into a Qualtrics Advanced TXT import file."
    )
    parser.add_argument("input_json", type=Path, help="Path to the input JSON file")
    parser.add_argument("output_txt", type=Path, help="Path to the output Qualtrics TXT file")
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/clips/set2"),
        help="Directory containing audio clips for manual Qualtrics attachment",
    )
    parser.add_argument(
        "--audio-map",
        type=Path,
        help="Optional CSV path for question-to-audio mapping; defaults next to OUTPUT_TXT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.input_json)
    survey_text = build_survey(items)
    audio_map_path = args.audio_map or args.output_txt.with_suffix(".audio_map.csv")
    audio_rows = build_audio_map_rows(items, args.input_json, args.clips_dir)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(survey_text, encoding="utf-8")
    write_audio_map(audio_rows, audio_map_path)

    print_preview(items, preview_count=3)
    print()
    print(f"Wrote {len(items)} question(s) to {args.output_txt}")
    print(f"Wrote audio mapping CSV to {audio_map_path}")

    matched_audio = sum(row["audio_exists"] == "yes" for row in audio_rows)
    print(f"Matched audio clips: {matched_audio}/{len(audio_rows)}")


if __name__ == "__main__":
    main()
