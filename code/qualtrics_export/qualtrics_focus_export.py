#!/usr/bin/env python3
"""Convert focus-survey JSON items into a Qualtrics Advanced TXT survey.

Usage:
    python3 code/qualtrics_export/qualtrics_focus_export.py INPUT_JSON OUTPUT_TXT

Example:
    python3 code/qualtrics_export/qualtrics_focus_export.py \
        code/qualtrics_export/input/f1.json \
        code/qualtrics_export/output/f1_focus_survey.txt

What it does:
    - Reads a JSON array of focus-survey items.
    - Converts each item into one Qualtrics multiple-choice question.
    - Writes a Qualtrics Advanced TXT file ready for import.
    - Writes a companion CSV mapping each question to its matching audio clip.
    - Prints a preview of the first 3 generated questions.

Audio mapping:
    - By default, the script looks for clips in data/clips/set1.
    - If the input file is named like f1.json, question 1 maps to f1_item0.wav,
      question 2 maps to f1_item1.wav, etc.
    - For custom mixed subsets, include item-level metadata like source_json,
      source_item_index, or audio_file if you want automatic audio mapping.

Input item shape:
    {
        "S1": "Sam only gave ROB bananas.",
        "S2": "Sam didn't give Tom bananas.",
        "A": "A",
        "focus": 1,
        "logic": "NEG",
        "alternative": "1"
    }
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalized_focus_choices
from qualtrics_export_common import (
    build_audio_map_rows,
    load_items,
    sanitize_qualtrics_text,
    write_audio_map,
)


ADVANCED_FORMAT_HEADER = "[[AdvancedFormat]]"
BLOCK_HEADER = "[[Block:Focus Survey]]"
QUESTION_HEADER = "[[Question:MC:SingleAnswer:Vertical]]"
CHOICES_HEADER = "[[Choices]]"


def build_qualtrics_question(item: dict[str, Any], idx: int) -> str:
    """Build one Qualtrics Advanced TXT question block from a JSON item."""
    required_fields = ("S1", "A", "focus", "logic", "alternative")
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise KeyError(f"Item {idx} is missing required fields: {', '.join(missing_fields)}")

    s1 = sanitize_qualtrics_text(item["S1"])
    focus = int(item["focus"])
    normalized_s1 = sanitize_qualtrics_text(item.get("S1_normalized", normalize_s1(s1)))
    emphasized_word, alternative_word = normalized_focus_choices(s1, focus)

    lines = [
        f"[[ED:item_id:{idx}]]",
        f"[[ED:label:{sanitize_qualtrics_text(item['A'])}]]",
        f"[[ED:focus:{focus}]]",
        f"[[ED:logic:{sanitize_qualtrics_text(item['logic'])}]]",
        f"[[ED:alternative:{sanitize_qualtrics_text(item['alternative'])}]]",
        QUESTION_HEADER,
        "In Sentence 1, which word was said with stronger emphasis?<br><br>",
        f"Sentence 1: {normalized_s1}<br>",
        CHOICES_HEADER,
        sanitize_qualtrics_text(emphasized_word),
        sanitize_qualtrics_text(alternative_word),
    ]
    return "\n".join(lines)


def build_survey(items: list[dict[str, Any]]) -> str:
    """Build the complete Qualtrics Advanced TXT survey."""
    question_blocks = [build_qualtrics_question(item, idx) for idx, item in enumerate(items, start=1)]
    return "\n\n".join([ADVANCED_FORMAT_HEADER, BLOCK_HEADER, *question_blocks]) + "\n"


def print_preview(items: list[dict[str, Any]], preview_count: int = 3) -> None:
    """Print a small console preview for the first few questions."""
    print(f"Previewing first {min(preview_count, len(items))} question(s):")
    for idx, item in enumerate(items[:preview_count], start=1):
        s1 = sanitize_qualtrics_text(item["S1"])
        normalized_s1 = sanitize_qualtrics_text(item.get("S1_normalized", normalize_s1(s1)))
        emphasized_word, alternative_word = normalized_focus_choices(s1, int(item["focus"]))
        print()
        print(f"Question {idx}")
        print(f"  Sentence 1: {normalized_s1}")
        print(f"  Choices: {emphasized_word} | {alternative_word}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a focus-survey JSON array into a Qualtrics Advanced TXT import file."
    )
    parser.add_argument("input_json", type=Path, help="Path to the input JSON file")
    parser.add_argument("output_txt", type=Path, help="Path to the output Qualtrics TXT file")
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/clips/set1"),
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
