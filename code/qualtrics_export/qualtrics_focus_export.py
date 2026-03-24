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
import csv
import json
import re
from pathlib import Path
from typing import Any


ADVANCED_FORMAT_HEADER = "[[AdvancedFormat]]"
BLOCK_HEADER = "[[Block:Focus Survey]]"
QUESTION_HEADER = "[[Question:MC:SingleAnswer:Vertical]]"
CHOICES_HEADER = "[[Choices]]"

S1_PATTERN = re.compile(r"^\s*(?P<subject>.+?)\s+only\s+gave\s+(?P<tail>.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def sanitize_qualtrics_text(value: Any) -> str:
    """Remove characters and sequences that can interfere with TXT import tags."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("[[", "[").replace("]]", "]")
    return text.strip()


def extract_objects_from_s1(s1: str) -> tuple[str, str]:
    """Extract the two objects from an S1 sentence of the form '<subject> only gave X Y.'."""
    sanitized_s1 = sanitize_qualtrics_text(s1)
    match = S1_PATTERN.match(sanitized_s1.rstrip(".!?"))
    if not match:
        raise ValueError(f"Could not parse S1 structure: {s1!r}")

    tail = match.group("tail")
    objects = TOKEN_PATTERN.findall(tail)
    if len(objects) != 2:
        raise ValueError(
            f"Expected exactly two objects after 'only gave' in S1, found {len(objects)}: {s1!r}"
        )

    return objects[0], objects[1]


def extract_emphasized_word(s1: str, focus: int) -> str:
    """Find the emphasized word, preferring the fully uppercase token when present."""
    object1, object2 = extract_objects_from_s1(s1)
    uppercase_objects = [
        token
        for token in (object1, object2)
        if any(char.isalpha() for char in token) and token == token.upper()
    ]

    if len(uppercase_objects) == 1:
        return uppercase_objects[0]
    if len(uppercase_objects) > 1:
        raise ValueError(f"Found multiple uppercase focus candidates in S1: {s1!r}")
    if focus == 1:
        return object1
    if focus == 2:
        return object2
    raise ValueError(f"Unsupported focus value {focus!r} for S1: {s1!r}")


def build_qualtrics_question(item: dict[str, Any], idx: int) -> str:
    """Build one Qualtrics Advanced TXT question block from a JSON item."""
    required_fields = ("S1", "S2", "A", "focus", "logic", "alternative")
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise KeyError(f"Item {idx} is missing required fields: {', '.join(missing_fields)}")

    s1 = sanitize_qualtrics_text(item["S1"])
    s2 = sanitize_qualtrics_text(item["S2"])
    focus = int(item["focus"])

    object1, object2 = extract_objects_from_s1(s1)
    emphasized_word = extract_emphasized_word(s1, focus)

    if emphasized_word == object1:
        alternative_word = object2
    elif emphasized_word == object2:
        alternative_word = object1
    else:
        raise ValueError(
            f"Emphasized word {emphasized_word!r} does not match parsed objects in S1: {s1!r}"
        )

    lines = [
        f"[[ED:item_id:{idx}]]",
        f"[[ED:label:{sanitize_qualtrics_text(item['A'])}]]",
        f"[[ED:focus:{focus}]]",
        f"[[ED:logic:{sanitize_qualtrics_text(item['logic'])}]]",
        f"[[ED:alternative:{sanitize_qualtrics_text(item['alternative'])}]]",
        QUESTION_HEADER,
        "In Sentence 1, which word was said with stronger emphasis?<br><br>",
        f"Sentence 1: {s1}<br>",
        f"Sentence 2: {s2}<br>",
        CHOICES_HEADER,
        sanitize_qualtrics_text(emphasized_word),
        sanitize_qualtrics_text(alternative_word),
    ]
    return "\n".join(lines)


def load_items(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the input JSON array."""
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be an array of items, got {type(data).__name__}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every entry in the input JSON array must be an object")

    return data


def build_survey(items: list[dict[str, Any]]) -> str:
    """Build the complete Qualtrics Advanced TXT survey."""
    question_blocks = [build_qualtrics_question(item, idx) for idx, item in enumerate(items, start=1)]
    return "\n\n".join([ADVANCED_FORMAT_HEADER, BLOCK_HEADER, *question_blocks]) + "\n"


def infer_audio_path(
    item: dict[str, Any],
    question_idx: int,
    input_json: Path,
    clips_dir: Path,
) -> Path | None:
    """Infer the matching audio clip path for a survey item when possible."""
    if "audio_file" in item and item["audio_file"]:
        return clips_dir / sanitize_qualtrics_text(item["audio_file"])

    if "source_json" in item and "source_item_index" in item:
        source_stem = Path(str(item["source_json"])).stem
        source_item_index = int(item["source_item_index"])
        return clips_dir / f"{source_stem}_item{source_item_index}.wav"

    if re.fullmatch(r"f\d+", input_json.stem):
        return clips_dir / f"{input_json.stem}_item{question_idx - 1}.wav"

    return None


def build_audio_map_rows(
    items: list[dict[str, Any]],
    input_json: Path,
    clips_dir: Path,
) -> list[dict[str, str]]:
    """Build rows describing which audio file belongs to each exported question."""
    rows: list[dict[str, str]] = []
    for question_idx, item in enumerate(items, start=1):
        audio_path = infer_audio_path(item, question_idx, input_json, clips_dir)
        audio_filename = audio_path.name if audio_path else ""
        audio_exists = "yes" if audio_path and audio_path.exists() else "no"
        rows.append(
            {
                "question_number": str(question_idx),
                "item_id": str(question_idx),
                "label": sanitize_qualtrics_text(item.get("A", "")),
                "focus": sanitize_qualtrics_text(item.get("focus", "")),
                "logic": sanitize_qualtrics_text(item.get("logic", "")),
                "alternative": sanitize_qualtrics_text(item.get("alternative", "")),
                "s1": sanitize_qualtrics_text(item.get("S1", "")),
                "s2": sanitize_qualtrics_text(item.get("S2", "")),
                "audio_filename": audio_filename,
                "audio_path": str(audio_path) if audio_path else "",
                "audio_exists": audio_exists,
            }
        )
    return rows


def write_audio_map(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the audio mapping CSV used for manual upload/attachment in Qualtrics."""
    fieldnames = [
        "question_number",
        "item_id",
        "label",
        "focus",
        "logic",
        "alternative",
        "s1",
        "s2",
        "audio_filename",
        "audio_path",
        "audio_exists",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_preview(items: list[dict[str, Any]], preview_count: int = 3) -> None:
    """Print a small console preview for the first few questions."""
    print(f"Previewing first {min(preview_count, len(items))} question(s):")
    for idx, item in enumerate(items[:preview_count], start=1):
        s1 = sanitize_qualtrics_text(item["S1"])
        emphasized_word = extract_emphasized_word(s1, int(item["focus"]))
        object1, object2 = extract_objects_from_s1(s1)
        alternative_word = object2 if emphasized_word == object1 else object1
        print()
        print(f"Question {idx}")
        print(f"  Sentence 1: {s1}")
        print(f"  Sentence 2: {sanitize_qualtrics_text(item['S2'])}")
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
