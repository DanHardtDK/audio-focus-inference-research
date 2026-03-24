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
    - Prints a preview of the first 3 generated questions.

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
        f"Sentence 1: {s1}",
        f"Sentence 2: {s2}",
        "",
        "In Sentence 1, which word was said with stronger emphasis?",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.input_json)
    survey_text = build_survey(items)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(survey_text, encoding="utf-8")

    print_preview(items, preview_count=3)
    print()
    print(f"Wrote {len(items)} question(s) to {args.output_txt}")


if __name__ == "__main__":
    main()
