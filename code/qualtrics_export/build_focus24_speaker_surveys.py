#!/usr/bin/env python3
"""Build one standard Focus-24 Qualtrics TXT import per speaker."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalized_sentence_order_choices
from qualtrics_export_common import sanitize_qualtrics_text


ADVANCED_FORMAT_HEADER = "[[AdvancedFormat]]"
BLOCK_HEADER = "[[Block:Focus Survey]]"
QUESTION_HEADER = "[[Question:MC:SingleAnswer:Vertical]]"
CHOICES_HEADER = "[[Choices]]"
RANDOMIZE_HEADER = "[[Randomize]]"
DEFAULT_SOURCE_STEMS = ("ns1", "ns2", "ns3")
DEFAULT_SPEAKERS = ("speaker0", "speaker1", "speaker2")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def load_focus24_items(stimuli_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source_stem in DEFAULT_SOURCE_STEMS:
        source_json_path = stimuli_dir / f"{source_stem}.json"
        source_items = load_json_array(source_json_path)
        for item_index, source_item in enumerate(source_items):
            item = dict(source_item)
            item["source_stem"] = source_stem
            item["source_item_index"] = item_index
            item["S1_normalized"] = normalize_s1(str(item["S1"]))
            items.append(item)
    return items


def question_id(speaker: str, source_stem: str, item_index: int) -> str:
    return f"{speaker}_{source_stem}_item{item_index}"


def build_question(item: dict[str, Any], speaker: str) -> str:
    required_fields = ("S1", "source_stem", "source_item_index")
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise KeyError(f"Item is missing required fields: {', '.join(missing_fields)}")

    s1 = sanitize_qualtrics_text(item["S1"])
    source_stem = sanitize_qualtrics_text(item["source_stem"])
    item_index = int(item["source_item_index"])
    normalized_s1 = sanitize_qualtrics_text(item.get("S1_normalized", normalize_s1(s1)))
    first_choice, second_choice = normalized_sentence_order_choices(s1)

    lines = [
        QUESTION_HEADER,
        f"[[ID:{question_id(speaker, source_stem, item_index)}]]",
        "In Sentence 1, which word was said with stronger emphasis?<br><br>",
        f"Sentence 1: {normalized_s1}<br>",
        RANDOMIZE_HEADER,
        CHOICES_HEADER,
        sanitize_qualtrics_text(first_choice),
        sanitize_qualtrics_text(second_choice),
    ]
    return "\n".join(lines)


def build_survey(items: list[dict[str, Any]], speaker: str) -> str:
    question_blocks = [build_question(item, speaker) for item in items]
    return "\n\n".join([ADVANCED_FORMAT_HEADER, BLOCK_HEADER, *question_blocks]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build plain Focus-24 Qualtrics TXT imports with speaker-aligned question IDs."
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=Path("data/stimuli"),
        help="Directory containing ns1.json, ns2.json, and ns3.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("code/qualtrics_export/output/focus_24_by_speaker"),
        help="Directory for generated Qualtrics TXT imports",
    )
    parser.add_argument(
        "--speaker",
        action="append",
        dest="speakers",
        help="Speaker ID prefix to export, e.g. speaker0. May be repeated.",
    )
    parser.add_argument(
        "--no-shuffle-questions",
        action="store_true",
        help="Keep questions in ns1, ns2, ns3 item order instead of shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible build-time question shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speakers = tuple(args.speakers or DEFAULT_SPEAKERS)
    items = load_focus24_items(args.stimuli_dir)

    export_items = [dict(item) for item in items]
    if not args.no_shuffle_questions:
        random.Random(args.seed).shuffle(export_items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for speaker in speakers:
        output_txt = args.output_dir / f"{speaker}_ns1-ns3_focus_survey.txt"
        output_txt.write_text(build_survey(export_items, speaker), encoding="utf-8")
        print(f"Wrote {len(export_items)} question(s) to {output_txt}")


if __name__ == "__main__":
    main()
