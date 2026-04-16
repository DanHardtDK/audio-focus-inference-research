#!/usr/bin/env python3
"""Build one standard 24-item inference Qualtrics TXT import per speaker."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalize_s2
from qualtrics_export_common import sanitize_qualtrics_text


ADVANCED_FORMAT_HEADER = "[[AdvancedFormat]]"
BLOCK_HEADER = "[[Block:Inference Survey]]"
QUESTION_HEADER = "[[Question:MC:SingleAnswer:Vertical]]"
CHOICES_HEADER = "[[Choices]]"
DEFAULT_SOURCE_STEMS = ("ns1", "ns2", "ns3")
DEFAULT_SPEAKERS = ("speaker0", "speaker1", "speaker2")
INFERENCE_CHOICES = [
    "Sentence 2 must be true.",
    "Sentence 2 might be true.",
    "Sentence 2 must be false.",
]


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def load_inference24_items(stimuli_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source_stem in DEFAULT_SOURCE_STEMS:
        source_json_path = stimuli_dir / f"{source_stem}.json"
        source_items = load_json_array(source_json_path)
        for item_index, source_item in enumerate(source_items):
            item = dict(source_item)
            item["source_stem"] = source_stem
            item["source_item_index"] = item_index
            item["S1_normalized"] = normalize_s1(str(item["S1"]))
            item["S2_normalized"] = normalize_s2(str(item["S2"]))
            items.append(item)
    return items


def question_id(speaker: str, source_stem: str, item_index: int) -> str:
    return f"{speaker}_{source_stem}_item{item_index}"


def build_question(item: dict[str, Any], speaker: str) -> str:
    required_fields = ("S1", "S2", "source_stem", "source_item_index")
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise KeyError(f"Item is missing required fields: {', '.join(missing_fields)}")

    source_stem = sanitize_qualtrics_text(item["source_stem"])
    item_index = int(item["source_item_index"])
    s1 = sanitize_qualtrics_text(item.get("S1_normalized", normalize_s1(str(item["S1"]))))
    s2 = sanitize_qualtrics_text(item.get("S2_normalized", normalize_s2(str(item["S2"]))))

    lines = [
        QUESTION_HEADER,
        f"[[ID:{question_id(speaker, source_stem, item_index)}]]",
        "Given Sentence 1, what can we say about Sentence 2?<br><br>",
        f"Sentence 1: {s1}<br>",
        f"Sentence 2: {s2}<br>",
        CHOICES_HEADER,
        INFERENCE_CHOICES[0],
        INFERENCE_CHOICES[1],
        INFERENCE_CHOICES[2],
    ]
    return "\n".join(lines)


def build_survey(items: list[dict[str, Any]], speaker: str) -> str:
    question_blocks = [build_question(item, speaker) for item in items]
    return "\n\n".join([ADVANCED_FORMAT_HEADER, BLOCK_HEADER, *question_blocks]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build plain 24-item inference Qualtrics TXT imports with speaker-aligned question IDs."
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
        default=Path("code/qualtrics_export/output/inference_24_by_speaker"),
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
    items = load_inference24_items(args.stimuli_dir)

    export_items = [dict(item) for item in items]
    if not args.no_shuffle_questions:
        random.Random(args.seed).shuffle(export_items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for speaker in speakers:
        output_txt = args.output_dir / f"{speaker}_ns1-ns3_inference_survey.txt"
        output_txt.write_text(build_survey(export_items, speaker), encoding="utf-8")
        print(f"Wrote {len(export_items)} question(s) to {output_txt}")


if __name__ == "__main__":
    main()
