#!/usr/bin/env python3
"""Build the 24-item multi-speaker focus survey for Qualtrics.

Default selection:
  - speaker0: f1 items 0-7
  - speaker1: ns1 items 0-7
  - speaker2: ns2 items 0-7
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalize_s2
from qualtrics_export_common import build_audio_map_rows, write_audio_map
from qualtrics_focus_export import build_survey, print_preview


DEFAULT_SELECTIONS = ("speaker0:f1:0-7", "speaker1:ns1:0-7", "speaker2:ns2:0-7")


def parse_item_range(raw_range: str) -> list[int]:
    """Parse item specs like 0-7 or 0,2,4."""
    item_indices: list[int] = []
    for part in raw_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending item range: {part}")
            item_indices.extend(range(start, end + 1))
        else:
            item_indices.append(int(part))
    if not item_indices:
        raise ValueError(f"Empty item range: {raw_range}")
    return item_indices


def parse_selection(raw_selection: str) -> tuple[str, str, list[int]]:
    """Parse selection specs like speaker1:ns1:0-7."""
    parts = raw_selection.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid selection {raw_selection!r}; expected speaker:source:item_range"
        )
    speaker, source_stem, raw_range = (part.strip() for part in parts)
    if not speaker or not source_stem:
        raise ValueError(f"Invalid selection {raw_selection!r}; speaker/source cannot be empty")
    return speaker, source_stem, parse_item_range(raw_range)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def copied_audio_name(speaker: str, source_stem: str, item_index: int) -> str:
    return f"{speaker}_{source_stem}_item{item_index}.wav"


def build_focus_items(
    selections: list[tuple[str, str, list[int]]],
    stimuli_dir: Path,
    speakers_dir: Path,
    audio_dir: Path,
) -> list[dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    audio_names: set[str] = set()

    for speaker, source_stem, item_indices in selections:
        source_json_path = stimuli_dir / f"{source_stem}.json"
        if not source_json_path.exists():
            raise FileNotFoundError(f"Missing source JSON: {source_json_path}")

        source_items = load_json_array(source_json_path)
        for item_index in item_indices:
            try:
                item = dict(source_items[item_index])
            except IndexError as exc:
                raise IndexError(
                    f"{source_json_path} has {len(source_items)} item(s), "
                    f"but selection requested item {item_index}"
                ) from exc

            source_audio_path = (
                speakers_dir / speaker / "clips" / f"{source_stem}_item{item_index}.wav"
            )
            if not source_audio_path.exists():
                raise FileNotFoundError(f"Missing audio clip: {source_audio_path}")

            audio_file = copied_audio_name(speaker, source_stem, item_index)
            if audio_file in audio_names:
                raise ValueError(f"Duplicate output audio filename: {audio_file}")
            audio_names.add(audio_file)
            shutil.copy2(source_audio_path, audio_dir / audio_file)

            item["speaker"] = speaker
            item["source_json"] = source_json_path.name
            item["source_item_index"] = item_index
            item["audio_file"] = audio_file
            if "S1" in item:
                item["S1_normalized"] = normalize_s1(str(item["S1"]))
            if "S2" in item:
                item["S2_normalized"] = normalize_s2(str(item["S2"]))
            items.append(item)

    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the default 24-item multi-speaker focus survey for Qualtrics."
    )
    parser.add_argument(
        "--selection",
        action="append",
        help=(
            "Selection in speaker:source:item_range form, e.g. speaker1:ns1:0-7. "
            "May be repeated. Defaults to the current 24-item focus survey."
        ),
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=Path("data/stimuli"),
        help="Directory containing f*.json and ns*.json stimulus files",
    )
    parser.add_argument(
        "--speakers-dir",
        type=Path,
        default=Path("data/speakers"),
        help="Directory containing speaker*/clips folders",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/clips/focus_24"),
        help="Directory where selected audio clips are copied for Qualtrics upload",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("code/qualtrics_export/input/focus_24/focus_24_items.json"),
        help="Path to write the generated 24-item JSON subset",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("code/qualtrics_export/output/focus_24/focus_24_focus_survey.txt"),
        help="Path to write the Qualtrics Advanced TXT import file",
    )
    parser.add_argument(
        "--audio-map",
        type=Path,
        help="Optional CSV path for question-to-audio mapping; defaults next to OUTPUT_TXT",
    )
    parser.add_argument(
        "--no-shuffle-questions",
        action="store_true",
        help="Disable the default build-time question shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible build-time question shuffling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_selections = args.selection or list(DEFAULT_SELECTIONS)
    selections = [parse_selection(raw_selection) for raw_selection in raw_selections]

    items = build_focus_items(selections, args.stimuli_dir, args.speakers_dir, args.audio_dir)
    args.input_json.parent.mkdir(parents=True, exist_ok=True)
    args.input_json.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    export_items = [dict(item) for item in items]
    if not args.no_shuffle_questions:
        random.Random(args.seed).shuffle(export_items)

    survey_text = build_survey(export_items, args.input_json, args.audio_dir)
    audio_map_path = args.audio_map or args.output_txt.with_suffix(".audio_map.csv")
    audio_rows = build_audio_map_rows(export_items, args.input_json, args.audio_dir)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(survey_text, encoding="utf-8")
    write_audio_map(audio_rows, audio_map_path)

    print_preview(export_items, args.input_json, args.audio_dir, preview_count=3)
    print()
    print(f"Wrote {len(items)} item(s) to {args.input_json}")
    print(f"Copied {len(items)} audio clip(s) to {args.audio_dir}")
    print(f"Wrote {len(export_items)} question(s) to {args.output_txt}")
    print(f"Wrote audio mapping CSV to {audio_map_path}")

    matched_audio = sum(row["audio_exists"] == "yes" for row in audio_rows)
    print(f"Matched audio clips: {matched_audio}/{len(audio_rows)}")


if __name__ == "__main__":
    main()
