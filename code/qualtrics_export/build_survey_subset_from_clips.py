#!/usr/bin/env python3
"""Build a survey JSON subset from clip filenames like f1_item0.wav."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from focus_text_utils import normalize_s1, normalize_s2


CLIP_PATTERN = re.compile(r"^(f\d+)_item(\d+)\.wav$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSON survey subset that matches a directory of audio clips."
    )
    parser.add_argument("clips_dir", type=Path, help="Directory containing clip files like f1_item0.wav")
    parser.add_argument("output_json", type=Path, help="Path to write the subset JSON array")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/input"),
        help="Directory containing the source fX.json files",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def build_subset(clips_dir: Path, input_dir: Path) -> list[dict[str, Any]]:
    subset: list[dict[str, Any]] = []
    clip_paths = sorted(path for path in clips_dir.iterdir() if path.is_file())

    for clip_path in clip_paths:
        match = CLIP_PATTERN.match(clip_path.name)
        if not match:
            continue

        source_stem = match.group(1).lower()
        source_item_index = int(match.group(2))
        source_json_path = input_dir / f"{source_stem}.json"
        if not source_json_path.exists():
            raise FileNotFoundError(
                f"Missing source JSON for clip {clip_path.name}: expected {source_json_path}"
            )

        items = load_json(source_json_path)
        try:
            item = dict(items[source_item_index])
        except IndexError as exc:
            raise IndexError(
                f"Clip {clip_path.name} refers to item {source_item_index}, "
                f"but {source_json_path} has only {len(items)} item(s)"
            ) from exc

        item["source_json"] = source_json_path.name
        item["source_item_index"] = source_item_index
        item["audio_file"] = clip_path.name
        if "S1" in item:
            item["S1_normalized"] = normalize_s1(str(item["S1"]))
        if "S2" in item:
            item["S2_normalized"] = normalize_s2(str(item["S2"]))
        subset.append(item)

    if not subset:
        raise ValueError(f"No matching clip filenames were found in {clips_dir}")

    return subset


def main() -> None:
    args = parse_args()
    subset = build_subset(args.clips_dir, args.input_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(subset)} item(s) to {args.output_json}")


if __name__ == "__main__":
    main()
