#!/usr/bin/env python3
"""Shared helpers for Qualtrics survey export scripts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def sanitize_qualtrics_text(value: Any) -> str:
    """Remove characters and sequences that can interfere with TXT import tags."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("[[", "[").replace("]]", "]")
    return text.strip()


def load_items(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the input JSON array."""
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be an array of items, got {type(data).__name__}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every entry in the input JSON array must be an object")

    return data


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


def infer_question_id(
    item: dict[str, Any],
    question_idx: int,
    input_json: Path,
    clips_dir: Path,
) -> str:
    """Infer a stable question identifier from the matching audio clip stem."""
    audio_path = infer_audio_path(item, question_idx, input_json, clips_dir)
    if audio_path:
        return sanitize_qualtrics_text(audio_path.stem)
    return f"{sanitize_qualtrics_text(input_json.stem)}_item{question_idx - 1}"


def build_audio_map_rows(
    items: list[dict[str, Any]],
    input_json: Path,
    clips_dir: Path,
) -> list[dict[str, str]]:
    """Build rows describing which audio file belongs to each exported question."""
    rows: list[dict[str, str]] = []
    for question_idx, item in enumerate(items, start=1):
        audio_path = infer_audio_path(item, question_idx, input_json, clips_dir)
        question_id = infer_question_id(item, question_idx, input_json, clips_dir)
        audio_filename = audio_path.name if audio_path else ""
        audio_exists = "yes" if audio_path and audio_path.exists() else "no"
        rows.append(
            {
                "question_number": str(question_idx),
                "question_id": question_id,
                "speaker": sanitize_qualtrics_text(item.get("speaker", "")),
                "source_json": sanitize_qualtrics_text(item.get("source_json", "")),
                "source_item_index": sanitize_qualtrics_text(item.get("source_item_index", "")),
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
    """Write the audio mapping CSV used for manual upload and attachment in Qualtrics."""
    fieldnames = [
        "question_number",
        "question_id",
        "speaker",
        "source_json",
        "source_item_index",
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
