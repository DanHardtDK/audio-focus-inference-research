#!/usr/bin/env python3
"""Build the production Qualtrics TXT imports.

The production workflow has two outputs:

1. Survey 1: small focus and inference imports from curated item JSON files.
2. 24-item speaker surveys: one focus and one inference import per speaker,
   using the shared ns1-ns3 stimulus set.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from focus_text_utils import normalize_s1, normalize_s2
from qualtrics_export_common import build_audio_map_rows, load_items, write_audio_map
from qualtrics_focus_export import build_survey as build_focus_survey
from qualtrics_inference_export import build_survey as build_inference_survey


SurveyKind = Literal["focus", "inference"]

DEFAULT_SPEAKERS = ("speaker0", "speaker1", "speaker2")
DEFAULT_SOURCE_STEMS = ("ns1", "ns2", "ns3")
SURVEY1_SEED = 1
SPEAKER24_SEED = 42


@dataclass(frozen=True)
class Survey1Spec:
    kind: SurveyKind
    input_json: Path
    output_txt: Path


def shuffled(items: list[dict[str, Any]], seed: int | None, no_shuffle: bool) -> list[dict[str, Any]]:
    """Return copied items in the production export order."""
    export_items = [dict(item) for item in items]
    if not no_shuffle:
        random.Random(seed).shuffle(export_items)
    return export_items


def build_survey_text(
    kind: SurveyKind,
    items: list[dict[str, Any]],
    input_json: Path,
    clips_dir: Path,
) -> str:
    if kind == "focus":
        return build_focus_survey(items, input_json, clips_dir)
    return build_inference_survey(items, input_json, clips_dir)


def build_survey1(
    specs: list[Survey1Spec],
    clips_dir: Path,
    seed: int | None,
    no_shuffle: bool,
) -> None:
    """Build the curated small Survey 1 focus and inference imports."""
    for spec in specs:
        items = [
            {**item, "speaker": item.get("speaker", "speaker0")}
            for item in shuffled(load_items(spec.input_json), seed, no_shuffle)
        ]
        survey_text = build_survey_text(spec.kind, items, spec.input_json, clips_dir)
        audio_rows = build_audio_map_rows(items, spec.input_json, clips_dir)

        spec.output_txt.parent.mkdir(parents=True, exist_ok=True)
        spec.output_txt.write_text(survey_text, encoding="utf-8")
        write_audio_map(audio_rows, spec.output_txt.with_suffix(".audio_map.csv"))

        matched_audio = sum(row["audio_exists"] == "yes" for row in audio_rows)
        print(
            f"Wrote {len(items)} {spec.kind} Survey 1 question(s) to {spec.output_txt} "
            f"with {matched_audio}/{len(audio_rows)} audio clips matched"
        )


def load_ns24_items(stimuli_dir: Path, source_stems: tuple[str, ...]) -> list[dict[str, Any]]:
    """Load the ns1-ns3 24-item stimulus set with normalized display text."""
    items: list[dict[str, Any]] = []
    for source_stem in source_stems:
        source_json = stimuli_dir / f"{source_stem}.json"
        source_items = load_items(source_json)
        for item_index, source_item in enumerate(source_items):
            item = dict(source_item)
            item["source_json"] = source_json.name
            item["source_stem"] = source_stem
            item["source_item_index"] = item_index
            item["S1_normalized"] = normalize_s1(str(item["S1"]))
            item["S2_normalized"] = normalize_s2(str(item["S2"]))
            items.append(item)
    return items


def speaker_item(item: dict[str, Any], speaker: str) -> dict[str, Any]:
    """Attach speaker-specific metadata used for stable Qualtrics IDs."""
    source_stem = str(item["source_stem"])
    item_index = int(item["source_item_index"])
    speaker_specific_item = dict(item)
    speaker_specific_item["speaker"] = speaker
    speaker_specific_item["audio_file"] = f"{speaker}_{source_stem}_item{item_index}.wav"
    return speaker_specific_item


def build_speaker24(
    kind: SurveyKind,
    stimuli_dir: Path,
    output_dir: Path,
    speakers: tuple[str, ...],
    source_stems: tuple[str, ...],
    seed: int | None,
    no_shuffle: bool,
) -> None:
    """Build one 24-item import per speaker for focus or inference."""
    items = shuffled(load_ns24_items(stimuli_dir, source_stems), seed, no_shuffle)

    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder_input_json = stimuli_dir / "ns1-ns3.json"
    for speaker in speakers:
        export_items = [speaker_item(item, speaker) for item in items]
        output_txt = output_dir / f"{speaker}_ns1-ns3_{kind}_survey.txt"
        survey_text = build_survey_text(kind, export_items, placeholder_input_json, Path())
        output_txt.write_text(survey_text, encoding="utf-8")
        print(f"Wrote {len(export_items)} {kind} question(s) to {output_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the production Qualtrics import files."
    )
    parser.add_argument(
        "--only",
        choices=("all", "survey1", "focus24", "inference24"),
        default="all",
        help="Limit the build to one output group.",
    )
    parser.add_argument(
        "--survey1-dir",
        type=Path,
        default=Path("code/qualtrics_export/input/survey1"),
        help="Directory containing set1_focus_items.json and set2_inference_items.json.",
    )
    parser.add_argument(
        "--survey1-output-dir",
        type=Path,
        default=Path("code/qualtrics_export/output/survey1"),
        help="Directory for Survey 1 TXT imports and audio maps.",
    )
    parser.add_argument(
        "--survey1-clips-dir",
        type=Path,
        default=Path("data/speakers/speaker0/clips"),
        help="Directory containing the Survey 1 audio clips.",
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=Path("data/stimuli"),
        help="Directory containing ns1.json, ns2.json, and ns3.json.",
    )
    parser.add_argument(
        "--focus24-output-dir",
        type=Path,
        default=Path("code/qualtrics_export/output/focus_24_by_speaker"),
        help="Directory for 24-item focus imports.",
    )
    parser.add_argument(
        "--inference24-output-dir",
        type=Path,
        default=Path("code/qualtrics_export/output/inference_24_by_speaker"),
        help="Directory for 24-item inference imports.",
    )
    parser.add_argument(
        "--speaker",
        action="append",
        dest="speakers",
        help="Speaker ID to export, e.g. speaker0. May be repeated.",
    )
    parser.add_argument(
        "--no-shuffle-questions",
        action="store_true",
        help="Keep source item order instead of the production shuffled order.",
    )
    parser.add_argument(
        "--survey1-seed",
        type=int,
        default=SURVEY1_SEED,
        help="Random seed for Survey 1 question order.",
    )
    parser.add_argument(
        "--speaker24-seed",
        type=int,
        default=SPEAKER24_SEED,
        help="Random seed for 24-item speaker survey question order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speakers = tuple(args.speakers or DEFAULT_SPEAKERS)
    source_stems = DEFAULT_SOURCE_STEMS

    if args.only in ("all", "survey1"):
        survey1_specs = [
            Survey1Spec(
                kind="focus",
                input_json=args.survey1_dir / "set1_focus_items.json",
                output_txt=args.survey1_output_dir / "set1_focus_survey.txt",
            ),
            Survey1Spec(
                kind="inference",
                input_json=args.survey1_dir / "set2_inference_items.json",
                output_txt=args.survey1_output_dir / "set2_inference_survey.txt",
            ),
        ]
        build_survey1(
            survey1_specs,
            args.survey1_clips_dir,
            args.survey1_seed,
            args.no_shuffle_questions,
        )

    if args.only in ("all", "focus24"):
        build_speaker24(
            "focus",
            args.stimuli_dir,
            args.focus24_output_dir,
            speakers,
            source_stems,
            args.speaker24_seed,
            args.no_shuffle_questions,
        )

    if args.only in ("all", "inference24"):
        build_speaker24(
            "inference",
            args.stimuli_dir,
            args.inference24_output_dir,
            speakers,
            source_stems,
            args.speaker24_seed,
            args.no_shuffle_questions,
        )


if __name__ == "__main__":
    main()
