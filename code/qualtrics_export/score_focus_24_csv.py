#!/usr/bin/env python3
"""Score raw Qualtrics CSV exports for the 24-item focus speaker survey.

This script is specific to the production focus survey built from
`data/stimuli/ns1.json`, `ns2.json`, and `ns3.json`.

It expects a Qualtrics response CSV whose question columns are named like
`speaker0_ns2_item7`. The speaker prefix is ignored for scoring because the
solution key is shared across speakers.

Qualtrics recode values are interpreted as:

- source JSON `focus=1` -> CSV response `1`
- source JSON `focus=2` -> CSV response `2`
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from focus_text_utils import normalize_s1, normalized_sentence_order_choices
from qualtrics_export_common import load_items


QUESTION_ID_PATTERN = re.compile(r"(speaker\d+)_(ns[123]_item\d+)")
QUESTION_ID_IN_IMPORT_PATTERN = re.compile(r"\[\[ID:speaker\d+_(ns[123]_item\d+)\]\]")
RESPONSE_ID_PATTERN = re.compile(r"^R_[A-Za-z0-9]+$")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIMULI_DIR = REPO_ROOT / "data/stimuli"
DEFAULT_IMPORT_TXT = REPO_ROOT / (
    "code/qualtrics_export/output/focus_24_by_speaker/"
    "speaker0_ns1-ns3_focus_survey.txt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "code/qualtrics_export/output/focus_scoring"
EXPECTED_QUESTION_COUNT = 24


@dataclass(frozen=True)
class QuestionColumn:
    """Metadata extracted from a speaker-specific Qualtrics response column."""

    column_name: str
    speaker: str
    question_id: str


@dataclass(frozen=True)
class FocusSolution:
    """Scoring metadata for one shared ns-item focus question."""

    question_id: str
    solution_numeric: str
    solution_word: str
    alternative_word: str
    choice_1_word: str
    choice_2_word: str
    sentence_1: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a raw Qualtrics CSV export for the 24-item focus survey."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the raw Qualtrics CSV export.",
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        default=DEFAULT_STIMULI_DIR,
        help="Directory containing ns1.json, ns2.json, and ns3.json.",
    )
    parser.add_argument(
        "--question-order-txt",
        type=Path,
        default=DEFAULT_IMPORT_TXT,
        help="Focus survey import TXT used to recover the production question order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the scored CSV outputs will be written.",
    )
    parser.add_argument(
        "--include-empty-responses",
        action="store_true",
        help="Keep zero-answer rows in the summary output.",
    )
    return parser.parse_args()


def load_solution_key(stimuli_dir: Path) -> dict[str, FocusSolution]:
    """Load the shared focus key from ns1-ns3 stimulus JSON files."""
    solution_key: dict[str, FocusSolution] = {}
    for source_stem in ("ns1", "ns2", "ns3"):
        items = load_items(stimuli_dir / f"{source_stem}.json")
        for item_index, item in enumerate(items):
            question_id = f"{source_stem}_item{item_index}"
            focus_value = str(item.get("focus", "")).strip()
            if focus_value not in {"1", "2"}:
                raise ValueError(
                    f"Unexpected focus value {focus_value!r} in {source_stem}.json item {item_index}"
                )

            s1 = str(item["S1"])
            choice_1_word, choice_2_word = normalized_sentence_order_choices(s1)
            solution_word = choice_1_word if focus_value == "1" else choice_2_word
            alternative_word = choice_2_word if focus_value == "1" else choice_1_word
            solution_key[question_id] = FocusSolution(
                question_id=question_id,
                solution_numeric=focus_value,
                solution_word=solution_word,
                alternative_word=alternative_word,
                choice_1_word=choice_1_word,
                choice_2_word=choice_2_word,
                sentence_1=normalize_s1(s1),
            )

    if len(solution_key) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_COUNT} ns1-ns3 items, found {len(solution_key)}"
        )

    return solution_key


def load_question_order(import_txt: Path) -> dict[str, int]:
    """Extract the production question order from a speaker focus import TXT."""
    text = import_txt.read_text(encoding="utf-8")
    question_ids = QUESTION_ID_IN_IMPORT_PATTERN.findall(text)
    if not question_ids:
        raise ValueError(f"No focus question IDs found in {import_txt}")

    unique_question_ids: list[str] = []
    seen: set[str] = set()
    for question_id in question_ids:
        if question_id in seen:
            continue
        seen.add(question_id)
        unique_question_ids.append(question_id)

    if len(unique_question_ids) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_COUNT} unique question IDs in {import_txt}, "
            f"found {len(unique_question_ids)}"
        )

    return {
        question_id: question_index
        for question_index, question_id in enumerate(unique_question_ids, start=1)
    }


def infer_question_columns(fieldnames: list[str]) -> tuple[list[str], list[QuestionColumn]]:
    """Split CSV columns into metadata and focus question columns."""
    metadata_fields: list[str] = []
    question_columns: list[QuestionColumn] = []

    for fieldname in fieldnames:
        match = QUESTION_ID_PATTERN.fullmatch(fieldname)
        if match is None:
            metadata_fields.append(fieldname)
            continue

        question_columns.append(
            QuestionColumn(
                column_name=fieldname,
                speaker=match.group(1),
                question_id=match.group(2),
            )
        )

    if not question_columns:
        raise ValueError(
            "No speaker question columns found. Expected headers like speaker0_ns2_item7."
        )

    return metadata_fields, question_columns


def output_paths(input_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve long and summary output paths from the input CSV name."""
    stem = input_csv.stem
    return (
        output_dir / f"{stem}.scored_long.csv",
        output_dir / f"{stem}.scored_summary.csv",
    )


def is_probable_response_row(row: dict[str, str]) -> bool:
    """Ignore Qualtrics metadata rows that appear below the header in some exports."""
    response_id = str(row.get("ResponseId", "")).strip()
    return bool(RESPONSE_ID_PATTERN.fullmatch(response_id))


def selected_word(response_numeric: str, solution: FocusSolution) -> str:
    """Map a numeric response to the corresponding sentence-order word choice."""
    if response_numeric == "1":
        return solution.choice_1_word
    if response_numeric == "2":
        return solution.choice_2_word
    return ""


def score_rows(
    reader: csv.DictReader,
    metadata_fields: list[str],
    question_columns: list[QuestionColumn],
    solution_key: dict[str, FocusSolution],
    question_order: dict[str, int],
    include_empty_responses: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build long-form answer rows and per-response summary rows."""
    long_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for source_row_index, row in enumerate(reader):
        if not is_probable_response_row(row):
            continue

        answered_rows: list[dict[str, str]] = []
        answered_speakers: set[str] = set()

        for question_column in question_columns:
            response_numeric = str(row.get(question_column.column_name, "")).strip()
            if not response_numeric:
                continue

            solution = solution_key.get(question_column.question_id)
            if solution is None:
                raise ValueError(f"Missing focus solution for {question_column.question_id}")

            answered_speakers.add(question_column.speaker)
            response_word = selected_word(response_numeric, solution)
            is_correct = (
                "1"
                if response_numeric == solution.solution_numeric
                else "0"
                if response_numeric in {"1", "2"}
                else ""
            )

            answered_rows.append(
                {
                    "source_row_index": str(source_row_index),
                    **{field: row.get(field, "") for field in metadata_fields},
                    "speaker": question_column.speaker,
                    "question_column": question_column.column_name,
                    "question_id": question_column.question_id,
                    "survey_order": str(question_order.get(question_column.question_id, "")),
                    "sentence_1": solution.sentence_1,
                    "choice_1_word": solution.choice_1_word,
                    "choice_2_word": solution.choice_2_word,
                    "response_numeric": response_numeric,
                    "response_word": response_word,
                    "solution_numeric": solution.solution_numeric,
                    "solution_word": solution.solution_word,
                    "is_correct": is_correct,
                }
            )

        if not answered_rows and not include_empty_responses:
            continue

        speaker_value = ",".join(sorted(answered_speakers))
        answered_count = len(answered_rows)
        correct_count = sum(1 for scored_row in answered_rows if scored_row["is_correct"] == "1")
        incorrect_rows = [
            scored_row for scored_row in answered_rows if scored_row["is_correct"] == "0"
        ]
        incorrect_details = "; ".join(
            (
                f"{scored_row['question_id']}: chose "
                f"{scored_row['response_numeric']}/{scored_row['response_word']}, "
                f"correct {scored_row['solution_numeric']}/{scored_row['solution_word']}"
            )
            for scored_row in sorted(
                incorrect_rows,
                key=lambda scored_row: int(scored_row["survey_order"])
                if scored_row["survey_order"]
                else 999,
            )
        )

        summary_rows.append(
            {
                "source_row_index": str(source_row_index),
                **{field: row.get(field, "") for field in metadata_fields},
                "speaker": speaker_value,
                "answered_questions": str(answered_count),
                "correct_answers": str(correct_count),
                "incorrect_answers": str(len(incorrect_rows)),
                "accuracy_pct": f"{(correct_count / answered_count) * 100:.2f}"
                if answered_count
                else "",
                "is_full_24_item_response": "1"
                if answered_count == EXPECTED_QUESTION_COUNT
                else "0",
                "incorrect_question_ids": "; ".join(
                    scored_row["question_id"]
                    for scored_row in sorted(
                        incorrect_rows,
                        key=lambda scored_row: int(scored_row["survey_order"])
                        if scored_row["survey_order"]
                        else 999,
                    )
                ),
                "incorrect_details": incorrect_details,
            }
        )

        answered_rows.sort(
            key=lambda scored_row: (
                int(scored_row["survey_order"]) if scored_row["survey_order"] else 999,
                scored_row["question_id"],
            )
        )
        long_rows.extend(answered_rows)

    return long_rows, summary_rows


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to CSV using UTF-8 and stable header order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    solution_key = load_solution_key(args.stimuli_dir)
    question_order = load_question_order(args.question_order_txt)

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV {args.input_csv} does not contain a header row")

        metadata_fields, question_columns = infer_question_columns(reader.fieldnames)
        long_rows, summary_rows = score_rows(
            reader,
            metadata_fields,
            question_columns,
            solution_key,
            question_order,
            args.include_empty_responses,
        )

    long_output_path, summary_output_path = output_paths(args.input_csv, args.output_dir)
    long_fieldnames = [
        "source_row_index",
        *metadata_fields,
        "speaker",
        "question_column",
        "question_id",
        "survey_order",
        "sentence_1",
        "choice_1_word",
        "choice_2_word",
        "response_numeric",
        "response_word",
        "solution_numeric",
        "solution_word",
        "is_correct",
    ]
    summary_fieldnames = [
        "source_row_index",
        *metadata_fields,
        "speaker",
        "answered_questions",
        "correct_answers",
        "incorrect_answers",
        "accuracy_pct",
        "is_full_24_item_response",
        "incorrect_question_ids",
        "incorrect_details",
    ]

    write_csv(long_output_path, long_fieldnames, long_rows)
    write_csv(summary_output_path, summary_fieldnames, summary_rows)

    print(f"Wrote {len(long_rows)} scored answer row(s) to {long_output_path}")
    print(f"Wrote {len(summary_rows)} response summary row(s) to {summary_output_path}")


if __name__ == "__main__":
    main()
