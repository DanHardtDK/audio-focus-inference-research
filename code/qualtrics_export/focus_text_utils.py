#!/usr/bin/env python3
"""Helpers for parsing and normalizing focus-survey sentence text."""

from __future__ import annotations

import re

from qualtrics_export_common import sanitize_qualtrics_text


S1_PATTERN = re.compile(r"^\s*(?P<subject>.+?)\s+only\s+gave\s+(?P<tail>.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def parse_s1_components(s1: str) -> tuple[str, str, str]:
    """Parse an S1 sentence into subject, object1, and object2."""
    sanitized_s1 = sanitize_qualtrics_text(s1)
    match = S1_PATTERN.match(sanitized_s1.rstrip(".!?"))
    if not match:
        raise ValueError(f"Could not parse S1 structure: {s1!r}")

    subject = match.group("subject")
    tail = match.group("tail")
    objects = TOKEN_PATTERN.findall(tail)
    if len(objects) != 2:
        raise ValueError(
            f"Expected exactly two objects after 'only gave' in S1, found {len(objects)}: {s1!r}"
        )

    return subject, objects[0], objects[1]


def normalize_object1(token: str) -> str:
    """Normalize the person/name slot in S1."""
    return token[:1].upper() + token[1:].lower() if token else token


def normalize_object2(token: str) -> str:
    """Normalize the thing/object slot in S1."""
    return token.lower()


def normalize_s1(s1: str) -> str:
    """Return a normalized S1 string without uppercase focus cues."""
    subject, object1, object2 = parse_s1_components(s1)
    normalized_subject = sanitize_qualtrics_text(subject)
    normalized_object1 = normalize_object1(object1)
    normalized_object2 = normalize_object2(object2)
    return f"{normalized_subject} only gave {normalized_object1} {normalized_object2}."


def extract_focus_position(s1: str, focus: int) -> int:
    """Determine which S1 object carries focus, preferring the uppercase cue when present."""
    _, object1, object2 = parse_s1_components(s1)
    uppercase_flags = [
        any(char.isalpha() for char in object1) and object1 == object1.upper(),
        any(char.isalpha() for char in object2) and object2 == object2.upper(),
    ]

    if uppercase_flags == [True, False]:
        return 1
    if uppercase_flags == [False, True]:
        return 2
    if uppercase_flags == [True, True]:
        raise ValueError(f"Found multiple uppercase focus candidates in S1: {s1!r}")
    if focus in (1, 2):
        return focus
    raise ValueError(f"Unsupported focus value {focus!r} for S1: {s1!r}")


def normalized_focus_choices(s1: str, focus: int) -> tuple[str, str]:
    """Return normalized focus and alternative choices for S1."""
    _, object1, object2 = parse_s1_components(s1)
    normalized_object1 = normalize_object1(object1)
    normalized_object2 = normalize_object2(object2)
    focus_position = extract_focus_position(s1, focus)
    if focus_position == 1:
        return normalized_object1, normalized_object2
    return normalized_object2, normalized_object1
