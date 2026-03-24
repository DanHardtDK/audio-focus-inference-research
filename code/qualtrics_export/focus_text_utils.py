#!/usr/bin/env python3
"""Helpers for parsing and normalizing survey sentence text."""

from __future__ import annotations

import re

from qualtrics_export_common import sanitize_qualtrics_text


DELIVERY_PATTERN = re.compile(
    r"^\s*(?P<subject>.+?)\s+(?P<relation>only\s+gave|didn't\s+give|also\s+gave)\s+(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def parse_delivery_components(sentence: str) -> tuple[str, str, str, str]:
    """Parse a survey sentence into subject, relation, object1, and object2."""
    sanitized_sentence = sanitize_qualtrics_text(sentence)
    match = DELIVERY_PATTERN.match(sanitized_sentence.rstrip(".!?"))
    if not match:
        raise ValueError(f"Could not parse delivery sentence structure: {sentence!r}")

    subject = match.group("subject")
    relation = " ".join(match.group("relation").split()).lower()
    tail = match.group("tail")
    objects = TOKEN_PATTERN.findall(tail)
    if len(objects) != 2:
        raise ValueError(
            f"Expected exactly two objects after the relation phrase, found {len(objects)}: {sentence!r}"
        )

    return subject, relation, objects[0], objects[1]


def parse_s1_components(s1: str) -> tuple[str, str, str]:
    """Parse an S1 sentence into subject, object1, and object2."""
    subject, relation, object1, object2 = parse_delivery_components(s1)
    if relation != "only gave":
        raise ValueError(f"S1 must use 'only gave': {s1!r}")
    return subject, object1, object2


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


def normalize_delivery_sentence(sentence: str) -> str:
    """Return a normalized delivery sentence without uppercase cues."""
    subject, relation, object1, object2 = parse_delivery_components(sentence)
    normalized_subject = sanitize_qualtrics_text(subject)
    normalized_relation = relation
    normalized_object1 = normalize_object1(object1)
    normalized_object2 = normalize_object2(object2)
    return f"{normalized_subject} {normalized_relation} {normalized_object1} {normalized_object2}."


def normalize_s2(s2: str) -> str:
    """Return a normalized S2 string without uppercase cues."""
    return normalize_delivery_sentence(s2)


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
