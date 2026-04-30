"""Compiled regular expressions shared across preprocessing modules."""

from __future__ import annotations

import re

WORD_PATTERN: re.Pattern[str] = re.compile(r"\S+")

WORD_EXTRACT_PATTERN: re.Pattern[str] = re.compile(r"(\S+)")

NON_WHITESPACE_CHAR_PATTERN: re.Pattern[str] = re.compile(r"\S")

DIGIT_PATTERN: re.Pattern[str] = re.compile(r"\d")

DIGIT_SEQUENCE_PATTERN: re.Pattern[str] = re.compile(r"\d+")

MULTIPLE_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")

HASHTAG_PATTERN: re.Pattern[str] = re.compile(r"(?<![\w#])#[\w]+")

HASHTAG_MULTI_HASH_PATTERN: re.Pattern[str] = re.compile(r"(?<![\w#])#+[\w]+")

MENTION_PATTERN: re.Pattern[str] = re.compile(r"(?<![\w@.])@[A-Za-z0-9_]+")

RT_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![\w@#])rt\b:?",
    flags=re.IGNORECASE,
)

SPECIAL_CHARACTER_PATTERN: re.Pattern[str] = re.compile(
    r"[^\w\s]|_",
    flags=re.UNICODE,
)

HTML_TAG_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")

EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

EMAIL_EXTRACT_PATTERN: re.Pattern[str] = re.compile(
    r"("
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
    r")"
)

URL_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![@\w])"
    r"("
    r"(?:https?://|www\.)"
    r"[^\s<>()\[\]{}\"']*[^\s<>()\[\]{}\"'.,!?;:]"
    r"|"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,}"
    r"(?:/[^\s<>()\[\]{}\"']*[^\s<>()\[\]{}\"'.,!?;:])?"
    r")",
    flags=re.IGNORECASE,
)
