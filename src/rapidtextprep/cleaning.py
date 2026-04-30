"""Text cleaning utilities for removing noisy text patterns."""

from __future__ import annotations

import pandas as pd

from ._patterns import (
    EMAIL_PATTERN,
    HTML_TAG_PATTERN,
    RT_PATTERN,
    SPECIAL_CHARACTER_PATTERN,
    URL_PATTERN,
)
from ._typing import TextInput, TextOutput


def remove_email(text: TextInput) -> TextOutput:
    """Remove email addresses from a string or pandas Series.

    Whitespace normalization is intentionally not handled here.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with email addresses removed.

    Example:
        >>> remove_email("contact me at suraj@example.com")
        'contact me at '
    """
    if isinstance(text, str):
        return EMAIL_PATTERN.sub("", text)

    if isinstance(text, pd.Series):
        return text.str.replace(EMAIL_PATTERN, "", regex=True)

    raise TypeError("text must be a string or pandas Series")


def remove_urls(text: TextInput) -> TextOutput:
    """Remove URLs from a string or pandas Series.

    Whitespace normalization is intentionally not handled here.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with URLs removed.

    Example:
        >>> remove_urls("visit https://example.com now")
        'visit  now'
    """
    if isinstance(text, str):
        return URL_PATTERN.sub("", text)

    if isinstance(text, pd.Series):
        return text.str.replace(URL_PATTERN, "", regex=True)

    raise TypeError("text must be a string or pandas Series")


def remove_rt(text: TextInput) -> TextOutput:
    """Remove standalone retweet markers from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with standalone RT markers removed.

    Example:
        >>> remove_rt("RT @user: this is amazing")
        ' @user: this is amazing'
    """
    if isinstance(text, str):
        return RT_PATTERN.sub("", text)

    if isinstance(text, pd.Series):
        return text.str.replace(RT_PATTERN, "", regex=True)

    raise TypeError("text must be a string or pandas Series")


def remove_special_characters(text: TextInput) -> TextOutput:
    """Remove special characters from a string or pandas Series.

    Punctuation, symbols, emojis, and underscores are replaced with spaces.
    Letters, digits, and whitespace are kept.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with special characters replaced by spaces.

    Example:
        >>> remove_special_characters("hello!!! world #python @user")
        'hello    world  python  user'
    """
    if isinstance(text, str):
        return SPECIAL_CHARACTER_PATTERN.sub(" ", text)

    if isinstance(text, pd.Series):
        return text.str.replace(SPECIAL_CHARACTER_PATTERN, " ", regex=True)

    raise TypeError("text must be a string or pandas Series")


def remove_html_tags(text: TextInput) -> TextOutput:
    """Remove HTML/XML-like tags from a string or pandas Series.

    Tags are replaced with spaces. Whitespace normalization is intentionally
    not handled here.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with HTML tags removed.

    Example:
        >>> remove_html_tags("<p>Hello <b>world</b></p>")
        ' Hello  world  '
    """
    if isinstance(text, str):
        return HTML_TAG_PATTERN.sub(" ", text)

    if isinstance(text, pd.Series):
        return text.str.replace(HTML_TAG_PATTERN, " ", regex=True)

    raise TypeError("text must be a string or pandas Series")
