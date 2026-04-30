"""Frequency-based word utilities."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from ._patterns import WORD_EXTRACT_PATTERN
from ._typing import TextInput, TextOutput


def get_value_counts(text: pd.Series) -> pd.Series:
    """Return word frequency counts across an entire pandas Series.

    Args:
        text: A pandas Series containing non-null text values.

    Returns:
        A pandas Series where the index contains words and values contain total
        counts across the full input Series.

    Example:
        >>> s = pd.Series(["python is fast", "python is popular"])
        >>> get_value_counts(s).to_dict()
        {'python': 2, 'is': 2, 'fast': 1, 'popular': 1}
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    return text.str.extractall(WORD_EXTRACT_PATTERN)[0].value_counts()


@lru_cache(maxsize=64)
def _compile_word_removal_pattern(
    words: tuple[str, ...],
    ignore_case: bool,
) -> re.Pattern[str] | None:
    """Compile and cache a word-removal regex pattern."""
    clean_words = tuple(
        sorted(
            {word for word in words if word},
            key=len,
            reverse=True,
        )
    )

    if not clean_words:
        return None

    return re.compile(
        r"(?<![\w'])"
        r"(?:" + "|".join(re.escape(word) for word in clean_words) + r")"
        r"(?![\w'])",
        flags=re.IGNORECASE if ignore_case else 0,
    )


def remove_common_word(
    text: TextInput,
    word_counts: pd.Series,
    n_words: int,
    ignore_case: bool = False,
) -> TextOutput:
    """Remove top N most common words from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.
        word_counts: A pandas Series returned by get_value_counts.
        n_words: Number of most common words to remove.
        ignore_case: Whether removal should be case-insensitive.

    Returns:
        Text with common words removed.
    """
    if not isinstance(word_counts, pd.Series):
        raise TypeError("word_counts must be a pandas Series")

    if n_words < 0:
        raise ValueError("n_words must be greater than or equal to 0")

    if n_words == 0 or word_counts.empty:
        return text

    words_to_remove = tuple(
        str(word) for word in word_counts.head(n_words).index if str(word)
    )

    pattern = _compile_word_removal_pattern(
        words=words_to_remove,
        ignore_case=ignore_case,
    )

    if pattern is None:
        return text

    if isinstance(text, str):
        return pattern.sub(" ", text)

    if isinstance(text, pd.Series):
        return text.str.replace(pattern, " ", regex=True)

    raise TypeError("text must be a string or pandas Series")


def remove_rarewords(
    text: TextInput,
    word_counts: pd.Series,
    n_words: int,
    ignore_case: bool = False,
) -> TextOutput:
    """Remove bottom N rare words from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.
        word_counts: A pandas Series returned by get_value_counts.
        n_words: Number of least common words to remove.
        ignore_case: Whether removal should be case-insensitive.

    Returns:
        Text with rare words removed.
    """
    if not isinstance(word_counts, pd.Series):
        raise TypeError("word_counts must be a pandas Series")

    if n_words < 0:
        raise ValueError("n_words must be greater than or equal to 0")

    if n_words == 0 or word_counts.empty:
        return text

    words_to_remove = tuple(
        str(word) for word in word_counts.tail(n_words).index if str(word)
    )

    pattern = _compile_word_removal_pattern(
        words=words_to_remove,
        ignore_case=ignore_case,
    )

    if pattern is None:
        return text

    if isinstance(text, str):
        return pattern.sub(" ", text)

    if isinstance(text, pd.Series):
        return text.str.replace(pattern, " ", regex=True)

    raise TypeError("text must be a string or pandas Series")
