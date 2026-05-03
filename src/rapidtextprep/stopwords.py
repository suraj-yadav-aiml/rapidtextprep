"""Stopword resources and removal utilities."""

from __future__ import annotations

import re
from collections.abc import Collection
from functools import lru_cache
from typing import Literal

import pandas as pd

from ._typing import TextInput, TextOutput

StopwordBackend = Literal["regex", "flashtext"]

_FALLBACK_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:
    ENGLISH_STOP_WORDS = _FALLBACK_STOP_WORDS

try:
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except ImportError:
    SPACY_STOP_WORDS = _FALLBACK_STOP_WORDS

STOPWORDS: frozenset[str] = frozenset(
    str(word).lower() for word in (set(SPACY_STOP_WORDS) | set(ENGLISH_STOP_WORDS))
)

STOPWORDS_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![\w'])"
    r"(?:"
    + "|".join(re.escape(word) for word in sorted(STOPWORDS, key=len, reverse=True))
    + r")"
    r"(?![\w'])",
    flags=re.IGNORECASE,
)


def get_stopwords_count(text: pd.Series) -> pd.Series:
    """Return stopword counts for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.

    Returns:
        A pandas Series containing stopword counts with dtype int32.

    Example:
        >>> s = pd.Series(["this is a good movie", "hello world"])
        >>> get_stopwords_count(s).tolist()
        [3, 0]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    return text.str.count(STOPWORDS_PATTERN).astype("int32")


DEFAULT_KEEP_STOPWORDS: frozenset[str] = frozenset(
    {
        # Negation
        "no",
        "not",
        "nor",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "without",
        "cannot",
        # Contrast
        "but",
        "however",
        "although",
        "though",
        "yet",
        "nevertheless",
        "nonetheless",
        "whereas",
        "while",
        "despite",
        "unless",
        "except",
        # Sentiment intensity / degree modifiers
        "very",
        "too",
        "really",
        "quite",
        "rather",
        "so",
        "such",
        "more",
        "most",
        "less",
        "least",
        "much",
    }
)


def _words_to_cache_key(words: Collection[str] | None) -> tuple[str, ...]:
    """Convert a word collection into a stable cache key."""
    if words is None:
        return ()

    return tuple(
        sorted({word.lower() for word in words if isinstance(word, str) and word})
    )


def _get_stopwords_to_remove(
    keep_words_key: tuple[str, ...],
    extra_stopwords_key: tuple[str, ...],
) -> set[str]:
    """Return the final stopword set after applying keep and extra words."""
    stopwords_to_remove = set(STOPWORDS)
    stopwords_to_remove.difference_update(keep_words_key)
    stopwords_to_remove.update(extra_stopwords_key)
    return {word for word in stopwords_to_remove if word}


@lru_cache(maxsize=64)
def _compile_stopword_removal_pattern(
    keep_words_key: tuple[str, ...],
    extra_stopwords_key: tuple[str, ...],
    ignore_case: bool,
) -> re.Pattern[str] | None:
    """Compile and cache a stopword removal regex pattern."""
    stopwords_to_remove = _get_stopwords_to_remove(
        keep_words_key=keep_words_key,
        extra_stopwords_key=extra_stopwords_key,
    )

    if not stopwords_to_remove:
        return None

    return re.compile(
        r"(?<![\w'])"
        r"(?:"
        + "|".join(
            re.escape(word)
            for word in sorted(stopwords_to_remove, key=len, reverse=True)
        )
        + r")"
        r"(?![\w'])",
        flags=re.IGNORECASE if ignore_case else 0,
    )


DEFAULT_STOPWORD_REMOVAL_PATTERN: re.Pattern[str] | None = (
    _compile_stopword_removal_pattern(
        keep_words_key=_words_to_cache_key(DEFAULT_KEEP_STOPWORDS),
        extra_stopwords_key=(),
        ignore_case=False,
    )
)


@lru_cache(maxsize=64)
def _compile_flashtext_stopword_processor(
    keep_words_key: tuple[str, ...],
    extra_stopwords_key: tuple[str, ...],
    ignore_case: bool,
):
    """Compile and cache a FlashText keyword processor for stopword removal."""
    try:
        from flashtext import KeywordProcessor
    except ImportError as exc:
        raise ImportError(
            "FlashText is required for stopword_backend='flashtext'. "
            "Install it with: pip install flashtext"
        ) from exc

    stopwords_to_remove = _get_stopwords_to_remove(
        keep_words_key=keep_words_key,
        extra_stopwords_key=extra_stopwords_key,
    )

    if not stopwords_to_remove:
        return None

    processor = KeywordProcessor(case_sensitive=not ignore_case)

    for char in ("'", "\u2019", "\u2018", "\u201b", "`", "\xb4"):
        processor.add_non_word_boundary(char)

    for word in sorted(stopwords_to_remove, key=len, reverse=True):
        processor.add_keyword(word, " ")

    return processor


def _validate_stopword_backend(backend: StopwordBackend) -> None:
    """Validate the stopword removal backend."""
    if backend not in {"regex", "flashtext"}:
        raise ValueError("stopword_backend must be 'regex' or 'flashtext'")


def remove_stopwords(
    text: TextInput,
    keep_words: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    ignore_case: bool = False,
    backend: StopwordBackend = "regex",
) -> TextOutput:
    """Remove stopwords from a string or pandas Series.

    By default, this keeps negation, contrast, and sentiment-important words
    such as "not", "no", "but", and "however".

    Args:
        text: A single text string or pandas Series.
        keep_words: Stopwords that should not be removed. Use None to remove
            all stopwords.
        extra_stopwords: Additional words to remove.
        ignore_case: Whether to remove stopwords case-insensitively.
        backend: Stopword removal backend. Use "regex" for the original
            pandas vectorized implementation or "flashtext" for trie-based
            keyword replacement on large stopword lists.

    Returns:
        Text with stopwords removed.

    Example:
        >>> remove_stopwords("this movie is not good but very emotional")
        ' movie  not good but very emotional'
    """
    _validate_stopword_backend(backend)

    use_default_options = (
        keep_words is DEFAULT_KEEP_STOPWORDS
        and extra_stopwords is None
        and not ignore_case
    )

    if backend == "regex":
        if use_default_options:
            pattern = DEFAULT_STOPWORD_REMOVAL_PATTERN
        else:
            pattern = _compile_stopword_removal_pattern(
                keep_words_key=_words_to_cache_key(keep_words),
                extra_stopwords_key=_words_to_cache_key(extra_stopwords),
                ignore_case=ignore_case,
            )

        if pattern is None:
            return text

        if isinstance(text, str):
            return pattern.sub(" ", text)

        if isinstance(text, pd.Series):
            return text.str.replace(pattern, " ", regex=True)

        raise TypeError("text must be a string or pandas Series")

    if use_default_options:
        processor = _compile_flashtext_stopword_processor(
            keep_words_key=_words_to_cache_key(DEFAULT_KEEP_STOPWORDS),
            extra_stopwords_key=(),
            ignore_case=False,
        )
    else:
        processor = _compile_flashtext_stopword_processor(
            keep_words_key=_words_to_cache_key(keep_words),
            extra_stopwords_key=_words_to_cache_key(extra_stopwords),
            ignore_case=ignore_case,
        )

    if processor is None:
        return text

    if isinstance(text, str):
        return processor.replace_keywords(text)

    if isinstance(text, pd.Series):
        return text.map(processor.replace_keywords)

    raise TypeError("text must be a string or pandas Series")
