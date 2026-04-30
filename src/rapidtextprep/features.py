"""Feature extraction utilities for pandas text columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._patterns import (
    DIGIT_PATTERN,
    DIGIT_SEQUENCE_PATTERN,
    HASHTAG_MULTI_HASH_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    NON_WHITESPACE_CHAR_PATTERN,
    WORD_PATTERN,
)
from .stopwords import get_stopwords_count


def get_word_count(text: pd.Series) -> pd.Series:
    """Return word counts for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.

    Returns:
        A pandas Series containing word counts with dtype int32.

    Example:
        >>> s = pd.Series(["hello world", "  pandas is fast  ", ""])
        >>> get_word_count(s).tolist()
        [2, 3, 0]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    return text.str.count(WORD_PATTERN).astype("int32")


def get_char_count(
    text: pd.Series,
    char_count_with_whitespace: bool = False,
) -> pd.Series:
    """Return character counts for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.
        char_count_with_whitespace: Whether to include whitespace characters
            in the character count. Defaults to False.

    Returns:
        A pandas Series containing character counts with dtype int32.

    Example:
        >>> s = pd.Series(["hello world", " pandas ", ""])
        >>> get_char_count(s).tolist()
        [10, 6, 0]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    if char_count_with_whitespace:
        return text.str.len().astype("int32")

    return text.str.count(NON_WHITESPACE_CHAR_PATTERN).astype("int32")


def get_avg_wordlength(text: pd.Series) -> pd.Series:
    """Return average word length for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.

    Returns:
        A pandas Series containing average word lengths with dtype float32.

    Example:
        >>> s = pd.Series(["hello world", "pandas is fast", "", "a bc def"])
        >>> get_avg_wordlength(s).tolist()
        [5.0, 4.0, 0.0, 2.0]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    char_count = get_char_count(text).to_numpy(dtype=np.float32, copy=False)
    word_count = get_word_count(text).to_numpy(dtype=np.float32, copy=False)

    avg_word_length = np.divide(
        char_count,
        word_count,
        out=np.zeros_like(char_count, dtype=np.float32),
        where=word_count != 0,
    )

    return pd.Series(
        avg_word_length,
        index=text.index,
        name="avg_word_length",
    )


def get_hashtag_count(
    text: pd.Series,
    count_multiple_hash: bool = False,
) -> pd.Series:
    """Return hashtag counts for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.
        count_multiple_hash: Whether to count hashtags with multiple leading
            hash symbols. Defaults to False.

    Returns:
        A pandas Series containing hashtag counts with dtype int32.

    Example:
        >>> s = pd.Series(["I love #Python", "##worldcup2k26"])
        >>> get_hashtag_count(s).tolist()
        [1, 0]
        >>> get_hashtag_count(s, count_multiple_hash=True).tolist()
        [1, 1]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    pattern = HASHTAG_MULTI_HASH_PATTERN if count_multiple_hash else HASHTAG_PATTERN
    return text.str.count(pattern).astype("int32")


def get_mentions_count(text: pd.Series) -> pd.Series:
    """Return mention counts for each row in a pandas Series.

    Args:
        text: A pandas Series containing non-null text values.

    Returns:
        A pandas Series containing mention counts with dtype int32.

    Example:
        >>> s = pd.Series(["thanks @suraj", "hello@example.com"])
        >>> get_mentions_count(s).tolist()
        [1, 0]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    return text.str.count(MENTION_PATTERN).astype("int32")


def get_digit_count(
    text: pd.Series,
    count_continuous_digit_sequences: bool = True,
) -> pd.Series:
    """Return digit counts for each row in a pandas Series.

    By default, continuous digit sequences are counted using ``\\d+``. If
    ``count_continuous_digit_sequences=False``, individual digits are counted.

    Args:
        text: A pandas Series containing non-null text values.
        count_continuous_digit_sequences: Whether to count continuous digit
            sequences instead of individual digits. Defaults to True.

    Returns:
        A pandas Series containing digit counts with dtype int32.

    Example:
        >>> s = pd.Series(["worldcup2k26", "Phone: 9876543210"])
        >>> get_digit_count(s).tolist()
        [2, 1]
        >>> get_digit_count(s, count_continuous_digit_sequences=False).tolist()
        [3, 10]
    """
    if not isinstance(text, pd.Series):
        raise TypeError("text must be a pandas Series")

    pattern = (
        DIGIT_SEQUENCE_PATTERN if count_continuous_digit_sequences else DIGIT_PATTERN
    )

    return text.str.count(pattern).astype("int32")


def get_basic_features(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Create basic text features from a dataframe text column.

    This function computes the following features:
        - char_count
        - word_count
        - avg_word_length
        - stopwords_count
        - hashtag_count
        - mentions_count
        - digit_count

    It computes char_count and word_count only once, then reuses them for
    average word length.

    Args:
        df: Input pandas dataframe.
        column: Name of the text column from which features should be created.

    Returns:
        A pandas DataFrame containing basic text features with the same index
        as the input dataframe.

    Example:
        >>> data = pd.DataFrame({"text": ["python is great #nlp"]})
        >>> get_basic_features(data, "text")
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(column, str):
        raise TypeError("column must be a string")

    if column not in df.columns:
        raise KeyError(f"{column!r} column not found in dataframe")

    text = df[column]

    char_count = get_char_count(text)
    word_count = get_word_count(text)

    char_count_array = char_count.to_numpy(dtype=np.float32, copy=False)
    word_count_array = word_count.to_numpy(dtype=np.float32, copy=False)

    avg_word_length = np.divide(
        char_count_array,
        word_count_array,
        out=np.zeros_like(char_count_array, dtype=np.float32),
        where=word_count_array != 0,
    )

    return pd.DataFrame(
        {
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "stopwords_count": get_stopwords_count(text),
            "hashtag_count": get_hashtag_count(text),
            "mentions_count": get_mentions_count(text),
            "digit_count": get_digit_count(text),
        },
        index=df.index,
    )
