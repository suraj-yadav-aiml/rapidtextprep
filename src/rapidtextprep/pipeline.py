"""Composable text preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from ._typing import TextInput, TextOutput
from .cleaning import (
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)
from .lemmatization import get_lemmatize_text_fast
from .normalization import (
    get_contraction_to_expansion,
    get_expand_abbreviations,
    get_lower_case,
    remove_accented_chars,
    remove_multiple_whitespaces,
)
from .stopwords import DEFAULT_KEEP_STOPWORDS, remove_stopwords


def _clean_text_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    use_lemmatization: bool,
    lemmatize_batch_size: int,
    n_process: int,
) -> TextOutput:
    """Clean one string or one pandas Series batch."""
    result = get_lower_case(text)

    result = get_contraction_to_expansion(result)
    result = get_expand_abbreviations(result)

    result = remove_accented_chars(result)

    result = remove_html_tags(result)
    result = remove_email(result)
    result = remove_urls(result)
    result = remove_rt(result)

    result = remove_special_characters(result)

    result = remove_stopwords(
        result,
        keep_words=keep_stopwords,
        extra_stopwords=extra_stopwords,
        ignore_case=False,
    )

    if use_lemmatization:
        result = get_lemmatize_text_fast(
            result,
            batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

    result = remove_multiple_whitespaces(result)

    return result


def get_complete_text_clean_up_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    use_lemmatization: bool = False,
    chunk_size: int | None = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
) -> TextOutput:
    """Perform complete batched text cleaning on a string or pandas Series.

    Pipeline order:
        1. Convert text to lowercase.
        2. Expand contractions.
        3. Expand social-media/texting abbreviations.
        4. Remove accented characters.
        5. Remove HTML tags.
        6. Remove email addresses.
        7. Remove URLs.
        8. Remove retweet markers.
        9. Remove special characters.
        10. Remove stopwords.
        11. Optionally lemmatize text using fast spaCy lookup lemmatization.
        12. Remove multiple consecutive whitespaces.

    Args:
        text: A single text string or pandas Series containing non-null text.
        keep_stopwords: Stopwords to keep during stopword removal. Defaults to
            DEFAULT_KEEP_STOPWORDS. Use None to remove all stopwords.
        extra_stopwords: Additional words to remove along with STOPWORDS.
        use_lemmatization: Whether to apply fast lookup-based lemmatization.
            Defaults to False.
        chunk_size: Number of rows to process per chunk for pandas Series
            input. Defaults to 100,000. Use None to process the full Series at
            once.
        lemmatize_batch_size: Batch size used by get_lemmatize_text_fast.
        n_process: Number of spaCy processes used during lemmatization.

    Returns:
        Fully cleaned text. If input is a string, returns a string. If input is
        a pandas Series, returns a pandas Series with the same index.

    Example:
        >>> raw = "RT @User: I CAN'T believe this caf\xe9 is 50% OFF!!! Visit https://shop.com"
        >>> get_complete_text_clean_up_batch(raw)
        'not believe cafe percent'
    """
    if isinstance(text, str):
        return _clean_text_batch(
            text=text,
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            use_lemmatization=use_lemmatization,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

    if not isinstance(text, pd.Series):
        raise TypeError("text must be a string or pandas Series")

    if chunk_size is None:
        return _clean_text_batch(
            text=text,
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            use_lemmatization=use_lemmatization,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    cleaned_chunks: list[pd.Series] = []

    for start in range(0, len(text), chunk_size):
        chunk = text.iloc[start : start + chunk_size]

        cleaned_chunk = _clean_text_batch(
            text=chunk,
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            use_lemmatization=use_lemmatization,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

        cleaned_chunks.append(cleaned_chunk)

    return pd.concat(cleaned_chunks)


def clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    use_lemmatization: bool = False,
    chunk_size: int = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
) -> pd.DataFrame:
    """Clean a text column in dataframe chunks.

    This version writes each cleaned chunk directly into the dataframe.

    Args:
        df: Input dataframe.
        source_col: Name of source text column.
        target_col: Name of output clean text column.
        keep_stopwords: Stopwords to keep during stopword removal.
        extra_stopwords: Additional stopwords to remove.
        use_lemmatization: Whether to apply fast lookup lemmatization.
        chunk_size: Number of rows to clean per chunk.
        lemmatize_batch_size: Batch size used by spaCy nlp.pipe.
        n_process: Number of spaCy processes.

    Returns:
        The dataframe with the cleaned text column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if source_col not in df.columns:
        raise KeyError(f"{source_col!r} column not found in dataframe")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    df[target_col] = ""

    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        index_slice = df.index[start:end]

        cleaned = get_complete_text_clean_up_batch(
            df[source_col].iloc[start:end],
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            use_lemmatization=use_lemmatization,
            chunk_size=None,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

        df.loc[index_slice, target_col] = cleaned.to_numpy(copy=False)

    return df


clean_text = get_complete_text_clean_up_batch
