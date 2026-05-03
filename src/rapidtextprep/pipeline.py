"""Composable text preprocessing pipelines."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Literal

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

ParallelBackend = Literal["thread", "process"]


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve the requested parallel worker count."""
    if n_jobs == -1:
        return os.cpu_count() or 1

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or greater than 0")

    return n_jobs


def _validate_parallel_backend(parallel_backend: ParallelBackend) -> None:
    """Validate the requested parallel execution backend."""
    if parallel_backend not in {"thread", "process"}:
        raise ValueError("parallel_backend must be 'thread' or 'process'")


def _iter_series_chunks(text: pd.Series, chunk_size: int) -> list[pd.Series]:
    """Split a Series into positional chunks."""
    return [
        text.iloc[start : start + chunk_size]
        for start in range(0, len(text), chunk_size)
    ]


def _pre_lemmatization_clean_text_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
) -> TextOutput:
    """Run text cleaning stages that are safe to apply per chunk."""
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

    return result


def _finalize_clean_text(
    text: TextOutput,
    use_lemmatization: bool,
    lemmatize_batch_size: int,
    n_process: int,
) -> TextOutput:
    """Apply optional lemmatization and final whitespace normalization."""
    result = text

    if use_lemmatization:
        result = get_lemmatize_text_fast(
            result,
            batch_size=lemmatize_batch_size,
            n_process=n_process,
        )

    result = remove_multiple_whitespaces(result)

    return result


def _clean_text_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    use_lemmatization: bool,
    lemmatize_batch_size: int,
    n_process: int,
) -> TextOutput:
    """Clean one string or one pandas Series batch."""
    result = _pre_lemmatization_clean_text_batch(
        text=text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
    )

    return _finalize_clean_text(
        text=result,
        use_lemmatization=use_lemmatization,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
    )


def _clean_series_chunks(
    text: pd.Series,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    chunk_size: int,
    n_jobs: int,
    parallel_backend: ParallelBackend,
) -> pd.Series:
    """Clean chunkable stages for a Series, optionally using an executor."""
    chunks = _iter_series_chunks(text, chunk_size)

    if not chunks:
        return pd.Series([], index=text.index, name=text.name, dtype=object)

    worker = partial(
        _pre_lemmatization_clean_text_batch,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
    )

    if n_jobs == 1 or len(chunks) == 1:
        cleaned_chunks = [worker(chunk) for chunk in chunks]
    else:
        max_workers = min(n_jobs, len(chunks))
        executor_class = (
            ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        )
        with executor_class(max_workers=max_workers) as executor:
            cleaned_chunks = list(executor.map(worker, chunks))

    return pd.concat(cleaned_chunks)


def get_complete_text_clean_up_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    use_lemmatization: bool = False,
    chunk_size: int | None = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
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
        n_jobs: Number of worker threads for chunked pre-lemmatization
            cleaning. Use 1 for sequential execution and -1 for all CPUs.
        parallel_backend: Executor backend for chunked cleaning. Use "thread"
            for lower overhead or "process" for CPU-bound workloads.

    Returns:
        Fully cleaned text. If input is a string, returns a string. If input is
        a pandas Series, returns a pandas Series with the same index.

    Example:
        >>> raw = "RT @User: I CAN'T believe this caf\xe9 is 50% OFF!!! Visit https://shop.com"
        >>> get_complete_text_clean_up_batch(raw)
        'not believe cafe percent'
    """
    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    _validate_parallel_backend(parallel_backend)

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

    result = _clean_series_chunks(
        text=text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        chunk_size=chunk_size,
        n_jobs=resolved_n_jobs,
        parallel_backend=parallel_backend,
    )

    return _finalize_clean_text(
        text=result,
        use_lemmatization=use_lemmatization,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
    )


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
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
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
        n_jobs: Number of worker threads for chunked pre-lemmatization
            cleaning. Use 1 for sequential execution and -1 for all CPUs.
        parallel_backend: Executor backend for chunked cleaning. Use "thread"
            for lower overhead or "process" for CPU-bound workloads.

    Returns:
        The dataframe with the cleaned text column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if source_col not in df.columns:
        raise KeyError(f"{source_col!r} column not found in dataframe")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    cleaned = get_complete_text_clean_up_batch(
        df[source_col],
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )

    df[target_col] = cleaned.to_numpy(copy=False)

    return df


async def async_get_complete_text_clean_up_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    use_lemmatization: bool = False,
    chunk_size: int | None = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
) -> TextOutput:
    """Run get_complete_text_clean_up_batch in the default executor."""
    loop = asyncio.get_running_loop()
    worker = partial(
        get_complete_text_clean_up_batch,
        text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    return await loop.run_in_executor(None, worker)


async def async_clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    use_lemmatization: bool = False,
    chunk_size: int = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
) -> pd.DataFrame:
    """Run clean_text_column_in_chunks in the default executor."""
    loop = asyncio.get_running_loop()
    worker = partial(
        clean_text_column_in_chunks,
        df,
        source_col=source_col,
        target_col=target_col,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    return await loop.run_in_executor(None, worker)


clean_text = get_complete_text_clean_up_batch
async_clean_text = async_get_complete_text_clean_up_batch
