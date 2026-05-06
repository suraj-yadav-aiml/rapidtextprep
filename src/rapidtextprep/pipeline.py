"""Composable text preprocessing pipelines."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from .stopwords import DEFAULT_KEEP_STOPWORDS, StopwordBackend, remove_stopwords

ParallelBackend = Literal["thread", "process"]


class _VerboseReporter:
    """Print human-readable pipeline progress when verbose mode is enabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._started_at = 0.0

    def start_run(
        self,
        *,
        input_type: str,
        rows: int,
        chunk_size: object,
        chunks: object,
        stopword_backend: str,
        parallel_backend: object,
        n_jobs: object,
        use_lemmatization: bool,
        n_process: int,
    ) -> None:
        """Print the run header and configuration."""
        if not self.enabled:
            return

        self._started_at = time.perf_counter()
        self._print("rapidtextprep clean_text")
        self._print("Input")
        self._field("type", input_type)
        self._field("rows", rows)
        self._field("chunk_size", chunk_size)
        self._field("chunks", chunks)

        self._print("")
        self._print("Processing")
        self._field("stopword_backend", stopword_backend)
        self._field("parallel_backend", parallel_backend)
        self._field("n_jobs", n_jobs)
        self._field("lemmatization", "enabled" if use_lemmatization else "disabled")
        self._field("spacy_n_process", n_process if use_lemmatization else "not used")

        self._print("")
        self._print("Stages")

    def stage_started(self, stage: int, total: int, name: str) -> None:
        """Print a stage start message."""
        if self.enabled:
            self._print(f"  [{stage}/{total}] {name} started")

    def stage_done(self, stage: int, total: int, name: str, elapsed: float) -> None:
        """Print a stage completion message."""
        if self.enabled:
            self._print(
                f"  [{stage}/{total}] {name} done in {self._format_elapsed(elapsed)}"
            )

    def stage_skipped(self, stage: int, total: int, name: str) -> None:
        """Print a stage skip message."""
        if self.enabled:
            self._print(f"  [{stage}/{total}] {name} skipped")

    def chunk_done(self, chunk_number: int, total_chunks: int, elapsed: float) -> None:
        """Print a chunk completion message."""
        if self.enabled:
            self._print(
                "        "
                f"chunk {chunk_number}/{total_chunks} "
                f"done in {self._format_elapsed(elapsed)}"
            )

    def done(self, rows: int) -> None:
        """Print the final run summary."""
        if not self.enabled:
            return

        elapsed = time.perf_counter() - self._started_at
        elapsed_minutes = elapsed / 60
        rows_per_second = rows / elapsed if elapsed > 0 else 0.0

        self._print("")
        self._print("Done")
        self._field("total_time", f"{elapsed:.2f}s ({elapsed_minutes:.2f} min)")
        self._field("rows_per_second", f"{rows_per_second:.2f}")

    def _field(self, label: str, value: object) -> None:
        self._print(f"  {label:<18}: {value}")

    def _format_elapsed(self, elapsed: float) -> str:
        minutes, seconds = divmod(elapsed, 60)
        return f"{int(minutes)}m {seconds:05.2f}s"

    def _print(self, message: str) -> None:
        print(message, flush=True)


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


def _validate_stopword_backend(stopword_backend: StopwordBackend) -> None:
    """Validate the requested stopword removal backend."""
    if stopword_backend not in {"regex", "flashtext"}:
        raise ValueError("stopword_backend must be 'regex' or 'flashtext'")


def _iter_series_chunks(text: pd.Series, chunk_size: int) -> list[pd.Series]:
    """Split a Series into positional chunks."""
    return [
        text.iloc[start : start + chunk_size]
        for start in range(0, len(text), chunk_size)
    ]


def _count_series_chunks(text: pd.Series, chunk_size: int) -> int:
    """Return the number of positional chunks for a Series."""
    if len(text) == 0:
        return 0

    return (len(text) + chunk_size - 1) // chunk_size


def _pre_lemmatization_clean_text_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    stopword_backend: StopwordBackend,
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
        backend=stopword_backend,
    )

    return result


def _clean_chunk_with_elapsed(
    chunk_index: int,
    chunk: pd.Series,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    stopword_backend: StopwordBackend,
) -> tuple[int, pd.Series, float]:
    """Clean one Series chunk and return its elapsed time."""
    started_at = time.perf_counter()
    cleaned = _pre_lemmatization_clean_text_batch(
        text=chunk,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
    )
    return chunk_index, cleaned, time.perf_counter() - started_at


def _finalize_clean_text(
    text: TextOutput,
    use_lemmatization: bool,
    lemmatize_batch_size: int,
    n_process: int,
    reporter: _VerboseReporter,
) -> TextOutput:
    """Apply optional lemmatization and final whitespace normalization."""
    result = text

    if use_lemmatization:
        reporter.stage_started(2, 3, "spaCy lemmatization")
        started_at = time.perf_counter()
        result = get_lemmatize_text_fast(
            result,
            batch_size=lemmatize_batch_size,
            n_process=n_process,
        )
        reporter.stage_done(
            2,
            3,
            "spaCy lemmatization",
            time.perf_counter() - started_at,
        )
    else:
        reporter.stage_skipped(2, 3, "spaCy lemmatization")

    reporter.stage_started(3, 3, "Whitespace normalization")
    started_at = time.perf_counter()
    result = remove_multiple_whitespaces(result)
    reporter.stage_done(
        3,
        3,
        "Whitespace normalization",
        time.perf_counter() - started_at,
    )

    return result


def _clean_text_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    stopword_backend: StopwordBackend,
    use_lemmatization: bool,
    lemmatize_batch_size: int,
    n_process: int,
    reporter: _VerboseReporter,
) -> TextOutput:
    """Clean one string or one pandas Series batch."""
    reporter.stage_started(1, 3, "Pre-lemmatization cleaning")
    started_at = time.perf_counter()
    result = _pre_lemmatization_clean_text_batch(
        text=text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
    )
    reporter.stage_done(
        1,
        3,
        "Pre-lemmatization cleaning",
        time.perf_counter() - started_at,
    )

    return _finalize_clean_text(
        text=result,
        use_lemmatization=use_lemmatization,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        reporter=reporter,
    )


def _clean_series_chunks(
    text: pd.Series,
    keep_stopwords: Collection[str] | None,
    extra_stopwords: Collection[str] | None,
    stopword_backend: StopwordBackend,
    chunk_size: int,
    n_jobs: int,
    parallel_backend: ParallelBackend,
    reporter: _VerboseReporter,
) -> pd.Series:
    """Clean chunkable stages for a Series, optionally using an executor."""
    chunks = _iter_series_chunks(text, chunk_size)

    if not chunks:
        return pd.Series([], index=text.index, name=text.name, dtype=object)

    worker = partial(
        _pre_lemmatization_clean_text_batch,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
    )

    if n_jobs == 1 or len(chunks) == 1:
        if reporter.enabled:
            cleaned_chunks = []
            for chunk_index, chunk in enumerate(chunks):
                started_at = time.perf_counter()
                cleaned_chunks.append(worker(chunk))
                reporter.chunk_done(
                    chunk_index + 1,
                    len(chunks),
                    time.perf_counter() - started_at,
                )
        else:
            cleaned_chunks = [worker(chunk) for chunk in chunks]
    else:
        max_workers = min(n_jobs, len(chunks))
        executor_class = (
            ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        )
        with executor_class(max_workers=max_workers) as executor:
            if reporter.enabled:
                futures = [
                    executor.submit(
                        _clean_chunk_with_elapsed,
                        chunk_index,
                        chunk,
                        keep_stopwords,
                        extra_stopwords,
                        stopword_backend,
                    )
                    for chunk_index, chunk in enumerate(chunks)
                ]
                cleaned_by_index = {}
                for future in as_completed(futures):
                    chunk_index, cleaned_chunk, elapsed = future.result()
                    cleaned_by_index[chunk_index] = cleaned_chunk
                    reporter.chunk_done(chunk_index + 1, len(chunks), elapsed)

                cleaned_chunks = [
                    cleaned_by_index[chunk_index] for chunk_index in range(len(chunks))
                ]
            else:
                cleaned_chunks = list(executor.map(worker, chunks))

    return pd.concat(cleaned_chunks)


def get_complete_text_clean_up_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    stopword_backend: StopwordBackend = "regex",
    use_lemmatization: bool = False,
    chunk_size: int | None = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
    verbose: bool = False,
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
        stopword_backend: Stopword removal backend. Use "regex" for the
            original pandas vectorized implementation or "flashtext" for
            trie-based keyword replacement on large stopword lists.
        use_lemmatization: Whether to apply fast lookup-based lemmatization.
            Defaults to False.
        chunk_size: Number of rows to process per chunk for pandas Series
            input. Defaults to 100,000. Use None to process the full Series at
            once.
        lemmatize_batch_size: Batch size used by get_lemmatize_text_fast.
        n_process: Number of spaCy processes used during lemmatization.
        n_jobs: Number of workers for chunked pre-lemmatization
            cleaning. Use 1 for sequential execution and -1 for all CPUs.
        parallel_backend: Executor backend for chunked cleaning. Use "thread"
            for lower overhead or "process" for CPU-bound workloads.
        verbose: Whether to print human-readable progress information.

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
    _validate_stopword_backend(stopword_backend)
    reporter = _VerboseReporter(verbose)

    if isinstance(text, str):
        reporter.start_run(
            input_type="str",
            rows=1,
            chunk_size="not used",
            chunks="not used",
            stopword_backend=stopword_backend,
            parallel_backend="not used",
            n_jobs="not used",
            use_lemmatization=use_lemmatization,
            n_process=n_process,
        )
        result = _clean_text_batch(
            text=text,
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            stopword_backend=stopword_backend,
            use_lemmatization=use_lemmatization,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
            reporter=reporter,
        )
        reporter.done(rows=1)
        return result

    if not isinstance(text, pd.Series):
        raise TypeError("text must be a string or pandas Series")

    if chunk_size is None:
        reporter.start_run(
            input_type="pandas.Series",
            rows=len(text),
            chunk_size="None (no chunking)",
            chunks=1 if len(text) else 0,
            stopword_backend=stopword_backend,
            parallel_backend="not used",
            n_jobs="not used",
            use_lemmatization=use_lemmatization,
            n_process=n_process,
        )
        result = _clean_text_batch(
            text=text,
            keep_stopwords=keep_stopwords,
            extra_stopwords=extra_stopwords,
            stopword_backend=stopword_backend,
            use_lemmatization=use_lemmatization,
            lemmatize_batch_size=lemmatize_batch_size,
            n_process=n_process,
            reporter=reporter,
        )
        reporter.done(rows=len(text))
        return result

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunk_count = _count_series_chunks(text, chunk_size)
    reporter.start_run(
        input_type="pandas.Series",
        rows=len(text),
        chunk_size=chunk_size,
        chunks=chunk_count,
        stopword_backend=stopword_backend,
        parallel_backend=(
            parallel_backend if chunk_count > 1 and resolved_n_jobs > 1 else "not used"
        ),
        n_jobs=resolved_n_jobs if chunk_count > 1 else "not used",
        use_lemmatization=use_lemmatization,
        n_process=n_process,
    )
    reporter.stage_started(1, 3, "Pre-lemmatization cleaning")
    started_at = time.perf_counter()
    result = _clean_series_chunks(
        text=text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
        chunk_size=chunk_size,
        n_jobs=resolved_n_jobs,
        parallel_backend=parallel_backend,
        reporter=reporter,
    )
    reporter.stage_done(
        1,
        3,
        "Pre-lemmatization cleaning",
        time.perf_counter() - started_at,
    )

    result = _finalize_clean_text(
        text=result,
        use_lemmatization=use_lemmatization,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        reporter=reporter,
    )
    reporter.done(rows=len(text))
    return result


def clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    stopword_backend: StopwordBackend = "regex",
    use_lemmatization: bool = False,
    chunk_size: int = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
    verbose: bool = False,
) -> pd.DataFrame:
    """Clean a text column in dataframe chunks.

    This version computes the cleaned Series first, then writes the target
    column once.

    Args:
        df: Input dataframe.
        source_col: Name of source text column.
        target_col: Name of output clean text column.
        keep_stopwords: Stopwords to keep during stopword removal.
        extra_stopwords: Additional stopwords to remove.
        stopword_backend: Stopword removal backend. Use "regex" for the
            original pandas vectorized implementation or "flashtext" for
            trie-based keyword replacement on large stopword lists.
        use_lemmatization: Whether to apply fast lookup lemmatization.
        chunk_size: Number of rows to clean per chunk.
        lemmatize_batch_size: Batch size used by spaCy nlp.pipe.
        n_process: Number of spaCy processes.
        n_jobs: Number of workers for chunked pre-lemmatization
            cleaning. Use 1 for sequential execution and -1 for all CPUs.
        parallel_backend: Executor backend for chunked cleaning. Use "thread"
            for lower overhead or "process" for CPU-bound workloads.
        verbose: Whether to print human-readable progress information.

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
        stopword_backend=stopword_backend,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )

    df[target_col] = cleaned.to_numpy(copy=False)

    return df


async def async_get_complete_text_clean_up_batch(
    text: TextInput,
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    stopword_backend: StopwordBackend = "regex",
    use_lemmatization: bool = False,
    chunk_size: int | None = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
    verbose: bool = False,
) -> TextOutput:
    """Run get_complete_text_clean_up_batch in the default executor."""
    loop = asyncio.get_running_loop()
    worker = partial(
        get_complete_text_clean_up_batch,
        text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )
    return await loop.run_in_executor(None, worker)


async def async_clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
    extra_stopwords: Collection[str] | None = None,
    stopword_backend: StopwordBackend = "regex",
    use_lemmatization: bool = False,
    chunk_size: int = 100_000,
    lemmatize_batch_size: int = 5_000,
    n_process: int = 1,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "thread",
    verbose: bool = False,
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
        stopword_backend=stopword_backend,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )
    return await loop.run_in_executor(None, worker)


clean_text = get_complete_text_clean_up_batch
async_clean_text = async_get_complete_text_clean_up_batch
