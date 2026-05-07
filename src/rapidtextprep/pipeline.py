"""Composable text preprocessing pipelines."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from typing import TypeVar, cast

import pandas as pd
from joblib import Parallel, delayed

from ._typing import TextInput, TextOutput
from .cleaning import (
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)
from .config import MissingValueStrategy, ParallelBackend, TextPrepConfig
from .lemmatization import get_lemmatize_text_fast
from .normalization import (
    get_contraction_to_expansion,
    get_expand_abbreviations,
    get_lower_case,
    remove_accented_chars,
    remove_multiple_whitespaces,
)
from .stopwords import StopwordBackend, remove_stopwords

_UNSET = object()
_T = TypeVar("_T")


def _config_value(value: object, config_value: _T) -> _T:
    """Return an explicit argument value or the configured value."""
    if value is _UNSET:
        return config_value

    return cast(_T, value)


def _resolve_text_prep_config(
    config: TextPrepConfig | None,
    *,
    keep_stopwords: object,
    extra_stopwords: object,
    stopword_backend: object,
    handle_missing: object,
    use_lemmatization: object,
    chunk_size: object,
    lemmatize_batch_size: object,
    n_process: object,
    n_jobs: object,
    parallel_backend: object,
    verbose: object,
) -> TextPrepConfig:
    """Merge an optional config object with explicit keyword overrides."""
    base = config or TextPrepConfig()
    return replace(
        base,
        keep_stopwords=_config_value(keep_stopwords, base.keep_stopwords),
        extra_stopwords=_config_value(extra_stopwords, base.extra_stopwords),
        stopword_backend=_config_value(stopword_backend, base.stopword_backend),
        handle_missing=_config_value(handle_missing, base.handle_missing),
        use_lemmatization=_config_value(use_lemmatization, base.use_lemmatization),
        chunk_size=_config_value(chunk_size, base.chunk_size),
        lemmatize_batch_size=_config_value(
            lemmatize_batch_size,
            base.lemmatize_batch_size,
        ),
        n_process=_config_value(n_process, base.n_process),
        n_jobs=_config_value(n_jobs, base.n_jobs),
        parallel_backend=_config_value(parallel_backend, base.parallel_backend),
        verbose=_config_value(verbose, base.verbose),
    )


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
        missing_values: int,
        stopword_backend: str,
        handle_missing: str,
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
        self._field("missing_values", missing_values)

        self._print("")
        self._print("Processing")
        self._field("stopword_backend", stopword_backend)
        self._field("handle_missing", handle_missing)
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


def _validate_handle_missing(handle_missing: MissingValueStrategy) -> None:
    """Validate the requested missing value strategy."""
    if handle_missing not in {"empty", "ignore", "raise"}:
        raise ValueError("handle_missing must be 'empty', 'ignore', or 'raise'")


def _is_missing_scalar(value: object) -> bool:
    """Return True when a scalar input is a pandas-style missing value."""
    if isinstance(value, str | pd.Series):
        return False

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _prepare_series_missing_values(
    text: pd.Series,
    handle_missing: MissingValueStrategy,
) -> tuple[pd.Series, pd.Series | None, int]:
    """Prepare Series missing values before running string operations."""
    missing_mask = text.isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return text, None, 0

    if handle_missing == "raise":
        raise ValueError(
            "text contains missing values; use handle_missing='empty' or "
            "handle_missing='ignore'"
        )

    prepared = text.fillna("")
    if handle_missing == "ignore":
        return prepared, missing_mask, missing_count

    return prepared, None, missing_count


def _restore_ignored_missing_values(
    result: pd.Series,
    original_text: pd.Series,
    missing_mask: pd.Series | None,
) -> pd.Series:
    """Restore original missing values when handle_missing='ignore'."""
    if missing_mask is None:
        return result

    restored = result.astype(object).copy()
    restored.loc[missing_mask] = original_text.loc[missing_mask]
    return restored


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
    elif parallel_backend == "thread":
        max_workers = min(n_jobs, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    else:
        max_workers = min(n_jobs, len(chunks))
        if reporter.enabled:
            results = Parallel(
                n_jobs=max_workers,
                backend="loky",
                return_as="generator_unordered",
            )(
                delayed(_clean_chunk_with_elapsed)(
                    chunk_index,
                    chunk,
                    keep_stopwords,
                    extra_stopwords,
                    stopword_backend,
                )
                for chunk_index, chunk in enumerate(chunks)
            )
            cleaned_by_index = {}
            for chunk_index, cleaned_chunk, elapsed in results:
                cleaned_by_index[chunk_index] = cleaned_chunk
                reporter.chunk_done(chunk_index + 1, len(chunks), elapsed)

            cleaned_chunks = [
                cleaned_by_index[chunk_index] for chunk_index in range(len(chunks))
            ]
        else:
            cleaned_chunks = Parallel(
                n_jobs=max_workers,
                backend="loky",
            )(
                delayed(_pre_lemmatization_clean_text_batch)(
                    chunk,
                    keep_stopwords,
                    extra_stopwords,
                    stopword_backend,
                )
                for chunk in chunks
            )

    return pd.concat(cleaned_chunks)


def get_complete_text_clean_up_batch(
    text: TextInput | None,
    keep_stopwords: Collection[str] | None | object = _UNSET,
    extra_stopwords: Collection[str] | None | object = _UNSET,
    stopword_backend: StopwordBackend | object = _UNSET,
    handle_missing: MissingValueStrategy | object = _UNSET,
    use_lemmatization: bool | object = _UNSET,
    chunk_size: int | None | object = _UNSET,
    lemmatize_batch_size: int | object = _UNSET,
    n_process: int | object = _UNSET,
    n_jobs: int | object = _UNSET,
    parallel_backend: ParallelBackend | object = _UNSET,
    verbose: bool | object = _UNSET,
    config: TextPrepConfig | None = None,
) -> TextOutput | None:
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
        text: A single text string or pandas Series.
        keep_stopwords: Stopwords to keep during stopword removal. Defaults to
            DEFAULT_KEEP_STOPWORDS. Use None to remove all stopwords.
        extra_stopwords: Additional words to remove along with STOPWORDS.
        stopword_backend: Stopword removal backend. Use "regex" for the
            original pandas vectorized implementation or "flashtext" for
            trie-based keyword replacement on large stopword lists.
        handle_missing: Missing value strategy. Use "empty" to convert missing
            values to empty strings, "ignore" to preserve missing values in the
            output, or "raise" to fail when missing values are present.
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
        config: Optional reusable configuration. Explicit keyword arguments
            override matching values from this config.

    Returns:
        Fully cleaned text. If input is a string, returns a string. If input is
        a pandas Series, returns a pandas Series with the same index.

    Example:
        >>> raw = "RT @User: I CAN'T believe this caf\xe9 is 50% OFF!!! Visit https://shop.com"
        >>> get_complete_text_clean_up_batch(raw)
        'not believe cafe percent'
    """
    settings = _resolve_text_prep_config(
        config,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
        handle_missing=handle_missing,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )
    resolved_n_jobs = _resolve_n_jobs(settings.n_jobs)
    _validate_parallel_backend(settings.parallel_backend)
    _validate_stopword_backend(settings.stopword_backend)
    _validate_handle_missing(settings.handle_missing)
    reporter = _VerboseReporter(settings.verbose)

    scalar_missing_count = 0
    if _is_missing_scalar(text):
        if settings.handle_missing == "raise":
            raise ValueError(
                "text is missing; use handle_missing='empty' or handle_missing='ignore'"
            )

        if settings.handle_missing == "ignore":
            return text

        text = ""
        scalar_missing_count = 1

    if isinstance(text, str):
        reporter.start_run(
            input_type="str",
            rows=1,
            chunk_size="not used",
            chunks="not used",
            missing_values=scalar_missing_count,
            stopword_backend=settings.stopword_backend,
            handle_missing=settings.handle_missing,
            parallel_backend="not used",
            n_jobs="not used",
            use_lemmatization=settings.use_lemmatization,
            n_process=settings.n_process,
        )
        result = _clean_text_batch(
            text=text,
            keep_stopwords=settings.keep_stopwords,
            extra_stopwords=settings.extra_stopwords,
            stopword_backend=settings.stopword_backend,
            use_lemmatization=settings.use_lemmatization,
            lemmatize_batch_size=settings.lemmatize_batch_size,
            n_process=settings.n_process,
            reporter=reporter,
        )
        reporter.done(rows=1)
        return result

    if not isinstance(text, pd.Series):
        raise TypeError("text must be a string or pandas Series")

    original_text = text
    text, missing_mask, missing_count = _prepare_series_missing_values(
        text,
        settings.handle_missing,
    )

    if settings.chunk_size is None:
        reporter.start_run(
            input_type="pandas.Series",
            rows=len(text),
            chunk_size="None (no chunking)",
            chunks=1 if len(text) else 0,
            missing_values=missing_count,
            stopword_backend=settings.stopword_backend,
            handle_missing=settings.handle_missing,
            parallel_backend="not used",
            n_jobs="not used",
            use_lemmatization=settings.use_lemmatization,
            n_process=settings.n_process,
        )
        result = _clean_text_batch(
            text=text,
            keep_stopwords=settings.keep_stopwords,
            extra_stopwords=settings.extra_stopwords,
            stopword_backend=settings.stopword_backend,
            use_lemmatization=settings.use_lemmatization,
            lemmatize_batch_size=settings.lemmatize_batch_size,
            n_process=settings.n_process,
            reporter=reporter,
        )
        result = _restore_ignored_missing_values(
            result,
            original_text=original_text,
            missing_mask=missing_mask,
        )
        reporter.done(rows=len(text))
        return result

    if settings.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunk_count = _count_series_chunks(text, settings.chunk_size)
    reporter.start_run(
        input_type="pandas.Series",
        rows=len(text),
        chunk_size=settings.chunk_size,
        chunks=chunk_count,
        missing_values=missing_count,
        stopword_backend=settings.stopword_backend,
        handle_missing=settings.handle_missing,
        parallel_backend=(
            settings.parallel_backend
            if chunk_count > 1 and resolved_n_jobs > 1
            else "not used"
        ),
        n_jobs=resolved_n_jobs if chunk_count > 1 else "not used",
        use_lemmatization=settings.use_lemmatization,
        n_process=settings.n_process,
    )
    reporter.stage_started(1, 3, "Pre-lemmatization cleaning")
    started_at = time.perf_counter()
    result = _clean_series_chunks(
        text=text,
        keep_stopwords=settings.keep_stopwords,
        extra_stopwords=settings.extra_stopwords,
        stopword_backend=settings.stopword_backend,
        chunk_size=settings.chunk_size,
        n_jobs=resolved_n_jobs,
        parallel_backend=settings.parallel_backend,
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
        use_lemmatization=settings.use_lemmatization,
        lemmatize_batch_size=settings.lemmatize_batch_size,
        n_process=settings.n_process,
        reporter=reporter,
    )
    result = _restore_ignored_missing_values(
        result,
        original_text=original_text,
        missing_mask=missing_mask,
    )
    reporter.done(rows=len(text))
    return result


def clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None | object = _UNSET,
    extra_stopwords: Collection[str] | None | object = _UNSET,
    stopword_backend: StopwordBackend | object = _UNSET,
    handle_missing: MissingValueStrategy | object = _UNSET,
    use_lemmatization: bool | object = _UNSET,
    chunk_size: int | object = _UNSET,
    lemmatize_batch_size: int | object = _UNSET,
    n_process: int | object = _UNSET,
    n_jobs: int | object = _UNSET,
    parallel_backend: ParallelBackend | object = _UNSET,
    verbose: bool | object = _UNSET,
    config: TextPrepConfig | None = None,
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
        handle_missing: Missing value strategy. Use "empty" to convert missing
            values to empty strings, "ignore" to preserve missing values in the
            output, or "raise" to fail when missing values are present.
        use_lemmatization: Whether to apply fast lookup lemmatization.
        chunk_size: Number of rows to clean per chunk.
        lemmatize_batch_size: Batch size used by spaCy nlp.pipe.
        n_process: Number of spaCy processes.
        n_jobs: Number of workers for chunked pre-lemmatization
            cleaning. Use 1 for sequential execution and -1 for all CPUs.
        parallel_backend: Executor backend for chunked cleaning. Use "thread"
            for lower overhead or "process" for CPU-bound workloads.
        verbose: Whether to print human-readable progress information.
        config: Optional reusable configuration. Explicit keyword arguments
            override matching values from this config.

    Returns:
        The dataframe with the cleaned text column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if source_col not in df.columns:
        raise KeyError(f"{source_col!r} column not found in dataframe")

    settings = _resolve_text_prep_config(
        config,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
        handle_missing=handle_missing,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )

    if settings.chunk_size is None:
        raise ValueError("chunk_size must be greater than 0")

    if settings.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    cleaned = get_complete_text_clean_up_batch(
        df[source_col],
        config=settings,
    )

    df[target_col] = cleaned.to_numpy(copy=False)

    return df


async def async_get_complete_text_clean_up_batch(
    text: TextInput | None,
    keep_stopwords: Collection[str] | None | object = _UNSET,
    extra_stopwords: Collection[str] | None | object = _UNSET,
    stopword_backend: StopwordBackend | object = _UNSET,
    handle_missing: MissingValueStrategy | object = _UNSET,
    use_lemmatization: bool | object = _UNSET,
    chunk_size: int | None | object = _UNSET,
    lemmatize_batch_size: int | object = _UNSET,
    n_process: int | object = _UNSET,
    n_jobs: int | object = _UNSET,
    parallel_backend: ParallelBackend | object = _UNSET,
    verbose: bool | object = _UNSET,
    config: TextPrepConfig | None = None,
) -> TextOutput | None:
    """Run get_complete_text_clean_up_batch in the default executor."""
    loop = asyncio.get_running_loop()
    worker = partial(
        get_complete_text_clean_up_batch,
        text,
        keep_stopwords=keep_stopwords,
        extra_stopwords=extra_stopwords,
        stopword_backend=stopword_backend,
        handle_missing=handle_missing,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
        config=config,
    )
    return await loop.run_in_executor(None, worker)


async def async_clean_text_column_in_chunks(
    df: pd.DataFrame,
    source_col: str = "text",
    target_col: str = "clean_text",
    keep_stopwords: Collection[str] | None | object = _UNSET,
    extra_stopwords: Collection[str] | None | object = _UNSET,
    stopword_backend: StopwordBackend | object = _UNSET,
    handle_missing: MissingValueStrategy | object = _UNSET,
    use_lemmatization: bool | object = _UNSET,
    chunk_size: int | object = _UNSET,
    lemmatize_batch_size: int | object = _UNSET,
    n_process: int | object = _UNSET,
    n_jobs: int | object = _UNSET,
    parallel_backend: ParallelBackend | object = _UNSET,
    verbose: bool | object = _UNSET,
    config: TextPrepConfig | None = None,
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
        handle_missing=handle_missing,
        use_lemmatization=use_lemmatization,
        chunk_size=chunk_size,
        lemmatize_batch_size=lemmatize_batch_size,
        n_process=n_process,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=verbose,
        config=config,
    )
    return await loop.run_in_executor(None, worker)


clean_text = get_complete_text_clean_up_batch
async_clean_text = async_get_complete_text_clean_up_batch
