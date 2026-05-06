"""Reusable configuration objects for text preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from .stopwords import DEFAULT_KEEP_STOPWORDS, StopwordBackend

ParallelBackend = Literal["thread", "process"]
MissingValueStrategy = Literal["empty", "ignore", "raise"]


@dataclass(frozen=True, slots=True)
class TextPrepConfig:
    """Reusable configuration for complete text cleaning.

    Pass this object to ``clean_text`` or related pipeline helpers to avoid
    repeating the same keyword arguments across production code.
    """

    keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS
    extra_stopwords: Collection[str] | None = None
    stopword_backend: StopwordBackend = "regex"
    handle_missing: MissingValueStrategy = "empty"
    use_lemmatization: bool = False
    chunk_size: int | None = 100_000
    lemmatize_batch_size: int = 5_000
    n_process: int = 1
    n_jobs: int = 1
    parallel_backend: ParallelBackend = "thread"
    verbose: bool = False
