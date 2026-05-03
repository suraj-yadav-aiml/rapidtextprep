"""Benchmark the rapidtextprep cleaning pipeline.

Run with:
    uv run python benchmarks/benchmark_pipeline.py
"""

from __future__ import annotations

import argparse
from time import perf_counter

import pandas as pd

from rapidtextprep import clean_text


def _make_sample_text(rows: int) -> pd.Series:
    """Create a reproducible text Series for pipeline benchmarking."""
    samples = [
        "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com",
        "<p>BTW idk why these cars were running faster today.</p>",
        "Email support@example.com ASAP because this isn't working!!!",
        "The children are eating candies and dogs were barking loudly.",
        "Visit www.example.org for more NLP preprocessing examples.",
    ]
    values = [samples[index % len(samples)] for index in range(rows)]
    return pd.Series(values, name="text")


def _measure(label: str, text: pd.Series, **kwargs: object) -> None:
    """Print elapsed time and throughput for one benchmark run."""
    start = perf_counter()
    clean_text(text, **kwargs)
    elapsed = perf_counter() - start
    rows_per_second = len(text) / elapsed
    print(f"{label}: {elapsed:.2f}s ({rows_per_second:,.0f} rows/s)")


def main() -> None:
    """Run the pipeline benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["thread", "process"],
        default="thread",
        help="Parallel backend used for chunked pre-lemmatization cleaning.",
    )
    parser.add_argument(
        "--stopword-backend",
        choices=["regex", "flashtext"],
        default="regex",
        help="Stopword removal backend used inside the cleaning pipeline.",
    )
    parser.add_argument("--lemmatize", action="store_true")
    args = parser.parse_args()

    text = _make_sample_text(args.rows)

    if args.lemmatize:
        clean_text(
            pd.Series(["cars were running faster"]),
            use_lemmatization=True,
            chunk_size=None,
        )

    _measure(
        "sequential",
        text,
        chunk_size=args.chunk_size,
        n_jobs=1,
        stopword_backend=args.stopword_backend,
        use_lemmatization=args.lemmatize,
    )
    _measure(
        args.backend,
        text,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        parallel_backend=args.backend,
        stopword_backend=args.stopword_backend,
        use_lemmatization=args.lemmatize,
    )


if __name__ == "__main__":
    main()
