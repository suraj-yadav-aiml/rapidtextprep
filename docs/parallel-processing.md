# Parallel Processing

Large `pandas.Series` inputs can be cleaned in chunks.

```python
from rapidtextprep import clean_text

cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
)
```

## Thread Backend

The thread backend has lower startup overhead.

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="thread",
)
```

## Process Backend

The process backend can help for CPU-heavy workloads.

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)
```

The process backend uses joblib's loky backend.

## Choosing Worker Counts

- Use `n_jobs=1` for sequential execution.
- Use `n_jobs=-1` to use all available CPU cores.
- Use `parallel_backend="thread"` for lower overhead.
- Use `parallel_backend="process"` after benchmarking on real data.

## Lemmatization

When `use_lemmatization=True`, rapidtextprep parallelizes the
pre-lemmatization cleaning stages and then runs spaCy lemmatization once over
the combined Series.

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
    use_lemmatization=True,
)
```

This avoids sharing the cached spaCy pipeline across worker processes.

## Verbose Output

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
    verbose=True,
)
```

Verbose output includes:

- input size
- chunk count
- missing value count
- backend settings
- stage timing
- rows per second
