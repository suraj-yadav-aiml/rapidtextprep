# rapidtextprep

[![PyPI version](https://img.shields.io/pypi/v/rapidtextprep.svg)](https://pypi.org/project/rapidtextprep/)
[![Python versions](https://img.shields.io/pypi/pyversions/rapidtextprep.svg)](https://pypi.org/project/rapidtextprep/)
[![License](https://img.shields.io/pypi/l/rapidtextprep.svg)](LICENSE)
[![Publish to PyPI](https://github.com/suraj-yadav-aiml/rapidtextprep/actions/workflows/publish.yml/badge.svg)](https://github.com/suraj-yadav-aiml/rapidtextprep/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://suraj-yadav-aiml.github.io/rapidtextprep/)

Fast, reusable, pandas-friendly text preprocessing utilities for NLP and machine
learning workflows.

`rapidtextprep` provides a clean public API for common text preprocessing tasks:
normalization, cleaning, stopword removal, feature extraction, lookup-based
lemmatization, large-Series processing, and scikit-learn pipeline integration.
It is designed for practical ML workflows where text cleaning should be
repeatable, readable, and easy to plug into pandas or scikit-learn.

## Features

- String and `pandas.Series` support for common text preprocessing operations.
- Lowercasing, whitespace normalization, contraction expansion, abbreviation
  expansion, and accent normalization.
- HTML tag, email, URL, retweet marker, and special character removal.
- Stopword counting and removal with sentiment-aware defaults.
- Regex and FlashText stopword backends.
- Missing value handling for real-world pandas text columns.
- Basic text feature generation for pandas dataframes.
- Corpus-level common and rare word removal.
- spaCy lookup lemmatization without `en_core_web_sm` or `en_core_web_md`.
- Chunked processing with optional thread or process based parallel cleaning.
- Async wrappers for async applications.
- scikit-learn compatible `TextPreprocessor` transformer.

## Installation

Install from PyPI:

```bash
pip install rapidtextprep
```

Or install with `uv`:

```bash
uv pip install rapidtextprep
```

Runtime dependencies are declared in `pyproject.toml` and installed
automatically.

## Quick Start

```python
from rapidtextprep import clean_text

text = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"

cleaned = clean_text(text)
print(cleaned)
```

## pandas Usage

Most cleaning and normalization utilities accept a `pandas.Series` and preserve
the original index.

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(
    [
        "I CAN'T wait!!!",
        "Visit https://example.com now",
        None,
    ],
    name="text",
)

cleaned = clean_text(texts, handle_missing="empty")
print(cleaned)
```

Use `handle_missing="ignore"` to preserve missing values, or
`handle_missing="raise"` to fail when missing values are present.

## scikit-learn Pipeline

Use `TextPreprocessor` before vectorizers in scikit-learn pipelines.

```python
from rapidtextprep import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("cleaner", TextPreprocessor(handle_missing="empty")),
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression()),
    ]
)

model.fit(texts, labels)
predictions = model.predict(new_texts)
```

## Large Data Processing

For large `pandas.Series` inputs, enable chunked processing and parallel
pre-lemmatization cleaning.

```python
from rapidtextprep import clean_text

cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)
```

Use `parallel_backend="thread"` for lower overhead and
`parallel_backend="process"` for CPU-heavy workloads after benchmarking on your
data. Use `n_jobs=-1` to use all available CPU cores.

## Lemmatization

Lemmatization uses spaCy lookup tables through `spacy.blank("en")`; no separate
spaCy model download is required.

```python
from rapidtextprep import lemmatize_text

lemmatized = lemmatize_text("cars were running faster")
print(lemmatized)
```

Enable lemmatization in the full cleaning pipeline:

```python
cleaned = clean_text(
    texts,
    use_lemmatization=True,
    lemmatize_batch_size=5_000,
    n_process=1,
)
```

## Reusable Configuration

Use `TextPrepConfig` when the same preprocessing settings are reused across
training, inference, or batch jobs.

```python
from rapidtextprep import TextPrepConfig, clean_text

config = TextPrepConfig(
    handle_missing="empty",
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)

cleaned = clean_text(texts, config=config)
```

## Documentation

| Resource | Link |
| --- | --- |
| Documentation site | [suraj-yadav-aiml.github.io/rapidtextprep](https://suraj-yadav-aiml.github.io/rapidtextprep/) |
| Function reference | [Detailed function docs](https://suraj-yadav-aiml.github.io/rapidtextprep/function-reference/) |
| scikit-learn guide | [Pipeline usage](https://suraj-yadav-aiml.github.io/rapidtextprep/sklearn-pipeline/) |
| Parallel processing | [Large-data processing](https://suraj-yadav-aiml.github.io/rapidtextprep/parallel-processing/) |
| API reference | [API docs](https://suraj-yadav-aiml.github.io/rapidtextprep/api-reference/) |
| PyPI | [pypi.org/project/rapidtextprep](https://pypi.org/project/rapidtextprep/) |
| Source code | [github.com/suraj-yadav-aiml/rapidtextprep](https://github.com/suraj-yadav-aiml/rapidtextprep) |

## Development

Clone the repository and install dependencies:

```bash
git clone https://github.com/suraj-yadav-aiml/rapidtextprep.git
cd rapidtextprep
uv sync
```

Run checks:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Build the package:

```bash
uv build
```

Build the documentation locally:

```bash
uv run mkdocs serve
```

## Requirements

- Python 3.11 or newer.
- Runtime dependencies are installed automatically from package metadata.
- No external spaCy model package is required for lemmatization.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
