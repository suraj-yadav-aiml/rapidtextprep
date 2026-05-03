# rapidtextprep

Fast, reusable, pandas-friendly text preprocessing utilities for NLP and machine
learning workflows.

`rapidtextprep` provides a small public API for common text preprocessing tasks:
cleaning, normalization, stopword removal, URL/email extraction, frequency-based
word removal, feature generation, and lookup-based lemmatization. It works with
both plain Python strings and `pandas.Series` where vectorized processing makes
sense.

## Features

- Lowercasing and whitespace normalization.
- English contraction expansion.
- Social-media abbreviation expansion.
- Accent normalization.
- HTML tag, email, URL, retweet marker, and special character removal.
- Stopword counting and removal with sentiment-aware default keep words.
- URL and email extraction.
- Basic text feature generation for pandas dataframes.
- Common and rare word removal from corpus-level word counts.
- spaCy lookup-based lemmatization without requiring `en_core_web_sm` or
  `en_core_web_md`.
- Chunked processing for large pandas Series.
- Optional thread or process based parallel chunk cleaning.
- Async wrapper functions for async applications.

## Installation

Install from PyPI:

```bash
pip install rapidtextprep
```

Or with `uv`:

```bash
uv pip install rapidtextprep
```

The package declares its runtime dependencies in `pyproject.toml`, so `numpy`,
`pandas`, `scikit-learn`, `spacy`, and `spacy-lookups-data` are installed
automatically.

## Quick Start

```python
from rapidtextprep import clean_text, remove_stopwords

text = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"

cleaned = clean_text(text)
print(cleaned)

without_stopwords = remove_stopwords("this movie is not good but very emotional")
print(without_stopwords)
```

## Pandas Usage

Most cleaning and normalization functions accept a `pandas.Series` and preserve
the original index.

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(
    [
        "I CAN'T wait!!!",
        "Visit https://example.com now",
        "RT @user: hello #NLP",
    ],
    name="text",
)

cleaned = clean_text(texts)
print(cleaned)
```

## Complete Cleaning Pipeline

`clean_text` is the beginner-friendly alias for
`get_complete_text_clean_up_batch`.

```python
from rapidtextprep import clean_text

cleaned = clean_text(
    texts,
    keep_stopwords=None,
    extra_stopwords={"example"},
    use_lemmatization=False,
    chunk_size=100_000,
)
```

The pipeline order is:

1. Lowercase text.
2. Expand contractions.
3. Expand social-media abbreviations.
4. Normalize accented characters.
5. Remove HTML tags.
6. Remove email addresses.
7. Remove URLs.
8. Remove standalone retweet markers.
9. Remove special characters.
10. Remove stopwords.
11. Optionally lemmatize text.
12. Normalize whitespace.

## Parallel Processing

For large `pandas.Series` inputs, enable parallel chunk cleaning with `n_jobs`.

```python
from rapidtextprep import clean_text

cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
)
```

By default, parallel cleaning uses threads:

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="thread",
)
```

For CPU-heavy workloads, you can opt into process-based chunk cleaning:

```python
cleaned = clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)
```

Guidance:

- Use `n_jobs=1` for sequential execution.
- Use `n_jobs=-1` to use all available CPU cores.
- Use `parallel_backend="thread"` for lower overhead.
- Use `parallel_backend="process"` only after benchmarking on real data.
- On Windows, process startup and pandas chunk serialization can be expensive.

When `use_lemmatization=True`, rapidtextprep parallelizes the pre-lemmatization
cleaning stages and then runs spaCy lemmatization once over the combined Series.
This avoids sharing the cached spaCy pipeline across worker threads or processes.

## FlashText Stopword Backend

The default stopword backend uses the original regex implementation. For large
custom stopword lists or longer documents, you can opt into FlashText-based
trie matching:

```python
cleaned = clean_text(
    texts,
    stopword_backend="flashtext",
)
```

You can also use it directly:

```python
from rapidtextprep import remove_stopwords

text = remove_stopwords(
    "this movie is not good but very emotional",
    backend="flashtext",
)
```

Use `stopword_backend="regex"` for the original pandas vectorized behavior and
`stopword_backend="flashtext"` when benchmark results show that trie-based
keyword replacement is faster for your data.

## Lemmatization

Lemmatization uses spaCy's lookup lemmatizer:

```python
from rapidtextprep import lemmatize_text

lemmatized = lemmatize_text("cars were running faster")
print(lemmatized)
```

No downloadable spaCy model is required. The package uses:

```python
spacy.blank("en")
```

with lookup lemmatization powered by `spacy-lookups-data`.

You can enable lemmatization in the complete cleaning pipeline:

```python
cleaned = clean_text(
    texts,
    use_lemmatization=True,
    lemmatize_batch_size=5_000,
    n_process=1,
)
```

For spaCy's own multiprocessing during lemmatization, increase `n_process`:

```python
cleaned = clean_text(
    texts,
    use_lemmatization=True,
    n_process=2,
)
```

## Async Usage

The async functions run the synchronous implementation in the event loop's
default executor. This is useful when calling rapidtextprep from an async
application, but it does not make CPU-bound work asynchronous internally.

```python
from rapidtextprep import async_clean_text

cleaned = await async_clean_text(
    texts,
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)
```

Available async wrappers:

- `async_clean_text`
- `async_get_complete_text_clean_up_batch`
- `async_clean_text_column_in_chunks`

## Common Utilities

### Normalization

```python
from rapidtextprep import (
    expand_abbreviations,
    expand_contractions,
    lowercase_text,
    normalize_whitespace,
    remove_accented_chars,
)

lowercase_text("Hello WORLD")
expand_contractions("i'm sure he won't go")
expand_abbreviations("btw idk irl")
remove_accented_chars("cafe")
normalize_whitespace("  hello    world  ")
```

### Cleaning

```python
from rapidtextprep import (
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)

remove_email("contact test@example.com")
remove_urls("visit https://example.com now")
remove_rt("RT @user: hello")
remove_html_tags("<p>Hello</p>")
remove_special_characters("hello!!! #nlp")
```

### Extraction

```python
from rapidtextprep import get_email, get_urls

email_count, emails = get_email("mail test@example.com")
url_count, urls = get_urls("visit https://example.com")
```

### Feature Generation

```python
import pandas as pd
from rapidtextprep import get_basic_features

df = pd.DataFrame({"text": ["python is great #nlp"]})
features = get_basic_features(df, "text")
```

Generated columns:

- `char_count`
- `word_count`
- `avg_word_length`
- `stopwords_count`
- `hashtag_count`
- `mentions_count`
- `digit_count`

### Frequency-Based Cleanup

```python
import pandas as pd
from rapidtextprep import get_value_counts, remove_common_word, remove_rarewords

texts = pd.Series(["python is fast", "python is popular"])
word_counts = get_value_counts(texts)

remove_common_word("python is fast", word_counts, n_words=1)
remove_rarewords("python is popular", word_counts, n_words=1)
```

## Public API Overview

Recommended beginner-friendly names:

- `clean_text`
- `async_clean_text`
- `lemmatize_text`
- `lowercase_text`
- `expand_contractions`
- `expand_abbreviations`
- `normalize_whitespace`

Compatibility names are also preserved, including:

- `get_complete_text_clean_up_batch`
- `clean_text_column_in_chunks`
- `get_lemmatize_text_fast`
- `get_lower_case`
- `get_contraction_to_expansion`
- `get_expand_abbreviations`
- `remove_multiple_whitespaces`

## Benchmarking

A simple benchmark script is included for local testing:

```bash
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend thread
```

Compare thread and process backends:

```bash
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend thread --lemmatize
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend process --lemmatize
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend process --stopword-backend flashtext
```

Benchmark results depend heavily on text length, CPU count, operating system,
chunk size, and whether lemmatization is enabled.

## Development

Clone the repository and install dependencies:

```bash
git clone https://github.com/suraj-yadav-aiml/rapidtextprep.git
cd rapidtextprep
uv sync
```

Run formatting, linting, and tests:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Build the package:

```bash
uv build
```

## Project Structure

```text
rapidtextprep/
  src/
    rapidtextprep/
      cleaning.py
      normalization.py
      extraction.py
      features.py
      frequency.py
      lemmatization.py
      pipeline.py
      stopwords.py
      data/
  tests/
  benchmarks/
  pyproject.toml
  README.md
  LICENSE
```


## Requirements

- Python 3.11 or newer.
- `numpy`
- `pandas`
- `flashtext`
- `scikit-learn`
- `spacy`
- `spacy-lookups-data`

These dependencies are installed automatically when installing the package.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
