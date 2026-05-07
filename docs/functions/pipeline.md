# Pipeline Functions

Pipeline functions combine multiple low-level utilities into complete text
cleaning workflows.

## `clean_text`

Beginner-friendly alias for `get_complete_text_clean_up_batch`.

## `get_complete_text_clean_up_batch`

Runs the complete cleaning pipeline on a single string or `pandas.Series`.

Pipeline order:

1. Lowercase text.
2. Expand contractions.
3. Expand social-media abbreviations.
4. Remove accented characters.
5. Remove HTML tags.
6. Remove email addresses.
7. Remove URLs.
8. Remove standalone retweet markers.
9. Remove special characters.
10. Remove stopwords.
11. Optionally lemmatize.
12. Normalize whitespace.

### Syntax

```python
clean_text(
    text,
    keep_stopwords=DEFAULT_KEEP_STOPWORDS,
    extra_stopwords=None,
    stopword_backend="regex",
    handle_missing="empty",
    use_lemmatization=False,
    chunk_size=100_000,
    lemmatize_batch_size=5_000,
    n_process=1,
    n_jobs=1,
    parallel_backend="thread",
    verbose=False,
    config=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str`, `pandas.Series`, or `None` | Required | Text to clean. |
| `keep_stopwords` | `Collection[str]` or `None` | `DEFAULT_KEEP_STOPWORDS` | Stopwords to keep. Use `None` to remove all stopwords. |
| `extra_stopwords` | `Collection[str]` or `None` | `None` | Additional words to remove. |
| `stopword_backend` | `"regex"` or `"flashtext"` | `"regex"` | Backend for stopword removal. |
| `handle_missing` | `"empty"`, `"ignore"`, or `"raise"` | `"empty"` | How missing values are handled. |
| `use_lemmatization` | `bool` | `False` | Enable spaCy lookup lemmatization. |
| `chunk_size` | `int` or `None` | `100000` | Rows per chunk for Series input. Use `None` for full-Series cleaning. |
| `lemmatize_batch_size` | `int` | `5000` | Batch size used by spaCy lemmatization. |
| `n_process` | `int` | `1` | Number of spaCy processes used during lemmatization. |
| `n_jobs` | `int` | `1` | Number of workers for chunked pre-lemmatization cleaning. Use `-1` for all CPUs. |
| `parallel_backend` | `"thread"` or `"process"` | `"thread"` | Parallel backend for chunked cleaning. |
| `verbose` | `bool` | `False` | Print readable stage progress. |
| `config` | `TextPrepConfig` or `None` | `None` | Reusable configuration. Explicit keyword arguments override config values. |

### Returns

Returns the same shape as the input:

- `str` input returns `str`
- `pandas.Series` input returns `pandas.Series` with the same index
- missing scalar input with `handle_missing="empty"` returns an empty string
- missing scalar input with `handle_missing="ignore"` returns the original missing value

### Examples

Clean a single string:

```python
from rapidtextprep import clean_text

text = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"
print(clean_text(text))
```

Clean a Series:

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(
    [
        "I CAN'T wait!!!",
        "Contact test@example.com",
        "Visit https://example.com now",
    ],
    name="text",
)

cleaned = clean_text(texts)
print(cleaned.tolist())
```

Handle missing values:

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(["I CAN'T wait", None, "Visit https://example.com"])

print(clean_text(texts, handle_missing="empty").tolist())
# ['cannot wait', '', 'visit']

print(clean_text(texts, handle_missing="ignore").tolist())
# ['cannot wait', nan, 'visit']
```

Customize stopword behavior:

```python
from rapidtextprep import clean_text

text = "this movie is not good but very emotional"

print(clean_text(text))
# movie not good but very emotional

print(clean_text(text, keep_stopwords=None))
# movie good emotional

print(clean_text(text, extra_stopwords={"movie", "emotional"}))
# not good but very
```

Use the FlashText stopword backend:

```python
from rapidtextprep import clean_text

text = "this movie is not good but very emotional"

cleaned = clean_text(
    text,
    stopword_backend="flashtext",
)
print(cleaned)
```

Enable lemmatization:

```python
from rapidtextprep import clean_text

text = "cars were running faster"

cleaned = clean_text(
    text,
    use_lemmatization=True,
    lemmatize_batch_size=5000,
    n_process=1,
)
print(cleaned)
# car run fast
```

Use chunked parallel processing for large Series:

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(["I CAN'T believe this is SO good!!!"] * 100000)

cleaned = clean_text(
    texts,
    chunk_size=20000,
    n_jobs=5,
    parallel_backend="process",
)
```

Show progress:

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(["I CAN'T believe this is SO good!!!"] * 100000)

cleaned = clean_text(
    texts,
    chunk_size=20000,
    n_jobs=5,
    parallel_backend="process",
    verbose=True,
)
```

Use a reusable config:

```python
import pandas as pd
from rapidtextprep import TextPrepConfig, clean_text

texts = pd.Series(["I CAN'T wait", None, "Visit https://example.com"])

config = TextPrepConfig(
    handle_missing="empty",
    chunk_size=20000,
    n_jobs=5,
    parallel_backend="process",
)

cleaned = clean_text(texts, config=config)
```

## `clean_text_column_in_chunks`

Cleans one dataframe column and writes the result to a target column.

### Syntax

```python
clean_text_column_in_chunks(
    df,
    source_col="text",
    target_col="clean_text",
    keep_stopwords=DEFAULT_KEEP_STOPWORDS,
    extra_stopwords=None,
    stopword_backend="regex",
    handle_missing="empty",
    use_lemmatization=False,
    chunk_size=100_000,
    lemmatize_batch_size=5_000,
    n_process=1,
    n_jobs=1,
    parallel_backend="thread",
    verbose=False,
    config=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pandas.DataFrame` | Required | Dataframe containing the source text column. |
| `source_col` | `str` | `"text"` | Column to clean. |
| `target_col` | `str` | `"clean_text"` | Column where cleaned text is written. |
| Other parameters | Same as `clean_text` | Same as `clean_text` | Cleaning configuration passed to the pipeline. |

### Returns

The same dataframe object with `target_col` added or overwritten.

### Examples

```python
import pandas as pd
from rapidtextprep import clean_text_column_in_chunks

df = pd.DataFrame(
    {
        "review": [
            "I CAN'T believe this is good!!!",
            "Visit https://example.com",
        ]
    }
)

result = clean_text_column_in_chunks(
    df,
    source_col="review",
    target_col="clean_review",
    chunk_size=1000,
)

print(result[["review", "clean_review"]])
```

Use parallel processing:

```python
result = clean_text_column_in_chunks(
    df,
    source_col="review",
    target_col="clean_review",
    chunk_size=20000,
    n_jobs=5,
    parallel_backend="process",
)
```

## Async Pipeline Wrappers

Async wrappers run the synchronous pipeline in an executor. They are useful when
your application is already async, such as a web API.

Available wrappers:

- `async_clean_text`
- `async_get_complete_text_clean_up_batch`
- `async_clean_text_column_in_chunks`

### Syntax

```python
await async_clean_text(text, ...)
await async_get_complete_text_clean_up_batch(text, ...)
await async_clean_text_column_in_chunks(df, ...)
```

### Parameters

The async wrappers accept the same parameters as their synchronous equivalents.

### Returns

The same return type as the synchronous function.

### Examples

```python
import asyncio
import pandas as pd
from rapidtextprep import async_clean_text


async def main():
    texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com"])

    cleaned = await async_clean_text(
        texts,
        chunk_size=20000,
        n_jobs=2,
        parallel_backend="thread",
    )
    print(cleaned.tolist())


asyncio.run(main())
```
