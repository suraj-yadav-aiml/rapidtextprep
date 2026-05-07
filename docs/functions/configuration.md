# Configuration Object

## `TextPrepConfig`

`TextPrepConfig` stores reusable settings for `clean_text`,
`get_complete_text_clean_up_batch`, `clean_text_column_in_chunks`, and async
pipeline wrappers.

It is useful in production code where the same cleaning parameters are reused
in multiple places.

### Syntax

```python
TextPrepConfig(
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
)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `keep_stopwords` | `Collection[str]` or `None` | `DEFAULT_KEEP_STOPWORDS` | Stopwords to preserve. |
| `extra_stopwords` | `Collection[str]` or `None` | `None` | Extra words to remove. |
| `stopword_backend` | `"regex"` or `"flashtext"` | `"regex"` | Stopword removal backend. |
| `handle_missing` | `"empty"`, `"ignore"`, or `"raise"` | `"empty"` | Missing value strategy. |
| `use_lemmatization` | `bool` | `False` | Enable spaCy lookup lemmatization. |
| `chunk_size` | `int` or `None` | `100000` | Rows per chunk for Series cleaning. Use `None` to disable chunking. |
| `lemmatize_batch_size` | `int` | `5000` | Batch size for lemmatization. |
| `n_process` | `int` | `1` | spaCy process count for lemmatization. |
| `n_jobs` | `int` | `1` | Parallel workers for chunked pre-lemmatization cleaning. Use `-1` for all CPUs. |
| `parallel_backend` | `"thread"` or `"process"` | `"thread"` | Parallel backend for chunked cleaning. |
| `verbose` | `bool` | `False` | Print progress information. |

### Returns

A frozen dataclass instance containing cleaning settings.

### Examples

Create a reusable config:

```python
import pandas as pd
from rapidtextprep import TextPrepConfig, clean_text

texts = pd.Series(["I CAN'T wait!!!", None, "Visit https://example.com"])

config = TextPrepConfig(
    handle_missing="empty",
    keep_stopwords=None,
    chunk_size=20000,
    n_jobs=4,
    parallel_backend="process",
)

cleaned = clean_text(texts, config=config)
print(cleaned.tolist())
```

Override one value at call time:

```python
from rapidtextprep import TextPrepConfig, clean_text

config = TextPrepConfig(
    handle_missing="empty",
    use_lemmatization=True,
    verbose=True,
)

cleaned = clean_text(
    "cars were running faster",
    config=config,
    verbose=False,
)
print(cleaned)
```

