# Quick Start

## Clean a String

```python
from rapidtextprep import clean_text

raw = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"

cleaned = clean_text(raw)
print(cleaned)
```

## Clean a pandas Series

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

## Handle Missing Values

By default, missing values are converted to empty strings:

```python
cleaned = clean_text(texts, handle_missing="empty")
```

Preserve missing values:

```python
cleaned = clean_text(texts, handle_missing="ignore")
```

Raise an error when missing values are present:

```python
cleaned = clean_text(texts, handle_missing="raise")
```

## Reuse Settings

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

Explicit keyword arguments override config values:

```python
cleaned = clean_text(texts, config=config, verbose=False)
```
