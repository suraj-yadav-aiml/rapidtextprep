# scikit-learn Transformer

## `TextPreprocessor`

`TextPreprocessor` is a stateless scikit-learn transformer that runs
`rapidtextprep.clean_text` inside an ML pipeline.

It returns a 1D NumPy array of cleaned strings, which works directly with
`TfidfVectorizer`.

### Syntax

```python
TextPreprocessor(
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
| `chunk_size` | `int` or `None` | `100000` | Rows per chunk for Series cleaning. |
| `lemmatize_batch_size` | `int` | `5000` | Batch size for lemmatization. |
| `n_process` | `int` | `1` | spaCy process count for lemmatization. |
| `n_jobs` | `int` | `1` | Parallel workers for chunked pre-lemmatization cleaning. |
| `parallel_backend` | `"thread"` or `"process"` | `"thread"` | Parallel backend for chunked cleaning. |
| `verbose` | `bool` | `False` | Print progress information. |

### Accepted Inputs

`transform(X)` accepts:

- `list[str]`
- `tuple[str, ...]`
- 1D `numpy.ndarray`
- `pandas.Series`
- single-column `pandas.DataFrame`
- a single `str`

Multi-column dataframes raise `ValueError`. Select one text column or use
`ColumnTransformer`.

### Returns

A 1D `numpy.ndarray` with dtype `object`.

### Examples

Standalone transformer usage:

```python
import pandas as pd
from rapidtextprep import TextPreprocessor

texts = pd.Series(["I CAN'T wait!!!", None, "Visit https://example.com"])

cleaner = TextPreprocessor(handle_missing="empty")
cleaned = cleaner.fit_transform(texts)

print(cleaned)
```

Use different input types:

```python
import numpy as np
import pandas as pd
from rapidtextprep import TextPreprocessor

cleaner = TextPreprocessor()

print(cleaner.transform(["HELLO!!!", "I CAN'T wait"]))
print(cleaner.transform(np.array(["HELLO!!!", "I CAN'T wait"], dtype=object)))
print(cleaner.transform(pd.DataFrame({"text": ["HELLO!!!", "I CAN'T wait"]})))
print(cleaner.transform("Single text input"))
```

Use with scikit-learn `Pipeline`:

```python
import pandas as pd
from rapidtextprep import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.DataFrame(
    {
        "text": [
            "I love this product",
            "This is not good",
            "Amazing quality and fast delivery",
            "Very bad experience",
        ],
        "label": [1, 0, 1, 0],
    }
)

model = Pipeline(
    [
        ("cleaner", TextPreprocessor(handle_missing="empty")),
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression()),
    ]
)

model.fit(train["text"], train["label"])

predictions = model.predict(
    [
        "I really love this",
        "not good quality",
    ]
)

print(predictions)
```

Use process-based chunk cleaning in a pipeline:

```python
from rapidtextprep import TextPreprocessor

cleaner = TextPreprocessor(
    chunk_size=20000,
    n_jobs=5,
    parallel_backend="process",
)
```

