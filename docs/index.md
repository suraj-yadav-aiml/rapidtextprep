# rapidtextprep

`rapidtextprep` is a fast, reusable, pandas-friendly text preprocessing package
for NLP and machine learning workflows.

It provides utilities for:

- text cleaning and normalization
- contraction and abbreviation expansion
- stopword removal
- lookup-based lemmatization
- chunked and parallel processing
- missing value handling
- scikit-learn pipeline integration

## Basic Example

```python
from rapidtextprep import clean_text

text = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"

cleaned = clean_text(text)
print(cleaned)
```

## pandas Example

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series([
    "I CAN'T wait!!!",
    "Visit https://example.com now",
    None,
])

cleaned = clean_text(texts, handle_missing="empty")
print(cleaned)
```

## scikit-learn Example

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
```

## Documentation Sections

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [User Guide](user-guide.md)
- [Function Reference](function-reference.md)
- [scikit-learn Pipeline](sklearn-pipeline.md)
- [Parallel Processing](parallel-processing.md)
- [API Reference](api-reference.md)
