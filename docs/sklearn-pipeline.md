# scikit-learn Pipeline

`TextPreprocessor` is compatible with `sklearn.pipeline.Pipeline`.

It cleans raw text before vectorizers such as `TfidfVectorizer`.

## End-to-end Example

Assume `train.csv` contains:

- `text`
- `labels`

```python
import pandas as pd
from rapidtextprep import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


df = pd.read_csv("train.csv")

X = df["text"]
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline(
    steps=[
        (
            "text_preprocessor",
            TextPreprocessor(
                handle_missing="empty",
                use_lemmatization=False,
                stopword_backend="regex",
                chunk_size=20_000,
                n_jobs=1,
                verbose=False,
            ),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_texts = [
    "I really loved this product, it was amazing!",
    "This was a terrible experience and I hated it.",
    "The service was okay but not very impressive.",
]

predictions = model.predict(new_texts)
probabilities = model.predict_proba(new_texts)

for text, label, probs in zip(new_texts, predictions, probabilities):
    print("Text:", text)
    print("Predicted label:", label)
    print("Class probabilities:", probs)
    print()
```

## Process Backend

`parallel_backend="process"` is supported:

```python
TextPreprocessor(
    chunk_size=20_000,
    n_jobs=5,
    parallel_backend="process",
)
```

The process backend uses joblib's loky backend, which works well with
scikit-learn workflows.

## Supported Inputs

`TextPreprocessor.transform()` accepts:

- `list[str]`
- `tuple[str, ...]`
- 1D `numpy.ndarray`
- `pandas.Series`
- single-column `pandas.DataFrame`
- single string

For multi-column data, select one column before passing data to the transformer
or use scikit-learn `ColumnTransformer`.
