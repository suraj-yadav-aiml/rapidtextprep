from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rapidtextprep import TextPreprocessor, clean_text


def test_text_preprocessor_fit_returns_self() -> None:
    transformer = TextPreprocessor()

    assert transformer.fit(["hello"]) is transformer


def test_text_preprocessor_transforms_list() -> None:
    transformer = TextPreprocessor()

    result = transformer.transform(["I CAN'T wait!!!", "Visit https://example.com"])

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["cannot wait", "visit"]


def test_text_preprocessor_transforms_series() -> None:
    text = pd.Series(["I CAN'T wait!!!", "Visit https://example.com"])
    transformer = TextPreprocessor(chunk_size=1)

    result = transformer.transform(text)
    expected = clean_text(text, chunk_size=1).to_numpy(dtype=object)

    np.testing.assert_array_equal(result, expected)


def test_text_preprocessor_transforms_single_column_dataframe() -> None:
    df = pd.DataFrame({"text": ["I CAN'T wait!!!", "Visit https://example.com"]})
    transformer = TextPreprocessor(chunk_size=1)

    assert transformer.transform(df).tolist() == ["cannot wait", "visit"]


def test_text_preprocessor_transforms_numpy_array() -> None:
    text = np.array(["I CAN'T wait!!!", "Visit https://example.com"], dtype=object)
    transformer = TextPreprocessor()

    assert transformer.transform(text).tolist() == ["cannot wait", "visit"]


def test_text_preprocessor_transforms_single_string() -> None:
    transformer = TextPreprocessor()

    assert transformer.transform("I CAN'T wait!!!").tolist() == ["cannot wait"]


def test_text_preprocessor_handles_missing_values() -> None:
    transformer = TextPreprocessor(handle_missing="empty")

    assert transformer.transform(["I CAN'T wait!!!", None]).tolist() == [
        "cannot wait",
        "",
    ]


def test_text_preprocessor_set_params_updates_behavior() -> None:
    transformer = TextPreprocessor(handle_missing="raise")
    transformer.set_params(handle_missing="empty")

    assert transformer.transform(["I CAN'T wait!!!", None]).tolist() == [
        "cannot wait",
        "",
    ]


def test_text_preprocessor_get_feature_names_out() -> None:
    transformer = TextPreprocessor()

    assert transformer.get_feature_names_out().tolist() == ["clean_text"]


def test_text_preprocessor_works_inside_sklearn_pipeline() -> None:
    texts = [
        "good useful movie",
        "bad boring movie",
        "excellent useful story",
        "terrible boring story",
    ]
    labels = [1, 0, 1, 0]

    model = Pipeline(
        [
            ("cleaner", TextPreprocessor()),
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression()),
        ]
    )

    model.fit(texts, labels)
    predictions = model.predict(["good useful story", "bad boring story"])

    assert predictions.tolist() == [1, 0]


def test_text_preprocessor_process_backend_works_inside_sklearn_pipeline() -> None:
    texts = [
        "good useful movie",
        "bad boring movie",
        "excellent useful story",
        "terrible boring story",
    ]
    labels = [1, 0, 1, 0]

    model = Pipeline(
        [
            (
                "cleaner",
                TextPreprocessor(
                    chunk_size=2,
                    n_jobs=2,
                    parallel_backend="process",
                ),
            ),
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression()),
        ]
    )

    model.fit(texts, labels)
    predictions = model.predict(["good useful story", "bad boring story"])

    assert predictions.tolist() == [1, 0]


def test_text_preprocessor_rejects_multi_column_dataframe() -> None:
    transformer = TextPreprocessor()
    df = pd.DataFrame(
        {
            "title": ["hello"],
            "body": ["world"],
        }
    )

    with pytest.raises(ValueError, match="single-column DataFrame"):
        transformer.transform(df)


def test_text_preprocessor_rejects_2d_numpy_array() -> None:
    transformer = TextPreprocessor()

    with pytest.raises(ValueError, match="1D numpy array"):
        transformer.transform(np.array([["hello"], ["world"]], dtype=object))
