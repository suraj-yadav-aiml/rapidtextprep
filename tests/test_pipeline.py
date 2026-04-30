from __future__ import annotations

import pandas as pd

from rapidtextprep import clean_text, clean_text_column_in_chunks, lemmatize_text


def test_complete_clean_text_string() -> None:
    raw = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"
    assert clean_text(raw) == "retweet user cannot believe cafe 50 percent visit"


def test_complete_clean_text_series() -> None:
    texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com now"])
    assert clean_text(texts, chunk_size=1).tolist() == ["cannot wait", "visit"]


def test_clean_text_column_in_chunks() -> None:
    df = pd.DataFrame({"text": ["I CAN'T wait!!!", "RT @user: hello"]})
    result = clean_text_column_in_chunks(df, chunk_size=1)

    assert result["clean_text"].tolist() == ["cannot wait", "retweet user hello"]


def test_lookup_lemmatization() -> None:
    assert lemmatize_text("cars were running faster") == "car be run fast"
