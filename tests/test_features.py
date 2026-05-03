from __future__ import annotations

import pandas as pd

from rapidtextprep import (
    get_basic_features,
    get_char_count,
    get_digit_count,
    get_hashtag_count,
    get_mentions_count,
    get_stopwords_count,
    get_value_counts,
    get_word_count,
    remove_common_word,
    remove_rarewords,
    remove_stopwords,
)


def test_feature_counts() -> None:
    text = pd.Series(["this is #Python 2026", "thanks @user"])

    assert get_word_count(text).tolist() == [4, 2]
    assert get_char_count(text).tolist() == [17, 11]
    assert get_stopwords_count(text).tolist() == [2, 0]
    assert get_hashtag_count(text).tolist() == [1, 0]
    assert get_mentions_count(text).tolist() == [0, 1]
    assert get_digit_count(text).tolist() == [1, 0]


def test_basic_features_dataframe() -> None:
    df = pd.DataFrame({"text": ["python is great #nlp"]})
    features = get_basic_features(df, "text")

    assert features.columns.tolist() == [
        "char_count",
        "word_count",
        "avg_word_length",
        "stopwords_count",
        "hashtag_count",
        "mentions_count",
        "digit_count",
    ]
    assert features.loc[0, "word_count"] == 4


def test_stopword_and_frequency_removal() -> None:
    assert remove_stopwords("this movie is not good but very emotional") == (
        "  movie   not good but very emotional"
    )
    assert remove_stopwords(
        "this movie is not good but very emotional",
        backend="flashtext",
    ) == ("  movie   not good but very emotional")
    assert (
        remove_stopwords(
            "can't can",
            extra_stopwords={"can"},
            backend="flashtext",
        )
        == "can't  "
    )

    text = pd.Series(["python is fast", "python is popular"])
    counts = get_value_counts(text)

    assert counts.to_dict() == {"python": 2, "is": 2, "fast": 1, "popular": 1}
    assert remove_common_word("python is fast", counts, 1) == "  is fast"
    assert remove_rarewords("python is popular", counts, 1) == "python is  "
