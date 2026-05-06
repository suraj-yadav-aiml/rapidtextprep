from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from rapidtextprep import (
    async_clean_text,
    async_clean_text_column_in_chunks,
    clean_text,
    clean_text_column_in_chunks,
    lemmatize_text,
)


def test_complete_clean_text_string() -> None:
    raw = "RT @User: I CAN'T believe this cafe is 50% OFF!!! Visit https://shop.com"
    assert clean_text(raw) == "retweet user cannot believe cafe 50 percent visit"


def test_complete_clean_text_series() -> None:
    texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com now"])
    assert clean_text(texts, chunk_size=1).tolist() == ["cannot wait", "visit"]


def test_clean_text_is_silent_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert clean_text("I CAN'T wait!!!") == "cannot wait"

    captured = capsys.readouterr()
    assert captured.out == ""


def test_verbose_clean_text_outputs_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = clean_text("I CAN'T wait!!!", verbose=True)

    captured = capsys.readouterr()
    assert result == "cannot wait"
    assert "rapidtextprep clean_text" in captured.out
    assert "Input" in captured.out
    assert "Stages" in captured.out
    assert "[1/3] Pre-lemmatization cleaning started" in captured.out
    assert "[2/3] spaCy lemmatization skipped" in captured.out
    assert "[3/3] Whitespace normalization done" in captured.out
    assert "done in 0m " in captured.out
    assert "Done" in captured.out
    assert "total_time" in captured.out
    assert " min)" in captured.out
    assert "rows_per_second" in captured.out


def test_clean_text_column_in_chunks() -> None:
    df = pd.DataFrame({"text": ["I CAN'T wait!!!", "RT @user: hello"]})
    result = clean_text_column_in_chunks(df, chunk_size=1)

    assert result["clean_text"].tolist() == ["cannot wait", "retweet user hello"]


def test_lookup_lemmatization() -> None:
    assert lemmatize_text("cars were running faster") == "car be run fast"


def test_parallel_clean_text_matches_sequential() -> None:
    texts = pd.Series(
        [
            "I CAN'T wait!!!",
            "Visit https://example.com now",
            "RT @user: hello #NLP",
            "<p>This is VERY good</p>",
            "btw idk irl",
        ],
        index=[10, 20, 30, 40, 50],
    )

    sequential = clean_text(texts, chunk_size=2, n_jobs=1)
    threaded = clean_text(texts, chunk_size=2, n_jobs=3)

    pd.testing.assert_series_equal(threaded, sequential)


def test_verbose_threaded_clean_text_matches_sequential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    texts = pd.Series(
        [
            "I CAN'T wait!!!",
            "Visit https://example.com now",
            "RT @user: hello #NLP",
            "<p>This is VERY good</p>",
        ],
        index=[10, 20, 30, 40],
    )

    sequential = clean_text(texts, chunk_size=2, n_jobs=1)
    threaded = clean_text(texts, chunk_size=2, n_jobs=2, verbose=True)

    captured = capsys.readouterr()
    pd.testing.assert_series_equal(threaded, sequential)
    assert "parallel_backend  : thread" in captured.out
    assert "chunk " in captured.out
    assert "Done" in captured.out


def test_flashtext_stopword_backend_matches_regex_pipeline() -> None:
    texts = pd.Series(
        [
            "I CAN'T wait!!!",
            "Visit https://example.com now",
            "this movie is not good but very emotional",
        ],
        index=[10, 20, 30],
    )

    regex_result = clean_text(texts, chunk_size=2, stopword_backend="regex")
    flashtext_result = clean_text(texts, chunk_size=2, stopword_backend="flashtext")

    pd.testing.assert_series_equal(flashtext_result, regex_result)


def test_process_parallel_clean_text_matches_sequential() -> None:
    texts = pd.Series(
        [
            "I CAN'T wait!!!",
            "Visit https://example.com now",
            "RT @user: hello #NLP",
            "<p>This is VERY good</p>",
            "btw idk irl",
        ],
        index=[10, 20, 30, 40, 50],
    )

    sequential = clean_text(texts, chunk_size=2, n_jobs=1)
    processed = clean_text(
        texts,
        chunk_size=2,
        n_jobs=2,
        parallel_backend="process",
    )

    pd.testing.assert_series_equal(processed, sequential)


def test_verbose_process_clean_text_matches_sequential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    texts = pd.Series(
        [
            "I CAN'T wait!!!",
            "Visit https://example.com now",
            "RT @user: hello #NLP",
            "<p>This is VERY good</p>",
        ],
        index=[10, 20, 30, 40],
    )

    sequential = clean_text(texts, chunk_size=2, n_jobs=1)
    processed = clean_text(
        texts,
        chunk_size=2,
        n_jobs=2,
        parallel_backend="process",
        verbose=True,
    )

    captured = capsys.readouterr()
    pd.testing.assert_series_equal(processed, sequential)
    assert "parallel_backend  : process" in captured.out
    assert "chunk " in captured.out
    assert "Done" in captured.out


def test_parallel_lemmatized_clean_text_matches_sequential() -> None:
    texts = pd.Series(
        [
            "cars were running faster",
            "children are eating candies",
            "dogs were barking loudly",
        ]
    )

    sequential = clean_text(
        texts,
        chunk_size=1,
        n_jobs=1,
        use_lemmatization=True,
    )
    threaded = clean_text(
        texts,
        chunk_size=1,
        n_jobs=2,
        use_lemmatization=True,
    )
    processed = clean_text(
        texts,
        chunk_size=1,
        n_jobs=2,
        use_lemmatization=True,
        parallel_backend="process",
    )

    pd.testing.assert_series_equal(threaded, sequential)
    pd.testing.assert_series_equal(processed, sequential)


def test_invalid_n_jobs_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        clean_text("hello", n_jobs=0)

    with pytest.raises(ValueError, match="n_jobs"):
        clean_text(pd.Series(["hello"]), n_jobs=-2)


def test_invalid_parallel_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="parallel_backend"):
        clean_text(pd.Series(["hello"]), parallel_backend="bad")


def test_invalid_stopword_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stopword_backend"):
        clean_text(pd.Series(["hello"]), stopword_backend="bad")


def test_async_clean_text_matches_sync() -> None:
    texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com now"])

    async_result = asyncio.run(
        async_clean_text(
            texts,
            chunk_size=1,
            n_jobs=2,
            parallel_backend="process",
        )
    )
    sync_result = clean_text(
        texts,
        chunk_size=1,
        n_jobs=2,
        parallel_backend="process",
    )

    pd.testing.assert_series_equal(async_result, sync_result)


def test_async_verbose_clean_text_outputs_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com now"])

    result = asyncio.run(
        async_clean_text(
            texts,
            chunk_size=1,
            verbose=True,
        )
    )

    captured = capsys.readouterr()
    assert result.tolist() == ["cannot wait", "visit"]
    assert "rapidtextprep clean_text" in captured.out
    assert "Done" in captured.out


def test_async_clean_text_column_in_chunks() -> None:
    df = pd.DataFrame({"text": ["I CAN'T wait!!!", "RT @user: hello"]})

    result = asyncio.run(
        async_clean_text_column_in_chunks(
            df,
            chunk_size=1,
            n_jobs=2,
            parallel_backend="process",
        )
    )

    assert result["clean_text"].tolist() == ["cannot wait", "retweet user hello"]
