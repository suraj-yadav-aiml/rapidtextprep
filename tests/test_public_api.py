from __future__ import annotations

import rapidtextprep as rtp


def test_existing_public_api_is_exported() -> None:
    expected = {
        "STOPWORDS",
        "DEFAULT_KEEP_STOPWORDS",
        "TextPrepConfig",
        "TextPreprocessor",
        "get_lower_case",
        "get_word_count",
        "get_char_count",
        "get_avg_wordlength",
        "get_stopwords_count",
        "get_hashtag_count",
        "get_mentions_count",
        "get_digit_count",
        "get_basic_features",
        "get_email",
        "get_urls",
        "get_contraction_to_expansion",
        "get_expand_abbreviations",
        "remove_accented_chars",
        "remove_email",
        "remove_urls",
        "remove_rt",
        "remove_special_characters",
        "remove_multiple_whitespaces",
        "remove_html_tags",
        "remove_stopwords",
        "get_lemmatize_text_fast",
        "get_value_counts",
        "remove_common_word",
        "remove_rarewords",
        "get_complete_text_clean_up_batch",
        "clean_text_column_in_chunks",
        "async_clean_text",
        "async_get_complete_text_clean_up_batch",
        "async_clean_text_column_in_chunks",
    }

    assert expected.issubset(set(rtp.__all__))
    for name in expected:
        assert hasattr(rtp, name)


def test_beginner_friendly_aliases_are_exported() -> None:
    assert rtp.clean_text is rtp.get_complete_text_clean_up_batch
    assert rtp.async_clean_text is rtp.async_get_complete_text_clean_up_batch
    assert rtp.lemmatize_text is rtp.get_lemmatize_text_fast
    assert rtp.lowercase_text is rtp.get_lower_case
    assert rtp.expand_contractions is rtp.get_contraction_to_expansion
    assert rtp.expand_abbreviations is rtp.get_expand_abbreviations
    assert rtp.normalize_whitespace is rtp.remove_multiple_whitespaces
