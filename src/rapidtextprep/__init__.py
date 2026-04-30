"""Fast, pandas-friendly text preprocessing utilities."""

from __future__ import annotations

from .cleaning import (
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)
from .extraction import get_email, get_urls
from .features import (
    get_avg_wordlength,
    get_basic_features,
    get_char_count,
    get_digit_count,
    get_hashtag_count,
    get_mentions_count,
    get_word_count,
)
from .frequency import get_value_counts, remove_common_word, remove_rarewords
from .lemmatization import get_lemmatize_text_fast, lemmatize_text
from .normalization import (
    expand_abbreviations,
    expand_contractions,
    get_contraction_to_expansion,
    get_expand_abbreviations,
    get_lower_case,
    lowercase_text,
    normalize_whitespace,
    remove_accented_chars,
    remove_multiple_whitespaces,
)
from .pipeline import (
    clean_text,
    clean_text_column_in_chunks,
    get_complete_text_clean_up_batch,
)
from .stopwords import (
    DEFAULT_KEEP_STOPWORDS,
    STOPWORDS,
    get_stopwords_count,
    remove_stopwords,
)

__all__ = [
    "STOPWORDS",
    "DEFAULT_KEEP_STOPWORDS",
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
    "clean_text",
    "lemmatize_text",
    "lowercase_text",
    "expand_contractions",
    "expand_abbreviations",
    "normalize_whitespace",
]
