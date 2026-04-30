from __future__ import annotations

import pandas as pd

from rapidtextprep import (
    expand_abbreviations,
    expand_contractions,
    get_lower_case,
    normalize_whitespace,
    remove_accented_chars,
)


def test_normalization_accepts_string_and_series() -> None:
    assert get_lower_case("Hello WORLD") == "hello world"

    series = pd.Series(["Hello WORLD", "Pandas"])
    assert get_lower_case(series).tolist() == ["hello world", "pandas"]


def test_contraction_mapping_uses_packaged_json() -> None:
    assert expand_contractions("i'm sure he won't go") == "i am sure he will not go"


def test_abbreviation_mapping_uses_packaged_json() -> None:
    assert expand_abbreviations("btw idk irl") == (
        "by the way i do not know in real life"
    )


def test_accent_and_whitespace_normalization() -> None:
    assert remove_accented_chars("caf\xe9 d\xe9j\xe0 vu") == "cafe deja vu"
    assert normalize_whitespace("  hello    world\nthis\tis fast  ") == (
        "hello world this is fast"
    )
