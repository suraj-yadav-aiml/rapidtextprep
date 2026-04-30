"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ._patterns import MULTIPLE_WHITESPACE_PATTERN
from ._resources import (
    CONTRACTION_TO_EXPANSION,
    SYMBOL_TRANSLATION,
    TOKEN_ABBREVIATION_TO_EXPANSION,
)
from ._typing import TextInput, TextOutput

APOSTROPHE_TRANSLATION: dict[int, str] = str.maketrans(
    {
        "\u2019": "'",
        "\u2018": "'",
        "\u201b": "'",
        "`": "'",
        "\xb4": "'",
    }
)

PAST_PARTICIPLES: str = (
    "been|done|gone|seen|known|made|taken|given|found|come|become|left|"
    "written|said|told|heard|felt|kept|lost|met|paid|put|read|run|"
    "thought|bought|brought|caught|taught|eaten|drunk|got|had|won|"
    "shown|worn|broken|chosen|driven|fallen|forgotten|grown|spoken|"
    "stolen|thrown|begun"
)

S_HAS_RE: re.Pattern[str] = re.compile(
    rf"\b("
    rf"he|she|it|who|what|that|there|where|when|why|how|which|"
    rf"someone|somebody|everyone|everybody|anyone|anybody|"
    rf"no one|nobody|something|anything|nothing|everything"
    rf")'s\s+({PAST_PARTICIPLES})\b"
)

D_HAD_RE: re.Pattern[str] = re.compile(
    rf"\b("
    rf"i|you|he|she|it|we|they|who|what|that|there|where|when|why|how|which"
    rf")'d\s+({PAST_PARTICIPLES})\b"
)

CONTRACTION_RE: re.Pattern[str] = re.compile(
    r"(?<![\w'])"
    r"(?:"
    + "|".join(
        re.escape(contraction)
        for contraction in sorted(CONTRACTION_TO_EXPANSION, key=len, reverse=True)
    )
    + r")"
    r"(?![\w'])"
)


def _replace_contraction(match: re.Match[str]) -> str:
    """Return expansion for a matched contraction."""
    return CONTRACTION_TO_EXPANSION[match.group(0)]


ABBREVIATION_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])"
    r"(?:"
    + "|".join(
        re.escape(abbreviation)
        for abbreviation in sorted(
            TOKEN_ABBREVIATION_TO_EXPANSION,
            key=len,
            reverse=True,
        )
    )
    + r")"
    r"(?![a-z0-9])"
)


def _replace_abbreviation(match: re.Match[str]) -> str:
    """Return expansion for a matched abbreviation."""
    return TOKEN_ABBREVIATION_TO_EXPANSION[match.group(0)]


ACCENTED_CHAR_TRANSLATION: dict[int, str] = str.maketrans(
    {
        "\xe6": "ae",
        "\xc6": "AE",
        "\u0153": "oe",
        "\u0152": "OE",
        "\xf8": "o",
        "\xd8": "O",
        "\u0111": "d",
        "\u0110": "D",
        "\xf0": "d",
        "\xd0": "D",
        "\xfe": "th",
        "\xde": "TH",
        "\u0142": "l",
        "\u0141": "L",
        "\xdf": "ss",
        "\u1e9e": "SS",
    }
)


def get_lower_case(text: TextInput) -> TextOutput:
    """Convert text to lowercase.

    Args:
        text: A single text string or a pandas Series containing non-null text.

    Returns:
        Lowercased text. If input is a string, returns a string. If input is a
        pandas Series, returns a pandas Series.

    Example:
        >>> get_lower_case("Hello WORLD")
        'hello world'
    """
    if isinstance(text, str):
        return text.lower()

    if isinstance(text, pd.Series):
        return text.str.lower()

    raise TypeError("text must be a string or pandas Series")


def get_contraction_to_expansion(text: TextInput) -> TextOutput:
    """Expand English contractions in a string or pandas Series.

    This function assumes text is already lowercase.

    Args:
        text: A single lowercase string or pandas Series.

    Returns:
        Text with contractions expanded.

    Example:
        >>> get_contraction_to_expansion("he's been working")
        'he has been working'
    """
    if isinstance(text, str):
        result = text.translate(APOSTROPHE_TRANSLATION)
        result = S_HAS_RE.sub(r"\1 has \2", result)
        result = D_HAD_RE.sub(r"\1 had \2", result)
        return CONTRACTION_RE.sub(_replace_contraction, result)

    if isinstance(text, pd.Series):
        result = text.str.translate(APOSTROPHE_TRANSLATION)
        result = result.str.replace(S_HAS_RE, r"\1 has \2", regex=True)
        result = result.str.replace(D_HAD_RE, r"\1 had \2", regex=True)
        return result.str.replace(
            CONTRACTION_RE,
            _replace_contraction,
            regex=True,
        )

    raise TypeError("text must be a string or pandas Series")


def get_expand_abbreviations(text: TextInput) -> TextOutput:
    """Expand social-media and texting abbreviations.

    This function assumes text is already lowercase. Whitespace normalization is
    intentionally not handled here.

    Args:
        text: A single lowercase string or pandas Series.

    Returns:
        Text with abbreviations expanded.

    Example:
        >>> get_expand_abbreviations("btw idk irl")
        'by the way i do not know in real life'
    """
    if isinstance(text, str):
        result = text.translate(SYMBOL_TRANSLATION)
        return ABBREVIATION_PATTERN.sub(_replace_abbreviation, result)

    if isinstance(text, pd.Series):
        result = text.str.translate(SYMBOL_TRANSLATION)
        return result.str.replace(
            ABBREVIATION_PATTERN,
            _replace_abbreviation,
            regex=True,
        )

    raise TypeError("text must be a string or pandas Series")


def remove_accented_chars(text: TextInput) -> TextOutput:
    """Remove accented characters from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with accented characters replaced by ASCII equivalents.

    Example:
        >>> remove_accented_chars("caf\xe9 d\xe9j\xe0 vu fa\xe7ade \xfcber")
        'cafe deja vu facade uber'
    """
    if isinstance(text, str):
        normalized_text = text.translate(ACCENTED_CHAR_TRANSLATION)
        return (
            unicodedata.normalize("NFKD", normalized_text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    if isinstance(text, pd.Series):
        return (
            text.str.translate(ACCENTED_CHAR_TRANSLATION)
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )

    raise TypeError("text must be a string or pandas Series")


def remove_multiple_whitespaces(text: TextInput) -> TextOutput:
    """Collapse multiple consecutive whitespaces into a single space.

    Args:
        text: A single text string or pandas Series.

    Returns:
        Text with normalized whitespace.

    Example:
        >>> remove_multiple_whitespaces("  hello    world\\nthis\\tis fast  ")
        'hello world this is fast'
    """
    if isinstance(text, str):
        return MULTIPLE_WHITESPACE_PATTERN.sub(" ", text).strip()

    if isinstance(text, pd.Series):
        return text.str.replace(
            MULTIPLE_WHITESPACE_PATTERN,
            " ",
            regex=True,
        ).str.strip()

    raise TypeError("text must be a string or pandas Series")


lowercase_text = get_lower_case
expand_contractions = get_contraction_to_expansion
expand_abbreviations = get_expand_abbreviations
normalize_whitespace = remove_multiple_whitespaces
