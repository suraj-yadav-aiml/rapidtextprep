"""Package resource loading and built-in fallback mappings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

_DATA_PACKAGE = "rapidtextprep.data"
_CONTRACTION_FILENAME = "english_contractions_mapping_updated.json"
_ABBREVIATION_FILENAME = "social_media_abbreviations_mapping.json"


_FALLBACK_CONTRACTION_OPTIONS: dict[str, str | list[str]] = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": ["he would", "he had"],
    "he'll": "he will",
    "he's": ["he is", "he has"],
    "how'd": ["how did", "how would"],
    "how'll": "how will",
    "how's": ["how is", "how has"],
    "i'd": ["i would", "i had"],
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": ["it would", "it had"],
    "it'll": "it will",
    "it's": ["it is", "it has"],
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": ["she would", "she had"],
    "she'll": "she will",
    "she's": ["she is", "she has"],
    "shouldn't": "should not",
    "that's": ["that is", "that has"],
    "there's": ["there is", "there has"],
    "they'd": ["they would", "they had"],
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": ["we would", "we had"],
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'd": ["what did", "what would"],
    "what's": ["what is", "what has"],
    "where'd": ["where did", "where would"],
    "where's": ["where is", "where has"],
    "who'd": ["who would", "who had"],
    "who'll": "who will",
    "who's": ["who is", "who has"],
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": ["you would", "you had"],
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_FALLBACK_ABBREVIATION_TO_EXPANSION: dict[str, str] = {
    "a3": "anytime anywhere anyplace",
    "afaik": "as far as i know",
    "afk": "away from keyboard",
    "asap": "as soon as possible",
    "atm": "at the moment",
    "b4": "before",
    "b/c": "because",
    "bc": "because",
    "brb": "be right back",
    "btw": "by the way",
    "dm": "direct message",
    "fomo": "fear of missing out",
    "fyi": "for your information",
    "gg": "good game",
    "idc": "i do not care",
    "idk": "i do not know",
    "ikr": "i know right",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "lmk": "let me know",
    "lol": "laughing out loud",
    "nvm": "never mind",
    "omg": "oh my god",
    "rn": "right now",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "tldr": "too long did not read",
    "tl;dr": "too long did not read",
    "ttyl": "talk to you later",
    "w/": "with",
    "w/o": "without",
    "wdym": "what do you mean",
    "wyd": "what are you doing",
    "yolo": "you only live once",
    "\u20b9": " rupee ",
    "\u20ac": " euro ",
    "\xa5": " yen ",
    "$": " dollar ",
    "%": " percent ",
}


def _load_json_mapping(filename: str, fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Load a package JSON mapping, falling back to a built-in mapping."""
    try:
        resource = files(_DATA_PACKAGE).joinpath(filename)
        with resource.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except FileNotFoundError:
        return dict(fallback)

    if not isinstance(loaded, dict):
        raise TypeError(f"{filename} must contain a JSON object")

    return dict(loaded)


CONTRACTION_OPTIONS: dict[str, str | list[str]] = {
    str(key).lower(): value
    for key, value in _load_json_mapping(
        _CONTRACTION_FILENAME,
        _FALLBACK_CONTRACTION_OPTIONS,
    ).items()
}

CONTRACTION_TO_EXPANSION: dict[str, str] = {
    contraction: expansions[0] if isinstance(expansions, list) else str(expansions)
    for contraction, expansions in CONTRACTION_OPTIONS.items()
}

CONTRACTION_TO_EXPANSION.update(
    {
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "here's": "here is",
        "who's": "who is",
        "what's": "what is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "it'd": "it would",
        "we'd": "we would",
        "they'd": "they would",
        "what'd": "what did",
        "where'd": "where did",
        "when'd": "when did",
        "why'd": "why did",
        "how'd": "how did",
    }
)

UNSAFE_CONTRACTION_KEYS: frozenset[str] = frozenset({"cause", "ya"})

for _unsafe_key in UNSAFE_CONTRACTION_KEYS:
    CONTRACTION_TO_EXPANSION.pop(_unsafe_key, None)

ABBREVIATION_TO_EXPANSION: dict[str, str] = {
    str(key).lower(): str(value)
    for key, value in _load_json_mapping(
        _ABBREVIATION_FILENAME,
        _FALLBACK_ABBREVIATION_TO_EXPANSION,
    ).items()
}

SYMBOL_TRANSLATION: dict[int, str] = str.maketrans(
    {
        "\u20b9": " rupee ",
        "\u20ac": " euro ",
        "\xa5": " yen ",
        "$": " dollar ",
        "%": " percent ",
    }
)

SYMBOL_KEYS: frozenset[str] = frozenset({"\u20b9", "\u20ac", "\xa5", "$", "%"})

TOKEN_ABBREVIATION_TO_EXPANSION: dict[str, str] = {
    abbreviation: expansion
    for abbreviation, expansion in ABBREVIATION_TO_EXPANSION.items()
    if abbreviation not in SYMBOL_KEYS
}
