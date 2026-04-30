from __future__ import annotations

import pandas as pd

from rapidtextprep import (
    get_email,
    get_urls,
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)


def test_email_and_url_extraction_and_removal() -> None:
    email_count, emails = get_email("mail test@example.com and admin@example.org")
    assert email_count == 2
    assert emails.tolist() == ["test@example.com", "admin@example.org"]

    url_count, urls = get_urls("visit https://example.com and www.python.org")
    assert url_count == 2
    assert urls.tolist() == ["https://example.com", "www.python.org"]

    assert remove_email("contact me at test@example.com") == "contact me at "
    assert remove_urls("visit https://example.com now") == "visit  now"


def test_cleaning_accepts_series() -> None:
    series = pd.Series(["RT @user: hi!!!", "<p>Hello</p>"])

    assert remove_rt(series).tolist() == [" @user: hi!!!", "<p>Hello</p>"]
    assert remove_html_tags(series).tolist() == ["RT @user: hi!!!", " Hello "]
    assert remove_special_characters(series).tolist() == [
        "RT  user  hi   ",
        " p Hello  p ",
    ]
