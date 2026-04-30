"""Pattern extraction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._patterns import EMAIL_EXTRACT_PATTERN, EMAIL_PATTERN, URL_PATTERN
from ._typing import EmailReturn, TextInput, UrlReturn


def get_email(
    text: TextInput,
    return_count: bool = True,
) -> EmailReturn:
    """Extract email addresses from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.
        return_count: Whether to return count along with emails. Defaults to
            True.

    Returns:
        If return_count is True, returns ``(count, emails_array)``.
        Otherwise returns only the emails array.

    Example:
        >>> get_email("mail me at test@example.com")
        (1, array(['test@example.com'], dtype=object))
    """
    if isinstance(text, str):
        emails = np.asarray(EMAIL_PATTERN.findall(text), dtype=object)

    elif isinstance(text, pd.Series):
        emails = text.str.extractall(EMAIL_EXTRACT_PATTERN)[0].to_numpy(
            dtype=object,
            copy=False,
        )

    else:
        raise TypeError("text must be a string or pandas Series")

    if return_count:
        return emails.size, emails

    return emails


def get_urls(
    text: TextInput,
    return_count: bool = True,
) -> UrlReturn:
    """Extract URLs from a string or pandas Series.

    Args:
        text: A single text string or pandas Series.
        return_count: Whether to return count along with URLs. Defaults to
            True.

    Returns:
        If return_count is True, returns ``(count, urls_array)``.
        Otherwise returns only the URLs array.

    Example:
        >>> get_urls("visit https://example.com")
        (1, array(['https://example.com'], dtype=object))
    """
    if isinstance(text, str):
        urls = np.asarray(URL_PATTERN.findall(text), dtype=object)

    elif isinstance(text, pd.Series):
        urls = text.str.extractall(URL_PATTERN)[0].to_numpy(
            dtype=object,
            copy=False,
        )

    else:
        raise TypeError("text must be a string or pandas Series")

    if return_count:
        return urls.size, urls

    return urls
