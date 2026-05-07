# Extraction Functions

Extraction functions return matched patterns instead of removing them.

For `pandas.Series` input, matches are collected across the full Series and
returned as one NumPy array, not one array per row.

## `get_email`

Extracts email addresses.

### Syntax

```python
get_email(text, return_count=True)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to search. |
| `return_count` | `bool` | `True` | Whether to return `(count, emails)` or only `emails`. |

### Returns

If `return_count=True`, returns:

```python
(count, emails_array)
```

If `return_count=False`, returns:

```python
emails_array
```

### Examples

```python
import pandas as pd
from rapidtextprep import get_email

text = "Mail a@test.com and b@test.com"
count, emails = get_email(text)
print(count)
print(emails.tolist())
# 2
# ['a@test.com', 'b@test.com']

emails_only = get_email(text, return_count=False)
print(emails_only.tolist())
# ['a@test.com', 'b@test.com']

texts = pd.Series(["first@example.com", "no email", "second@example.com"])
count, emails = get_email(texts)
print(count)
print(emails.tolist())
# 2
# ['first@example.com', 'second@example.com']
```

## `get_urls`

Extracts URLs.

### Syntax

```python
get_urls(text, return_count=True)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to search. |
| `return_count` | `bool` | `True` | Whether to return `(count, urls)` or only `urls`. |

### Returns

If `return_count=True`, returns:

```python
(count, urls_array)
```

If `return_count=False`, returns:

```python
urls_array
```

### Examples

```python
import pandas as pd
from rapidtextprep import get_urls

text = "Open https://example.com and http://docs.example.com"
count, urls = get_urls(text)
print(count)
print(urls.tolist())
# 2
# ['https://example.com', 'http://docs.example.com']

urls_only = get_urls(text, return_count=False)
print(urls_only.tolist())
# ['https://example.com', 'http://docs.example.com']

texts = pd.Series(["visit https://a.com", "plain text", "see https://b.com"])
count, urls = get_urls(texts)
print(count)
print(urls.tolist())
# 2
# ['https://a.com', 'https://b.com']
```

