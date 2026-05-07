# Cleaning Functions

Cleaning functions remove noisy text patterns such as emails, URLs, HTML tags,
retweet markers, and punctuation. These functions intentionally do not normalize
whitespace. Run `remove_multiple_whitespaces` after them when you want compact
output.

## `remove_email`

Removes email addresses.

### Syntax

```python
remove_email(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text that may contain email addresses. |

### Returns

Text with email addresses removed.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_email

text = "Contact support@example.com for help"
print(repr(remove_email(text)))
# 'Contact  for help'

texts = pd.Series(["a@test.com replied", "no email here"])
print(remove_email(texts).tolist())
# [' replied', 'no email here']
```

## `remove_urls`

Removes URLs from text.

### Syntax

```python
remove_urls(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text that may contain URLs. |

### Returns

Text with URLs removed.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_urls

text = "Read https://example.com/docs today"
print(repr(remove_urls(text)))
# 'Read  today'

texts = pd.Series(["go to https://a.com", "plain text"])
print(remove_urls(texts).tolist())
# ['go to ', 'plain text']
```

## `remove_rt`

Removes standalone retweet markers such as `RT`.

### Syntax

```python
remove_rt(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text that may contain standalone retweet markers. |

### Returns

Text with standalone retweet markers removed.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_rt

text = "RT @user: useful post"
print(repr(remove_rt(text)))
# ' @user: useful post'

texts = pd.Series(["RT this is shared", "not a retweet"])
print(remove_rt(texts).tolist())
# [' this is shared', 'not a retweet']
```

## `remove_special_characters`

Replaces punctuation, symbols, underscores, and other special characters with
spaces. Letters, digits, and whitespace are preserved.

### Syntax

```python
remove_special_characters(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text that may contain punctuation or symbols. |

### Returns

Text with special characters replaced by spaces.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_special_characters

text = "Hello!!! #NLP @user 50%"
print(repr(remove_special_characters(text)))
# 'Hello    NLP  user 50 '

texts = pd.Series(["fast_text", "price=$20"])
print(remove_special_characters(texts).tolist())
# ['fast text', 'price  20']
```

## `remove_html_tags`

Removes HTML/XML-like tags and replaces them with spaces.

### Syntax

```python
remove_html_tags(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text that may contain HTML tags. |

### Returns

Text with HTML tags removed.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_html_tags

text = "<p>Hello <b>world</b></p>"
print(repr(remove_html_tags(text)))
# ' Hello  world  '

texts = pd.Series(["<div>Fast</div>", "plain text"])
print(remove_html_tags(texts).tolist())
# [' Fast ', 'plain text']
```

