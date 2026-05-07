# Normalization Functions

Normalization functions standardize text before or after cleaning. They do not
handle missing values by themselves. If your data contains `None` or `NaN`, use
`clean_text(..., handle_missing=...)` or clean missing values before calling
these functions directly.

## `get_lower_case`

Converts text to lowercase.

Alias: `lowercase_text`

### Syntax

```python
get_lower_case(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text to lowercase. |

### Returns

Returns the same shape as the input:

- `str` input returns `str`
- `pandas.Series` input returns `pandas.Series`

### Examples

```python
import pandas as pd
from rapidtextprep import get_lower_case, lowercase_text

text = "Python NLP Is FAST"
print(get_lower_case(text))
# python nlp is fast

texts = pd.Series(["HELLO World", "Text PREPROCESSING"])
print(get_lower_case(texts).tolist())
# ['hello world', 'text preprocessing']

print(lowercase_text("AN ALIAS WORKS TOO"))
# an alias works too
```

## `get_contraction_to_expansion`

Expands English contractions using the packaged contraction mapping.

Alias: `expand_contractions`

This function works best after lowercasing because the contraction mapping is
lowercase-focused.

### Syntax

```python
get_contraction_to_expansion(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Lowercase text containing contractions. |

### Returns

Text with matched contractions expanded.

### Examples

```python
import pandas as pd
from rapidtextprep import expand_contractions, get_contraction_to_expansion

text = "i can't go because he won't wait"
print(get_contraction_to_expansion(text))
# i cannot go because he will not wait

texts = pd.Series(["i'm ready", "they're coming", "he's been working"])
print(expand_contractions(texts).tolist())
# ['i am ready', 'they are coming', 'he has been working']
```

## `get_expand_abbreviations`

Expands social-media and texting abbreviations using the packaged abbreviation
mapping.

Alias: `expand_abbreviations`

This function assumes lowercase input.

### Syntax

```python
get_expand_abbreviations(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Lowercase text containing abbreviations. |

### Returns

Text with known abbreviations expanded.

### Examples

```python
import pandas as pd
from rapidtextprep import expand_abbreviations, get_expand_abbreviations

text = "btw idk what is happening rn"
print(get_expand_abbreviations(text))
# by the way i do not know what is happening right now

texts = pd.Series(["brb ttyl", "imo this is good"])
print(expand_abbreviations(texts).tolist())
# ['be right back talk to you later', 'in my opinion this is good']
```

## `remove_accented_chars`

Converts accented characters to ASCII equivalents when possible.

### Syntax

```python
remove_accented_chars(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text containing accented characters. |

### Returns

Text with accented characters normalized to ASCII.

### Examples

```python
import pandas as pd
from rapidtextprep import remove_accented_chars

text = "caf\u00e9 d\u00e9j\u00e0 vu fa\u00e7ade"
print(remove_accented_chars(text))
# cafe deja vu facade

texts = pd.Series(["na\u00efve approach", "r\u00e9sum\u00e9 parsing"])
print(remove_accented_chars(texts).tolist())
# ['naive approach', 'resume parsing']
```

## `remove_multiple_whitespaces`

Collapses consecutive whitespace into a single space and strips leading and
trailing whitespace.

Alias: `normalize_whitespace`

### Syntax

```python
remove_multiple_whitespaces(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Yes | Text containing repeated spaces, tabs, or newlines. |

### Returns

Text with normalized whitespace.

### Examples

```python
import pandas as pd
from rapidtextprep import normalize_whitespace, remove_multiple_whitespaces

text = "  hello     world\nthis\tis fast  "
print(remove_multiple_whitespaces(text))
# hello world this is fast

texts = pd.Series(["  a   b  ", "x\n\ny\tz"])
print(normalize_whitespace(texts).tolist())
# ['a b', 'x y z']
```

