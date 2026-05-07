# Stopword Functions

Stopword utilities remove or count common English stopwords. The package exports
two useful constants:

| Constant | Description |
| --- | --- |
| `STOPWORDS` | Combined English stopword set from spaCy and scikit-learn. |
| `DEFAULT_KEEP_STOPWORDS` | Stopwords preserved by default, including negation and sentiment words such as `not`, `no`, `but`, and `very`. |

## `get_stopwords_count`

Counts stopwords in each row of a `pandas.Series`.

### Syntax

```python
get_stopwords_count(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Yes | Series containing non-null text values. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_stopwords_count

texts = pd.Series(["this is a good movie", "hello world"])
print(get_stopwords_count(texts).tolist())
# [3, 0]
```

## `remove_stopwords`

Removes stopwords from a string or `pandas.Series`.

By default, negation, contrast, and sentiment-important words are preserved.
That is useful for ML tasks where `not good` should keep `not`.

### Syntax

```python
remove_stopwords(
    text,
    keep_words=DEFAULT_KEEP_STOPWORDS,
    extra_stopwords=None,
    ignore_case=False,
    backend="regex",
)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to process. |
| `keep_words` | `Collection[str]` or `None` | `DEFAULT_KEEP_STOPWORDS` | Stopwords that should not be removed. Use `None` to remove all stopwords. |
| `extra_stopwords` | `Collection[str]` or `None` | `None` | Extra words to remove in addition to the built-in stopwords. |
| `ignore_case` | `bool` | `False` | Whether stopword removal should be case-insensitive. |
| `backend` | `"regex"` or `"flashtext"` | `"regex"` | Backend used for keyword removal. |

### Returns

Text with stopwords replaced by spaces. Whitespace is not normalized by this
function.

### Examples

Default behavior keeps sentiment-important words:

```python
from rapidtextprep import remove_stopwords

text = "this movie is not good but very emotional"
print(repr(remove_stopwords(text)))
# '  movie   not good but very emotional'
```

Remove all stopwords, including words normally preserved:

```python
from rapidtextprep import remove_stopwords

text = "this movie is not good but very emotional"
print(repr(remove_stopwords(text, keep_words=None)))
# '  movie     good     emotional'
```

Add project-specific stopwords:

```python
from rapidtextprep import remove_stopwords

text = "this movie is not good but very emotional"
print(repr(remove_stopwords(text, extra_stopwords={"movie", "emotional"})))
# '      not good but very  '
```

Use case-insensitive matching:

```python
from rapidtextprep import remove_stopwords

text = "This movie IS Not Good"
print(repr(remove_stopwords(text, ignore_case=True)))
# '  movie   Not Good'
```

Use the FlashText backend:

```python
from rapidtextprep import remove_stopwords

text = "this movie is not good but very emotional"
print(repr(remove_stopwords(text, backend="flashtext")))
# '  movie   not good but very emotional'
```

Use a `pandas.Series`:

```python
import pandas as pd
from rapidtextprep import remove_stopwords

texts = pd.Series(["this is useful", "we are not done"])
print(remove_stopwords(texts, ignore_case=True).tolist())
# ['    useful', '    not  ']
```
