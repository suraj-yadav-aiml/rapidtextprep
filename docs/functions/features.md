# Feature Functions

Feature functions create simple numeric signals from text columns. They are
useful for quick EDA, classical ML models, and model diagnostics.

These functions expect non-null `pandas.Series` values. For missing values,
fill them before calling feature functions.

## `get_word_count`

Counts words in each row.

### Syntax

```python
get_word_count(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Yes | Text Series to count. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_word_count

texts = pd.Series(["hello world", "pandas is fast", ""])
print(get_word_count(texts).tolist())
# [2, 3, 0]
```

## `get_char_count`

Counts characters in each row.

### Syntax

```python
get_char_count(text, char_count_with_whitespace=False)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Required | Text Series to count. |
| `char_count_with_whitespace` | `bool` | `False` | Include whitespace in the count when `True`. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_char_count

texts = pd.Series(["hello world", " a b "])

print(get_char_count(texts).tolist())
# [10, 2]

print(get_char_count(texts, char_count_with_whitespace=True).tolist())
# [11, 5]
```

## `get_avg_wordlength`

Returns average word length for each row.

### Syntax

```python
get_avg_wordlength(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Yes | Text Series to analyze. |

### Returns

A `pandas.Series` with dtype `float32`. Empty rows return `0.0`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_avg_wordlength

texts = pd.Series(["hello world", "a bb ccc", ""])
print(get_avg_wordlength(texts).tolist())
# [5.0, 2.0, 0.0]
```

## `get_hashtag_count`

Counts hashtags in each row.

### Syntax

```python
get_hashtag_count(text, count_multiple_hash=False)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Required | Text Series to analyze. |
| `count_multiple_hash` | `bool` | `False` | Count hashtags with multiple leading hash symbols, such as `##topic`, when `True`. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_hashtag_count

texts = pd.Series(["I like #NLP and #Python", "##topic"])

print(get_hashtag_count(texts).tolist())
# [2, 0]

print(get_hashtag_count(texts, count_multiple_hash=True).tolist())
# [2, 1]
```

## `get_mentions_count`

Counts social-media mentions such as `@user`.

### Syntax

```python
get_mentions_count(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Yes | Text Series to analyze. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_mentions_count

texts = pd.Series(["thanks @alice and @bob", "hello@example.com"])
print(get_mentions_count(texts).tolist())
# [2, 0]
```

## `get_digit_count`

Counts digit sequences or individual digits in each row.

### Syntax

```python
get_digit_count(text, count_continuous_digit_sequences=True)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Required | Text Series to analyze. |
| `count_continuous_digit_sequences` | `bool` | `True` | Count continuous digit groups when `True`; count individual digits when `False`. |

### Returns

A `pandas.Series` with dtype `int32`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_digit_count

texts = pd.Series(["abc 12 345", "a1b2"])

print(get_digit_count(texts).tolist())
# [2, 2]

print(get_digit_count(texts, count_continuous_digit_sequences=False).tolist())
# [5, 2]
```

## `get_basic_features`

Creates a dataframe with common text feature columns.

The output columns are:

- `char_count`
- `word_count`
- `avg_word_length`
- `stopwords_count`
- `hashtag_count`
- `mentions_count`
- `digit_count`

### Syntax

```python
get_basic_features(df, column)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `df` | `pandas.DataFrame` | Yes | Input dataframe. |
| `column` | `str` | Yes | Name of the text column. |

### Returns

A `pandas.DataFrame` of features with the same index as `df`.

### Examples

```python
import pandas as pd
from rapidtextprep import get_basic_features

df = pd.DataFrame(
    {
        "text": [
            "Hello world #AI 123",
            "No mention here",
        ]
    }
)

features = get_basic_features(df, column="text")
print(features)
```

Example output:

```text
   char_count  word_count  avg_word_length  stopwords_count  hashtag_count  mentions_count  digit_count
0          16           4         4.000000                0              1               0            1
1          13           3         4.333333                2              0               0            0
```

The exact float display may vary slightly depending on pandas display options.
