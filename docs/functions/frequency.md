# Frequency Utilities

Frequency utilities use word counts across a full Series to remove common or
rare words. These helpers are useful when you want dataset-specific vocabulary
filtering before vectorization.

## `get_value_counts`

Returns word frequencies across a full `pandas.Series`.

### Syntax

```python
get_value_counts(text)
```

### Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | `pandas.Series` | Yes | Text Series used to compute word counts. |

### Returns

A `pandas.Series` where the index contains words and the values contain counts.

### Examples

```python
import pandas as pd
from rapidtextprep import get_value_counts

texts = pd.Series(["red blue red", "green blue red"])
counts = get_value_counts(texts)

print(counts.to_dict())
# {'red': 3, 'blue': 2, 'green': 1}
```

## `remove_common_word`

Removes the top `n_words` most frequent words.

### Syntax

```python
remove_common_word(text, word_counts, n_words, ignore_case=False)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to process. |
| `word_counts` | `pandas.Series` | Required | Counts returned by `get_value_counts`. |
| `n_words` | `int` | Required | Number of most common words to remove. |
| `ignore_case` | `bool` | `False` | Remove words case-insensitively when `True`. |

### Returns

Text with common words replaced by spaces.

### Examples

```python
import pandas as pd
from rapidtextprep import get_value_counts, remove_common_word

texts = pd.Series(["red blue red", "green blue red"])
counts = get_value_counts(texts)

print(remove_common_word(texts, counts, n_words=1).tolist())
# ['  blue  ', 'green blue  ']
```

Use case-insensitive removal:

```python
import pandas as pd
from rapidtextprep import get_value_counts, remove_common_word

texts = pd.Series(["Red blue", "red green"])
counts = get_value_counts(texts.str.lower())

print(remove_common_word(texts, counts, n_words=1, ignore_case=True).tolist())
# ['  blue', '  green']
```

## `remove_rarewords`

Removes the bottom `n_words` least frequent words.

### Syntax

```python
remove_rarewords(text, word_counts, n_words, ignore_case=False)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to process. |
| `word_counts` | `pandas.Series` | Required | Counts returned by `get_value_counts`. |
| `n_words` | `int` | Required | Number of least common words to remove. |
| `ignore_case` | `bool` | `False` | Remove words case-insensitively when `True`. |

### Returns

Text with rare words replaced by spaces.

### Examples

```python
import pandas as pd
from rapidtextprep import get_value_counts, remove_rarewords

texts = pd.Series(["red blue red", "green blue red"])
counts = get_value_counts(texts)

print(remove_rarewords(texts, counts, n_words=1).tolist())
# ['red blue red', '  blue red']
```

Use `n_words=0` to leave the input unchanged:

```python
import pandas as pd
from rapidtextprep import get_value_counts, remove_rarewords

texts = pd.Series(["red blue", "green blue"])
counts = get_value_counts(texts)

print(remove_rarewords(texts, counts, n_words=0).tolist())
# ['red blue', 'green blue']
```

