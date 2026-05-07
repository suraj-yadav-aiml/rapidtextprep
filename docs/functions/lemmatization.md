# Lemmatization Functions

`rapidtextprep` uses spaCy's blank English pipeline with lookup lemmatization.
It does not require `en_core_web_sm` or `en_core_web_md`.

## `get_lemmatize_text_fast`

Lemmatizes text using spaCy lookup tables.

Alias: `lemmatize_text`

### Syntax

```python
get_lemmatize_text_fast(text, batch_size=5000, n_process=1)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` or `pandas.Series` | Required | Text to lemmatize. |
| `batch_size` | `int` | `5000` | Number of documents per spaCy batch for Series input. |
| `n_process` | `int` | `1` | Number of spaCy worker processes for `nlp.pipe`. |

### Returns

Lemmatized text with the same shape as the input.

### Examples

```python
import pandas as pd
from rapidtextprep import get_lemmatize_text_fast, lemmatize_text

text = "cars were running faster"
print(get_lemmatize_text_fast(text))
# car be run fast

texts = pd.Series(["dogs are barking", "children were playing"])
print(lemmatize_text(texts).tolist())
# ['dog be bark', 'child be play']
```

Use larger batches for bigger Series:

```python
import pandas as pd
from rapidtextprep import lemmatize_text

texts = pd.Series(["cars were running faster"] * 10000)

lemmatized = lemmatize_text(
    texts,
    batch_size=5000,
    n_process=1,
)
```

Use multiple spaCy processes when it is beneficial for your machine and data:

```python
import pandas as pd
from rapidtextprep import lemmatize_text

texts = pd.Series(["cars were running faster"] * 10000)

lemmatized = lemmatize_text(
    texts,
    batch_size=2000,
    n_process=2,
)
```

