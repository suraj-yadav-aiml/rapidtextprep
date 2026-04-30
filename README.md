# rapidtextprep

Fast, pandas-friendly text preprocessing utilities for common NLP cleanup tasks.

## Install

```powershell
uv pip install rapidtextprep
```

For local development:

```powershell
uv sync
```

## Usage

```python
from rapidtextprep import clean_text, remove_stopwords

clean_text("RT @User: I CAN'T believe this cafe is 50% OFF!!!")
remove_stopwords("this movie is not good but very emotional")
```

```python
import pandas as pd
from rapidtextprep import clean_text

texts = pd.Series(["I CAN'T wait!!!", "Visit https://example.com"])
cleaned = clean_text(texts)
```
