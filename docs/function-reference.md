# Function Reference

This section explains the public functions in `rapidtextprep` with syntax,
parameters, return values, and runnable examples.

Most low-level functions accept either:

- a single Python `str`
- a `pandas.Series` of text

Feature functions usually require a `pandas.Series`, and dataframe helpers
require a `pandas.DataFrame`.

## Recommended Reading Order

1. [Normalization](functions/normalization.md)
2. [Cleaning](functions/cleaning.md)
3. [Stopwords](functions/stopwords.md)
4. [Extraction](functions/extraction.md)
5. [Features](functions/features.md)
6. [Frequency Utilities](functions/frequency.md)
7. [Lemmatization](functions/lemmatization.md)
8. [Pipeline](functions/pipeline.md)
9. [Configuration](functions/configuration.md)
10. [scikit-learn Transformer](functions/transformers.md)

## Public Aliases

These aliases are exported for beginner-friendly imports:

| Alias | Same as |
| --- | --- |
| `clean_text` | `get_complete_text_clean_up_batch` |
| `lemmatize_text` | `get_lemmatize_text_fast` |
| `lowercase_text` | `get_lower_case` |
| `expand_contractions` | `get_contraction_to_expansion` |
| `expand_abbreviations` | `get_expand_abbreviations` |
| `normalize_whitespace` | `remove_multiple_whitespaces` |

Example:

```python
from rapidtextprep import clean_text, lowercase_text

print(lowercase_text("HELLO World"))
print(clean_text("RT @User: I CAN'T believe this!!!"))
```

