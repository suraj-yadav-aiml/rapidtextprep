# User Guide

## Complete Cleaning Pipeline

`clean_text` is the beginner-friendly alias for
`get_complete_text_clean_up_batch`.

```python
from rapidtextprep import clean_text

cleaned = clean_text(
    texts,
    handle_missing="empty",
    keep_stopwords=None,
    extra_stopwords={"example"},
    use_lemmatization=False,
    chunk_size=100_000,
)
```

The pipeline order is:

1. Lowercase text.
2. Expand contractions.
3. Expand social-media abbreviations.
4. Normalize accented characters.
5. Remove HTML tags.
6. Remove email addresses.
7. Remove URLs.
8. Remove standalone retweet markers.
9. Remove special characters.
10. Remove stopwords.
11. Optionally lemmatize text.
12. Normalize whitespace.

## Normalization Utilities

```python
from rapidtextprep import (
    expand_abbreviations,
    expand_contractions,
    lowercase_text,
    normalize_whitespace,
    remove_accented_chars,
)

lowercase_text("Hello WORLD")
expand_contractions("i'm sure he won't go")
expand_abbreviations("btw idk irl")
remove_accented_chars("cafe")
normalize_whitespace("  hello    world  ")
```

## Cleaning Utilities

```python
from rapidtextprep import (
    remove_email,
    remove_html_tags,
    remove_rt,
    remove_special_characters,
    remove_urls,
)

remove_email("contact test@example.com")
remove_urls("visit https://example.com now")
remove_rt("RT @user: hello")
remove_html_tags("<p>Hello</p>")
remove_special_characters("hello!!! #nlp")
```

## Stopword Removal

```python
from rapidtextprep import remove_stopwords

text = remove_stopwords("this movie is not good but very emotional")
```

By default, sentiment-relevant words such as `not`, `no`, `but`, and `very`
are preserved.

Use FlashText when it is faster for your data:

```python
text = remove_stopwords(
    "this movie is not good but very emotional",
    backend="flashtext",
)
```

## Lemmatization

```python
from rapidtextprep import lemmatize_text

lemmatized = lemmatize_text("cars were running faster")
print(lemmatized)
```

Enable lemmatization inside the full cleaning pipeline:

```python
cleaned = clean_text(
    texts,
    use_lemmatization=True,
    lemmatize_batch_size=5_000,
    n_process=1,
)
```

## Feature Generation

```python
from rapidtextprep import get_basic_features

features = get_basic_features(df, column="text")
```

## URL and Email Extraction

```python
from rapidtextprep import get_email, get_urls

email_count, emails = get_email("mail test@example.com")
url_count, urls = get_urls("visit https://example.com")
```
