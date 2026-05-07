# Installation

## Install from PyPI

```bash
pip install rapidtextprep
```

## Install with uv

```bash
uv pip install rapidtextprep
```

## Development Install

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/suraj-yadav-aiml/rapidtextprep.git
cd rapidtextprep
uv sync
uv pip install -e .
```

## Runtime Dependencies

The package declares its runtime dependencies in `pyproject.toml`.

Core dependencies include:

- `numpy`
- `pandas`
- `joblib`
- `flashtext`
- `scikit-learn`
- `spacy`
- `spacy-lookups-data`

No downloadable spaCy model such as `en_core_web_sm` or `en_core_web_md` is
required. Lemmatization uses `spacy.blank("en")` with lookup data.

## Documentation Dependencies

To build the documentation locally:

```bash
uv sync
uv run mkdocs serve
```

Then open:

```text
http://127.0.0.1:8000
```
