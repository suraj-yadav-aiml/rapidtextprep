"""scikit-learn compatible text preprocessing transformers."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MissingValueStrategy, ParallelBackend
from .pipeline import clean_text
from .stopwords import DEFAULT_KEEP_STOPWORDS, StopwordBackend


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn compatible transformer for rapidtextprep cleaning.

    This transformer is stateless and can be used before vectorizers such as
    ``TfidfVectorizer`` inside a scikit-learn ``Pipeline``.
    """

    def __init__(
        self,
        keep_stopwords: Collection[str] | None = DEFAULT_KEEP_STOPWORDS,
        extra_stopwords: Collection[str] | None = None,
        stopword_backend: StopwordBackend = "regex",
        handle_missing: MissingValueStrategy = "empty",
        use_lemmatization: bool = False,
        chunk_size: int | None = 100_000,
        lemmatize_batch_size: int = 5_000,
        n_process: int = 1,
        n_jobs: int = 1,
        parallel_backend: ParallelBackend = "thread",
        verbose: bool = False,
    ) -> None:
        self.keep_stopwords = keep_stopwords
        self.extra_stopwords = extra_stopwords
        self.stopword_backend = stopword_backend
        self.handle_missing = handle_missing
        self.use_lemmatization = use_lemmatization
        self.chunk_size = chunk_size
        self.lemmatize_batch_size = lemmatize_batch_size
        self.n_process = n_process
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.verbose = verbose

    def fit(self, X: object, y: object | None = None) -> TextPreprocessor:
        """Return self because the transformer has no learned state."""
        return self

    def transform(self, X: object) -> np.ndarray:
        """Clean text input and return a 1D numpy array."""
        text = self._to_series(X)
        cleaned = clean_text(
            text,
            keep_stopwords=self.keep_stopwords,
            extra_stopwords=self.extra_stopwords,
            stopword_backend=self.stopword_backend,
            handle_missing=self.handle_missing,
            use_lemmatization=self.use_lemmatization,
            chunk_size=self.chunk_size,
            lemmatize_batch_size=self.lemmatize_batch_size,
            n_process=self.n_process,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            verbose=self.verbose,
        )

        if not isinstance(cleaned, pd.Series):
            return np.asarray([cleaned], dtype=object)

        return cleaned.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the output feature name used by this transformer."""
        return np.array(["clean_text"], dtype=object)

    def _to_series(self, X: object) -> pd.Series:
        """Convert supported scikit-learn inputs into a pandas Series."""
        if isinstance(X, str):
            return pd.Series([X], name="text", dtype=object)

        if isinstance(X, pd.Series):
            return X

        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "TextPreprocessor expects a single-column DataFrame. "
                    "Select one text column or use ColumnTransformer."
                )

            return X.iloc[:, 0]

        if isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError("TextPreprocessor expects a 1D numpy array")

            return pd.Series(X, name="text", dtype=object)

        if isinstance(X, list | tuple):
            return pd.Series(X, name="text", dtype=object)

        raise TypeError(
            "X must be a string, list, tuple, 1D numpy array, pandas Series, "
            "or single-column pandas DataFrame"
        )
