"""Shared type aliases used by rapidtextprep modules."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

TextInput: TypeAlias = pd.Series | str
TextOutput: TypeAlias = pd.Series | str
EmailReturn: TypeAlias = np.ndarray | tuple[int, np.ndarray]
UrlReturn: TypeAlias = np.ndarray | tuple[int, np.ndarray]
