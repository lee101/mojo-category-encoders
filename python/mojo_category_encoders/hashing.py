from __future__ import annotations

import hashlib
import math
import multiprocessing
import platform

import numpy as np
import pandas as pd

from . import _lib
from ._base import BaseEncoder


class HashingEncoder(BaseEncoder):
    def __init__(
        self,
        max_process=0,
        max_sample=0,
        verbose=0,
        n_components=8,
        cols=None,
        drop_invariant=False,
        return_df=True,
        hash_method="md5",
        process_creation_method="fork",
    ):
        if max_process not in range(1, 128):
            self.max_process = min(128, max(1, int(math.ceil(multiprocessing.cpu_count() / 2))))
        else:
            self.max_process = max_process
        self.max_sample = int(max_sample)
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.hash_method = hash_method
        self.process_creation_method = "spawn" if platform.system() == "Windows" else process_creation_method
        self.invariant_cols = []
        self._dim = None

    def _validate_strategies(self):
        if not callable(getattr(hashlib, self.hash_method, None)):
            raise ValueError(f"Hashing Method: {self.hash_method} not available.")
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise TypeError("n_components must be an integer")
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than zero")

    def _check_missing(self, frame):
        return None

    def fit(self, X, y=None, **kwargs):
        frame, _ = self._fit_setup(X)
        generated = [f"col_{i}" for i in range(self.n_components)]
        remaining = [col for col in frame.columns if col not in self.cols]
        self._finish_fit(frame, generated, generated + remaining)
        return self

    @staticmethod
    def hash_chunk(hash_method: str, np_df: np.ndarray, N: int) -> np.ndarray:
        if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N <= 0:
            raise ValueError("N must be greater than zero")
        constructor = getattr(hashlib, hash_method, None)
        if not callable(constructor):
            raise ValueError(f"Hashing Method: {hash_method} not available.")
        result = np.zeros((np_df.shape[0], N), dtype=int)
        for i, row in enumerate(np_df):
            for value in row:
                if value is not None:
                    digest = constructor(str(value).encode("utf-8")).digest()
                    result[i, int.from_bytes(digest, byteorder="big") % N] += 1
        return result

    def _hash(self, frame: pd.DataFrame, components: int) -> np.ndarray:
        values = frame.to_numpy()
        if self.hash_method == "md5":
            return _lib.hash_md5(values, components)
        return self.hash_chunk(self.hash_method, values, components)

    def hashing_trick_with_np_no_parallel(self, df: pd.DataFrame, N: int) -> pd.DataFrame:
        return pd.DataFrame(self._hash(df, N), index=df.index)

    def hashing_trick_with_np_parallel(self, df: pd.DataFrame, N: int) -> pd.DataFrame:
        return self.hashing_trick_with_np_no_parallel(df, N)

    def hashing_trick(self, X_in, hashing_method="md5", N=2, cols=None, make_copy=False):
        if hashing_method not in hashlib.algorithms_available:
            raise ValueError(f"Hashing Method: {hashing_method} not available.")
        frame = X_in.copy(deep=True) if make_copy else X_in
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        cols = list(frame.columns) if cols is None else list(cols)
        previous = self.hash_method
        self.hash_method = hashing_method
        try:
            encoded = self._hash(frame.loc[:, cols], N)
        finally:
            self.hash_method = previous
        hashed = pd.DataFrame(
            encoded, index=frame.index, columns=[f"col_{i}" for i in range(N)]
        )
        remaining = frame.loc[:, [col for col in frame.columns if col not in cols]]
        return pd.concat([hashed, remaining], axis=1)

    def transform(self, X, override_return_df: bool = False):
        frame = self._transform_setup(X)
        if not self.cols:
            return self._return(frame, override_return_df)
        transformed = self.hashing_trick(
            frame, hashing_method=self.hash_method, N=self.n_components, cols=self.cols
        )
        return self._return(transformed, override_return_df)
