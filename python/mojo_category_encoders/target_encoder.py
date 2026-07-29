from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

from . import _lib
from ._base import BaseEncoder
from .ordinal import OrdinalEncoder


class TargetEncoder(BaseEncoder):
    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_missing: str = "value",
        handle_unknown: str = "value",
        min_samples_leaf: int = 20,
        smoothing: float = 10,
        hierarchy: dict = None,
    ) -> None:
        self.verbose = verbose
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.hierarchy = hierarchy
        self.ordinal_encoder = None
        self.mapping = None
        self._mean = None
        self.invariant_cols = []
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        if self.hierarchy is not None:
            raise NotImplementedError("hierarchical target encoding is not in the accelerated subset")
        frame, target = self._fit_setup(X, y, supervised=True)
        if not is_numeric_dtype(target):
            self.lab_encoder_ = LabelEncoder()
            target = pd.Series(
                self.lab_encoder_.fit_transform(target), index=target.index, dtype=np.float64
            )
        else:
            self.lab_encoder_ = None
        values = np.ascontiguousarray(target, dtype=np.float64)
        self._mean = float(values.mean())
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        ).fit(frame)
        ordinal = self.ordinal_encoder.transform(frame, override_return_df=True)
        self.mapping = {}
        self._dense_mapping = {}
        self._position_mapping = {}
        self._category_keys = {}
        for item in self.ordinal_encoder.category_mapping:
            col = item["col"]
            codes = np.ascontiguousarray(ordinal[col], dtype=np.int64)
            groups = max(3, int(codes.max(initial=-2)) + 3)
            counts, sums = _lib.target_stats(codes, values, groups)
            dense = np.full(
                groups,
                np.nan if self.handle_unknown == "return_nan" else self._mean,
                dtype=np.float64,
            )
            present = counts > 0
            weight = np.zeros(groups, dtype=np.float64)
            weight[present] = 1.0 / (
                1.0 + np.exp(-(counts[present] - self.min_samples_leaf) / self.smoothing)
            )
            dense[present] = (
                self._mean * (1.0 - weight[present])
                + (sums[present] / counts[present]) * weight[present]
            )
            observed_codes = np.flatnonzero(present) - 2
            mapping = pd.Series(dense[present], index=observed_codes)
            mapping.loc[-1] = np.nan if self.handle_unknown == "return_nan" else self._mean
            if self.handle_missing == "return_nan":
                missing_code = int(item["mapping"][item["mapping"].index.isna()].iloc[0])
                if missing_code + 2 >= len(dense):
                    dense = np.pad(dense, (0, missing_code + 3 - len(dense)), constant_values=self._mean)
                dense[missing_code + 2] = np.nan
                mapping.loc[missing_code] = np.nan
            elif self.handle_missing == "value":
                mapping.loc[-2] = self._mean
                dense[0] = self._mean
            self.mapping[col] = mapping
            self._dense_mapping[col] = dense
            keys = [value for value in item["mapping"].index if not pd.isna(value)]
            ordinal_codes = item["mapping"].loc[keys].to_numpy(dtype=np.int64)
            position_mapping = np.full(
                len(keys) + 2,
                np.nan if self.handle_unknown == "return_nan" else self._mean,
                dtype=np.float64,
            )
            missing_rows = item["mapping"].index.isna()
            if missing_rows.any():
                missing_code = int(item["mapping"][missing_rows].iloc[0])
                position_mapping[0] = dense[missing_code + 2]
            position_mapping[2:] = dense[ordinal_codes + 2]
            self._category_keys[col] = keys
            self._position_mapping[col] = position_mapping
        self._finish_fit(frame, list(self.cols))
        return self

    def transform(self, X, y=None, override_return_df: bool = False):
        frame = self._transform_setup(X)
        if not self.cols:
            return self._return(frame, override_return_df)
        for col in self.cols:
            series = frame[col]
            codes = pd.Categorical(
                series, categories=self._category_keys[col]
            ).codes.astype(np.int64)
            if series.hasnans:
                codes[series.isna().to_numpy()] = -2
            if self.handle_unknown == "error" and (codes == -1).any():
                raise ValueError("Unexpected categories found in dataframe")
            default = np.nan if self.handle_unknown == "return_nan" else self._mean
            frame[col] = _lib.target_apply(
                codes, self._position_mapping[col], default
            )
        return self._return(frame, override_return_df)

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise TypeError("fit_transform() missing argument: y")
        return self.fit(X, y, **fit_params).transform(X, y)

    def target_encode(self, X_in: pd.DataFrame) -> pd.DataFrame:
        frame = X_in.copy(deep=True)
        for col in self.cols:
            codes = np.ascontiguousarray(frame[col], dtype=np.int64)
            default = np.nan if self.handle_unknown == "return_nan" else self._mean
            frame[col] = _lib.target_apply(codes, self._dense_mapping[col], default)
        return frame

    def _weighting(self, n):
        return 1.0 / (1.0 + np.exp(-(n - self.min_samples_leaf) / self.smoothing))
