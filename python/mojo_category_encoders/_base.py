from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


def convert_input(
    X, columns: Sequence | None = None, index: Sequence | None = None, deep: bool = False
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=True) if deep else X
    if isinstance(X, pd.Series):
        return pd.DataFrame(X, copy=deep)
    try:
        return pd.DataFrame(X, columns=columns, index=index, copy=deep)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected input type: {type(X)}") from exc


def convert_target(y, index: pd.Index) -> pd.Series:
    if y is None:
        raise ValueError("Supervised encoders need a target for the fitting. The target cannot be None")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Unexpected input shape: {y.shape}")
        y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        array = np.asarray(y)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(f"Unexpected input shape: {array.shape}")
        y = pd.Series(array, index=index, name="target")
    if len(y) != len(index):
        raise ValueError(f"The length of X is {len(index)} but length of y is {len(y)}.")
    if not y.index.equals(index):
        raise ValueError(
            "`X` and `y` both have indexes, but they do not match. "
            "Use numpy arrays when the indexes are intentionally unrelated."
        )
    return y


def categorical_columns(frame: pd.DataFrame) -> list:
    return [
        col
        for col, dtype in frame.dtypes.items()
        if is_object_dtype(dtype) or is_string_dtype(dtype) or isinstance(dtype, CategoricalDtype)
    ]


class BaseEncoder(BaseEstimator, TransformerMixin):
    def _fit_setup(self, X, y=None, supervised: bool = False):
        frame = convert_input(X, deep=True)
        target = convert_target(y, frame.index) if supervised else None
        if supervised and target.isna().any():
            raise ValueError("The target column y must not contain missing values.")
        self.feature_names_in_ = frame.columns.tolist()
        self.n_features_in_ = frame.shape[1]
        self._dim = frame.shape[1]
        if isinstance(self.cols, str) and self.cols.lower() == "all":
            self.cols = frame.columns.tolist()
        elif self.cols is None:
            self.cols = categorical_columns(frame)
        elif np.isscalar(self.cols):
            self.cols = [self.cols]
        else:
            self.cols = list(self.cols)
        if not set(self.cols).issubset(frame.columns):
            raise ValueError("X does not contain the columns listed in cols")
        self._validate_strategies()
        self._check_missing(frame)
        return frame, target

    def _validate_strategies(self):
        for name in ("handle_missing", "handle_unknown"):
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if value not in ("error", "return_nan", "value"):
                raise ValueError(f"Unexpected {name} value {value!r}")

    def _check_missing(self, frame: pd.DataFrame):
        if getattr(self, "handle_missing", None) == "error":
            if frame[self.cols].isna().any().any():
                raise ValueError("Columns to be encoded cannot contain null")

    def _transform_setup(self, X) -> pd.DataFrame:
        if getattr(self, "_dim", None) is None:
            raise NotFittedError("Must train encoder before it can be used to transform data.")
        frame = convert_input(X, deep=True)
        if frame.shape[1] != self._dim:
            raise ValueError(f"Unexpected input dimension {frame.shape[1]}, expected {self._dim}")
        self._check_missing(frame)
        return frame

    def _finish_fit(self, frame: pd.DataFrame, generated: list):
        transformed = self.transform(frame, override_return_df=True)
        self.invariant_cols = []
        if self.drop_invariant:
            self.invariant_cols = [
                col for col in generated if transformed[col].nunique() <= 1
            ]
            transformed = transformed.drop(columns=self.invariant_cols)
        self.feature_names_out_ = transformed.columns.to_numpy()

    def _return(self, frame: pd.DataFrame, override_return_df: bool):
        if self.drop_invariant and self.invariant_cols:
            frame = frame.drop(columns=self.invariant_cols)
        return frame if self.return_df or override_return_df else frame.to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not isinstance(getattr(self, "feature_names_out_", None), np.ndarray):
            raise NotFittedError("Estimator has to be fitted to return feature names.")
        return self.feature_names_out_

    def get_feature_names_in(self) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            raise NotFittedError("Estimator has to be fitted to return feature names.")
        return np.asarray(self.feature_names_in_)

    def get_feature_names(self) -> np.ndarray:
        warnings.warn(
            "`get_feature_names` is deprecated; use `get_feature_names_out`.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_feature_names_out()
