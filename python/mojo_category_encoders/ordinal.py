from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import _lib
from ._base import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    def __init__(
        self,
        verbose: int = 0,
        mapping: list[dict] | None = None,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = "value",
        handle_missing: str = "value",
        index_start: int = 1,
    ):
        self.verbose = verbose
        self.mapping_supplied = mapping is not None
        self.mapping = self._validate_supplied_mapping(mapping) if mapping is not None else None
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.index_start = index_start
        self.invariant_cols = []
        self._dim = None

    @property
    def category_mapping(self):
        return self.mapping

    @staticmethod
    def _validate_supplied_mapping(supplied):
        if not isinstance(supplied, list) or any(not isinstance(item, dict) for item in supplied):
            raise ValueError(
                "Invalid supplied mapping, must be of type "
                "List[Dict[str, Union[Dict, pd.Series]]]."
            )
        validated = []
        for item in supplied:
            if "col" not in item:
                raise KeyError("Mapping must contain a key 'col' for each column to encode")
            if "mapping" not in item:
                raise KeyError("Mapping must contain a key 'mapping' for each column to encode")
            copy = dict(item)
            copy["mapping"] = pd.Series(copy["mapping"])
            copy.setdefault("data_type", copy["mapping"].index.dtype)
            validated.append(copy)
        return validated

    def fit(self, X, y=None, **kwargs):
        frame, _ = self._fit_setup(X)
        if not self.mapping_supplied:
            self.mapping = []
            for col in self.cols:
                categories = list(frame[col].unique())
                if any(pd.isna(value) for value in categories):
                    categories = [value for value in categories if not pd.isna(value)] + [np.nan]
                if isinstance(frame[col].dtype, pd.CategoricalDtype) and frame[col].dtype.ordered:
                    present = set(value for value in categories if not pd.isna(value))
                    ordered = [value for value in frame[col].dtype.categories if value in present]
                    if any(pd.isna(value) for value in categories):
                        ordered.append(np.nan)
                    categories = ordered
                mapping = pd.Series(
                    range(self.index_start, self.index_start + len(categories)),
                    index=pd.Index(categories),
                )
                has_nan = mapping.index.isna().any()
                if self.handle_missing == "value" and not has_nan:
                    mapping.loc[np.nan] = -2
                elif self.handle_missing == "return_nan":
                    mapping.loc[np.nan] = -2
                self.mapping.append(
                    {"col": col, "mapping": mapping.astype(np.int64), "data_type": frame[col].dtype}
                )
        else:
            self.cols = [item["col"] for item in self.mapping]
            if not set(self.cols).issubset(frame.columns):
                raise ValueError("X does not contain the columns listed in cols")
        self._finish_fit(frame, list(self.cols))
        return self

    @staticmethod
    def _positions(series: pd.Series, mapping: pd.Series):
        keys = [value for value in mapping.index if not pd.isna(value)]
        missing_mask = pd.isna(series).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
            positions = pd.Categorical(series, categories=keys).codes.astype(np.int64)
        lookup = mapping.loc[keys].to_numpy(dtype=np.float64)
        missing_rows = mapping.index.isna()
        if missing_rows.any():
            positions[missing_mask] = -2
            missing = float(mapping[missing_rows].iloc[0])
        else:
            missing = np.nan
        return positions, lookup, missing

    def transform(self, X, override_return_df: bool = False):
        frame = self._transform_setup(X)
        if not self.cols:
            return self._return(frame, override_return_df)
        for item in self.mapping:
            col = item["col"]
            positions, lookup, mapped_missing = self._positions(frame[col], item["mapping"])
            unknown_mask = positions == -1
            if self.handle_unknown == "error" and unknown_mask.any():
                raise ValueError(f"Unexpected categories found in column {col}")
            unknown = np.nan if self.handle_unknown == "return_nan" else -1.0
            if self.handle_missing == "return_nan" and mapped_missing == -2:
                missing = np.nan
            elif np.isnan(mapped_missing):
                missing = unknown
            else:
                missing = mapped_missing
            encoded = _lib.ordinal_apply(positions, lookup, unknown, missing)
            if np.isfinite(encoded).all() and np.equal(encoded, np.floor(encoded)).all():
                encoded = encoded.astype(np.int64)
            frame[col] = encoded
        return self._return(frame, override_return_df)

    def inverse_transform(self, X_in):
        frame = self._transform_setup(X_in)
        if self.handle_unknown == "value":
            for col in self.cols:
                if (frame[col] == -1).any():
                    warnings.warn(
                        "inverse_transform cannot reconstruct unknown category -1",
                        stacklevel=2,
                    )
        for item in self.mapping:
            mapping = item["mapping"]
            inverse = pd.Series(mapping.index, index=mapping.values)
            frame[item["col"]] = frame[item["col"]].map(inverse)
            try:
                frame[item["col"]] = frame[item["col"]].astype(item["data_type"])
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Could not restore dtype {item['data_type']!r} for column "
                    f"{item['col']!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return frame if self.return_df else frame.to_numpy()

    @staticmethod
    def ordinal_encoding(
        X_in,
        mapping=None,
        cols=None,
        handle_unknown="value",
        handle_missing="value",
        index_start=1,
    ):
        encoder = OrdinalEncoder(
            mapping=mapping,
            cols=cols,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            index_start=index_start,
        ).fit(X_in)
        return encoder.transform(X_in), encoder.mapping
