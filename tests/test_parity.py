from __future__ import annotations

import hashlib

import category_encoders as upstream
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import mojo_category_encoders as mojo
from mojo_category_encoders import _lib


def assert_parity(ours, theirs, X, y=None):
    ours.fit(X, y)
    theirs.fit(X, y)
    got = ours.transform(X)
    expected = theirs.transform(X)
    if isinstance(expected, pd.DataFrame):
        assert_frame_equal(got, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    elif np.asarray(expected).dtype.kind in "biufc":
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, expected)
    return ours, theirs


@pytest.fixture
def mixed():
    return pd.DataFrame(
        {
            "city": ["Oslo", "Lima", None, "Oslo", "東京", np.nan, "Lima"],
            "tier": ["a", "b", "a", None, "c", "a", "c"],
            "amount": [1.5, 2.0, 3.5, 4.0, 2.5, 1.0, 9.0],
        },
        index=[11, 13, 17, 19, 23, 29, 31],
    )


def test_ordinal_basic_parity(mixed):
    assert_parity(
        mojo.OrdinalEncoder(cols=["city", "tier"]),
        upstream.OrdinalEncoder(cols=["city", "tier"]),
        mixed,
    )


def test_ordinal_default_columns_and_index_start(mixed):
    assert_parity(
        mojo.OrdinalEncoder(index_start=0),
        upstream.OrdinalEncoder(index_start=0),
        mixed,
    )


def test_ordinal_ordered_categorical_parity():
    X = pd.DataFrame(
        {"grade": pd.Categorical(["medium", "low", "high"], ["low", "medium", "high"], ordered=True)}
    )
    assert_parity(mojo.OrdinalEncoder(), upstream.OrdinalEncoder(), X)


def test_ordinal_supplied_mapping_parity():
    X = pd.DataFrame({"grade": ["low", "high", None, "medium"]})
    mapping = [{"col": "grade", "mapping": {"low": 10, "medium": 20, "high": 30, np.nan: 99}}]
    assert_parity(
        mojo.OrdinalEncoder(mapping=mapping),
        upstream.OrdinalEncoder(mapping=mapping),
        X,
    )


@pytest.mark.parametrize("missing", [None, -2, 99])
@pytest.mark.parametrize("missing_strategy", ["value", "return_nan"])
def test_ordinal_supplied_mapping_missing_parity(missing, missing_strategy):
    X = pd.DataFrame({"grade": ["low", None]})
    values = {"low": 10}
    if missing is not None:
        values[np.nan] = missing
    mapping = [{"col": "grade", "mapping": values}]
    ours = mojo.OrdinalEncoder(
        mapping=mapping, handle_missing=missing_strategy
    ).fit(X)
    theirs = upstream.OrdinalEncoder(
        mapping=mapping, handle_missing=missing_strategy
    ).fit(X)
    assert_frame_equal(ours.transform(X), theirs.transform(X), check_dtype=False)


@pytest.mark.parametrize("strategy", ["value", "return_nan"])
def test_ordinal_unknown_policy_parity(strategy):
    train = pd.DataFrame({"x": ["a", "b", "a"]})
    test = pd.DataFrame({"x": ["b", "new", None]})
    ours = mojo.OrdinalEncoder(handle_unknown=strategy).fit(train)
    theirs = upstream.OrdinalEncoder(handle_unknown=strategy).fit(train)
    assert_frame_equal(
        ours.transform(test), theirs.transform(test), check_dtype=False
    )


def test_ordinal_unknown_error_parity():
    train = pd.DataFrame({"x": ["a", "b"]})
    test = pd.DataFrame({"x": ["new"]})
    for encoder in (
        mojo.OrdinalEncoder(handle_unknown="error").fit(train),
        upstream.OrdinalEncoder(handle_unknown="error").fit(train),
    ):
        with pytest.raises(ValueError):
            encoder.transform(test)


@pytest.mark.parametrize("strategy", ["value", "return_nan"])
def test_ordinal_missing_policy_parity(strategy):
    train = pd.DataFrame({"x": ["a", "b", None]})
    ours, theirs = assert_parity(
        mojo.OrdinalEncoder(handle_missing=strategy),
        upstream.OrdinalEncoder(handle_missing=strategy),
        train,
    )
    assert_frame_equal(
        ours.transform(pd.DataFrame({"x": [None, "a"]})),
        theirs.transform(pd.DataFrame({"x": [None, "a"]})),
        check_dtype=False,
    )


def test_ordinal_missing_error():
    with pytest.raises(ValueError):
        mojo.OrdinalEncoder(handle_missing="error").fit(pd.DataFrame({"x": ["a", None]}))


def test_ordinal_inverse_transform_parity():
    X = pd.DataFrame({"x": ["a", "b", "a"], "n": [1, 2, 3]})
    ours = mojo.OrdinalEncoder().fit(X)
    expected = X
    assert_frame_equal(ours.inverse_transform(ours.transform(X)), expected)


def test_ordinal_array_return_and_feature_names():
    X = np.array([["a", "x"], ["b", "x"], ["a", "x"]], dtype=object)
    ours, theirs = assert_parity(
        mojo.OrdinalEncoder(cols=[0], return_df=False, drop_invariant=True),
        upstream.OrdinalEncoder(cols=[0], return_df=False, drop_invariant=True),
        X,
    )
    assert ours.get_feature_names_out().tolist() == theirs.get_feature_names_out().tolist()


def test_feature_names_require_fit():
    with pytest.raises(NotFittedError):
        mojo.OrdinalEncoder().get_feature_names_out()


def test_hashing_basic_md5_parity(mixed):
    assert_parity(
        mojo.HashingEncoder(cols=["city", "tier"], n_components=7, max_process=1),
        upstream.HashingEncoder(cols=["city", "tier"], n_components=7, max_process=1),
        mixed,
    )


def test_hashing_none_nan_unicode_and_long_strings():
    X = pd.DataFrame(
        {
            "x": [None, np.nan, "", "東京", "x" * 55, "x" * 56, "x" * 129],
            "y": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    assert_parity(
        mojo.HashingEncoder(cols=["x"], n_components=31, max_process=1),
        upstream.HashingEncoder(cols=["x"], n_components=31, max_process=1),
        X,
    )


@pytest.mark.parametrize("components", [1, 2, 8, 37])
def test_hashing_component_counts(components):
    X = pd.DataFrame({"a": ["x", "y", "z"], "b": ["q", "q", None]})
    ours, _ = assert_parity(
        mojo.HashingEncoder(cols=["a", "b"], n_components=components, max_process=1),
        upstream.HashingEncoder(cols=["a", "b"], n_components=components, max_process=1),
        X,
    )
    assert ours.transform(X).shape == (3, components)


def test_hashing_sha1_fallback_parity():
    X = pd.DataFrame({"x": ["alpha", "beta", None, np.nan]})
    assert_parity(
        mojo.HashingEncoder(cols=["x"], n_components=9, hash_method="sha1", max_process=1),
        upstream.HashingEncoder(cols=["x"], n_components=9, hash_method="sha1", max_process=1),
        X,
    )


def test_hashing_rejects_hashlib_names_without_public_constructors():
    method = next(
        (name for name in hashlib.algorithms_available if not hasattr(hashlib, name)),
        None,
    )
    if method is None:
        pytest.skip("this hashlib build exposes constructors for every algorithm")
    with pytest.raises(ValueError, match="not available"):
        mojo.HashingEncoder(hash_method=method).fit(pd.DataFrame({"x": ["alpha"]}))


def test_hash_chunk_matches_published_hashlib_definition():
    values = np.array([["a", "b"], [None, "東京"]], dtype=object)
    got = mojo.HashingEncoder.hash_chunk("md5", values, 13)
    expected = np.zeros((2, 13), dtype=int)
    for row, items in enumerate(values):
        for item in items:
            if item is not None:
                bucket = int.from_bytes(hashlib.md5(str(item).encode()).digest(), "big") % 13
                expected[row, bucket] += 1
    np.testing.assert_array_equal(got, expected)


def test_hashing_serial_and_parallel_thresholds(monkeypatch):
    values = np.array(
        [[f"value_{row}", f"other_{row % 3}"] for row in range(19)],
        dtype=object,
    )
    monkeypatch.setattr(_lib, "_HASH_PARALLEL_THRESHOLD", 10_000)
    monkeypatch.setattr(_lib, "_ACCUMULATE_PARALLEL_THRESHOLD", 10_000)
    serial = _lib.hash_md5(values, 17)
    monkeypatch.setattr(_lib, "_HASH_PARALLEL_THRESHOLD", 4)
    monkeypatch.setattr(_lib, "_ACCUMULATE_PARALLEL_THRESHOLD", 5)
    parallel = _lib.hash_md5(values, 17)
    np.testing.assert_array_equal(parallel, serial)


def test_native_hash_empty_inputs_do_not_cross_null_pointers():
    empty_rows = _lib.hash_md5(np.empty((0, 2), dtype=object), 3)
    all_missing = _lib.hash_md5(np.full((4, 2), None, dtype=object), 3)
    assert empty_rows.shape == (0, 3)
    np.testing.assert_array_equal(all_missing, np.zeros((4, 3), dtype=np.int64))


@pytest.mark.parametrize("components", [0, -1])
def test_hashing_rejects_nonpositive_component_count(components):
    with pytest.raises(ValueError):
        mojo.HashingEncoder(n_components=components).fit(pd.DataFrame({"x": ["a"]}))


def test_hashing_all_columns_return_array_and_invariants():
    X = pd.DataFrame({"a": ["x"] * 4, "b": [1] * 4})
    ours, theirs = assert_parity(
        mojo.HashingEncoder(
            cols="all", n_components=4, max_process=1, return_df=False, drop_invariant=True
        ),
        upstream.HashingEncoder(
            cols="all", n_components=4, max_process=1, return_df=False, drop_invariant=True
        ),
        X,
    )
    assert ours.get_feature_names_out().tolist() == theirs.get_feature_names_out().tolist()


def test_target_continuous_parity(mixed):
    y = pd.Series([0.2, 0.9, 0.5, 0.1, 1.0, 0.4, 0.8], index=mixed.index)
    assert_parity(
        mojo.TargetEncoder(cols=["city", "tier"], min_samples_leaf=2, smoothing=3),
        upstream.TargetEncoder(cols=["city", "tier"], min_samples_leaf=2, smoothing=3),
        mixed,
        y,
    )


def test_target_string_label_parity():
    X = pd.DataFrame({"x": ["a", "a", "b", "c", "c", "c"]})
    y = np.array(["no", "yes", "yes", "no", "yes", "yes"])
    assert_parity(
        mojo.TargetEncoder(cols=["x"], min_samples_leaf=1, smoothing=2),
        upstream.TargetEncoder(cols=["x"], min_samples_leaf=1, smoothing=2),
        X,
        y,
    )


@pytest.mark.parametrize("strategy", ["value", "return_nan"])
def test_target_unknown_policy_parity(strategy):
    X = pd.DataFrame({"x": ["a", "a", "b", "c"]})
    y = [0.0, 1.0, 1.0, 0.0]
    test = pd.DataFrame({"x": ["a", "new", None]})
    ours = mojo.TargetEncoder(cols=["x"], handle_unknown=strategy).fit(X, y)
    theirs = upstream.TargetEncoder(cols=["x"], handle_unknown=strategy).fit(X, y)
    assert_frame_equal(
        ours.transform(test), theirs.transform(test), check_dtype=False, rtol=1e-12, atol=1e-12
    )


def test_target_unknown_error_parity():
    X = pd.DataFrame({"x": ["a", "b"]})
    y = [0, 1]
    for encoder in (
        mojo.TargetEncoder(handle_unknown="error").fit(X, y),
        upstream.TargetEncoder(handle_unknown="error").fit(X, y),
    ):
        with pytest.raises(ValueError):
            encoder.transform(pd.DataFrame({"x": ["new"]}))


@pytest.mark.parametrize("strategy", ["value", "return_nan"])
def test_target_missing_policy_parity(strategy):
    X = pd.DataFrame({"x": ["a", None, "a", "b", None]})
    y = [0.0, 1.0, 0.5, 1.0, 0.0]
    assert_parity(
        mojo.TargetEncoder(handle_missing=strategy, min_samples_leaf=1, smoothing=2),
        upstream.TargetEncoder(handle_missing=strategy, min_samples_leaf=1, smoothing=2),
        X,
        y,
    )


@pytest.mark.parametrize(
    ("leaf", "smooth"), [(1, 0.5), (20, 10.0), (100, 1000.0)]
)
def test_target_regularization_parity(leaf, smooth):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.choice(["a", "b", "c", "d"], 1001)})
    y = rng.normal(size=1001)
    assert_parity(
        mojo.TargetEncoder(min_samples_leaf=leaf, smoothing=smooth),
        upstream.TargetEncoder(min_samples_leaf=leaf, smoothing=smooth),
        X,
        y,
    )


def test_target_array_return_invariant_and_names():
    X = np.array([["same", "a"], ["same", "b"], ["same", "a"]], dtype=object)
    y = [0.0, 1.0, 0.0]
    ours, theirs = assert_parity(
        mojo.TargetEncoder(cols=[0], return_df=False, drop_invariant=True),
        upstream.TargetEncoder(cols=[0], return_df=False, drop_invariant=True),
        X,
        y,
    )
    assert ours.get_feature_names_out().tolist() == theirs.get_feature_names_out().tolist()


def test_target_apply_simd_tail():
    codes = np.array(
        [-2, -1, 0, 1, 2, 3, 4, 5, -3, 2, 0, 5, 4, 3, 1, -2, -1, 2, 5],
        dtype=np.int64,
    )
    mapping = np.arange(8, dtype=np.float64) + 0.25
    got = _lib.target_apply(codes, mapping, -99.0)
    expected = np.array(
        [mapping[code + 2] if 0 <= code + 2 < len(mapping) else -99.0 for code in codes]
    )
    np.testing.assert_array_equal(got, expected)


def test_apply_serial_and_parallel_thresholds(monkeypatch):
    codes = np.tile(np.arange(-3, 11, dtype=np.int64), 15001)
    mapping = np.arange(13, dtype=np.float64) + 0.5
    monkeypatch.setattr(_lib, "_APPLY_PARALLEL_THRESHOLD", codes.size + 1)
    serial_target = _lib.target_apply(codes, mapping, -7.0)
    serial_ordinal = _lib.ordinal_apply(codes, mapping, -8.0, -9.0)
    monkeypatch.setattr(_lib, "_APPLY_PARALLEL_THRESHOLD", 1)
    np.testing.assert_array_equal(
        _lib.target_apply(codes, mapping, -7.0), serial_target
    )
    np.testing.assert_array_equal(
        _lib.ordinal_apply(codes, mapping, -8.0, -9.0), serial_ordinal
    )


def test_target_stats_simd_initialization_tail():
    codes = np.array([-3, -2, -1, 0, 0, 3, 10, 11], dtype=np.int64)
    target = np.arange(codes.size, dtype=np.float64) + 0.25
    counts, sums = _lib.target_stats(codes, target, 13)
    expected_counts = np.zeros(13, dtype=np.int64)
    expected_sums = np.zeros(13, dtype=np.float64)
    for code, value in zip(codes, target, strict=True):
        group = int(code) + 2
        if 0 <= group < 13:
            expected_counts[group] += 1
            expected_sums[group] += value
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(sums, expected_sums)


def test_transform_does_not_modify_input(mixed):
    original = mixed.copy(deep=True)
    mojo.OrdinalEncoder(cols=["city", "tier"]).fit(mixed).transform(mixed)
    assert_frame_equal(mixed, original)


def test_native_wrappers_validate_shapes_ranges_and_empty_buffers():
    assert _lib.ordinal_apply([], [], -1.0, -2.0).shape == (0,)
    np.testing.assert_array_equal(
        _lib.ordinal_apply(np.array([-1, -2]), [], -3.0, -4.0),
        np.array([-3.0, -4.0]),
    )
    counts, sums = _lib.target_stats([], [], 3)
    np.testing.assert_array_equal(counts, np.zeros(3, dtype=np.int64))
    np.testing.assert_array_equal(sums, np.zeros(3, dtype=np.float64))
    assert _lib.target_apply([], [], 1.0).shape == (0,)
    np.testing.assert_array_equal(_lib.target_apply([-1, 0], [], 7.0), [7.0, 7.0])
    with pytest.raises(ValueError):
        _lib.target_stats([0], [], 3)
    with pytest.raises(TypeError):
        _lib.target_apply([0.5], [1.0], 0.0)
    with pytest.raises(TypeError):
        _lib.target_apply([0], np.array([1 + 2j]), 0.0)
    with pytest.raises(OverflowError):
        _lib.target_apply(np.array([2**63], dtype=np.uint64), [1.0], 0.0)


@pytest.mark.parametrize(
    "encoder",
    [
        mojo.OrdinalEncoder(cols=["x"]),
        mojo.HashingEncoder(cols=["x"], max_process=1),
        mojo.TargetEncoder(cols=["x"]),
    ],
)
def test_sklearn_clone_preserves_parameters(encoder):
    copied = clone(encoder)
    assert copied.get_params() == encoder.get_params()


def test_hierarchy_is_explicitly_outside_subset():
    with pytest.raises(NotImplementedError):
        mojo.TargetEncoder(cols=["x"], hierarchy={"x": {"g": ("a", "b")}}).fit(
            pd.DataFrame({"x": ["a", "b"]}), [0, 1]
        )


@pytest.mark.parametrize(
    ("ours", "theirs", "y"),
    [
        (mojo.OrdinalEncoder(), upstream.OrdinalEncoder(), None),
        (
            mojo.HashingEncoder(max_process=1),
            upstream.HashingEncoder(max_process=1),
            None,
        ),
        (
            mojo.TargetEncoder(min_samples_leaf=1, smoothing=2),
            upstream.TargetEncoder(min_samples_leaf=1, smoothing=2),
            [0, 1, 0],
        ),
    ],
)
def test_fit_transform_and_input_feature_names(ours, theirs, y):
    X = pd.DataFrame({"x": ["a", "b", "a"], "number": [1, 2, 3]})
    got = ours.fit_transform(X, y)
    expected = theirs.fit_transform(X, y)
    assert_frame_equal(got, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        ours.get_feature_names_in(), theirs.get_feature_names_in()
    )
