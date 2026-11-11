"""ctypes bridge to the Mojo kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_CATEGORY_ENCODERS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-category-encoders.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mce_hash_md5_buckets": ([I] * 8, None),
    "mce_hash_md5_buckets_parallel": ([I] * 10, None),
    "mce_hash_accumulate": ([I] * 6, None),
    "mce_hash_accumulate_parallel": ([I] * 8, None),
    "mce_ordinal_apply": ([I, I, I, I, I, F, F, I], None),
    "mce_ordinal_apply_parallel": ([I, I, I, I, I, F, F, I, I], None),
    "mce_target_stats": ([I, I, I, I, I, I], None),
    "mce_target_apply": ([I, I, I, I, I, F, I], None),
    "mce_target_apply_parallel": ([I, I, I, I, I, F, I, I], None),
}

_lib: ctypes.CDLL | None = None

# Measured 2026-11-11 on a 72-core Xeon E5-2697 v4 shared with unrelated work
# (load average 13-26). Numbers from `pixi run bench-parallel`; see README
# "Parallelism" for the full before/after and the 1/2/4/8-thread sweep.
#
# A `max.algorithm.parallelize` call with >=2 workers costs a flat ~5ms before
# any work runs (5.03ms with one no-op chunk, 4.74-6.9ms with 4-256). A size
# threshold therefore has to sit where the serial kernel costs far more than
# that, and callers below it stay on the plain serial export.
#
# Distinct keys hashed: parallel wins from ~8k keys (1.1x), 2.1x at 16k and
# 2.9x at 32k, so 16 384 with a 2x margin ships (starts at 32 768 keys).
_HASH_PARALLEL_THRESHOLD = 16384
_HASH_PARALLEL_MARGIN = 2
# Rows accumulated: the earlier 65 536 threshold lost to serial at every size
# that was measured through the public API (265ms vs 267ms serial at 1M rows).
_ACCUMULATE_PARALLEL_THRESHOLD = 2**36
_ACCUMULATE_PARALLEL_MARGIN = 2
# Elements applied: serial costs 0.41ms at 262k and 1.78ms at 1M, both below
# the ~5ms launch cost; parallel only wins at ~4.2M (1.55x at 4M, 1.78x at 8M).
# 8 388 608 with a 2x margin ships (starts at 16 777 216 elements = 128 MB).
_APPLY_PARALLEL_THRESHOLD = 8388608
_APPLY_PARALLEL_MARGIN = 2
# Per-worker chunk. Below two chunks there is nothing to share out.
_CHUNK = 65536
# Hard cap from the parallel-phase rules. The hash kernel kept scaling to 8
# threads (4.8x at 262k keys) but the host is shared with production and the
# apply kernel was past its best by 4, so the cap stays at 4.
_MAX_WORKERS = 4


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError(f"Mojo library not found at {LIB}; run `pixi run build`")
        _lib = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty buffer")
    return address


def _int64_buffer(array, name: str) -> np.ndarray:
    source = np.asarray(array)
    if not source.size:
        return np.ascontiguousarray(source, dtype=np.int64)
    if source.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers, got {source.dtype}")
    if source.dtype.kind == "u" and source.size:
        if int(source.max()) > np.iinfo(np.int64).max:
            raise OverflowError(f"{name} contains a value outside the int64 range")
    return np.ascontiguousarray(source, dtype=np.int64)


def _positive_int64(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if result > np.iinfo(np.int64).max:
        raise OverflowError(f"{name} exceeds the int64 range")
    return result


def _float64_buffer(array, name: str) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numbers, got {source.dtype}")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError(f"{name} cannot be narrowed from {source.dtype} to float64")
    return np.ascontiguousarray(source, dtype=np.float64)


_MD5_SHIFTS = np.array(
    [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ],
    dtype=np.uint32,
)
_MD5_CONSTANTS = np.array(
    [
        0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
        0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
        0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
        0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
        0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
        0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
        0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
        0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
        0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C,
        0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
        0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05,
        0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
        0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039,
        0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
        0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1,
        0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391,
    ],
    dtype=np.uint32,
)


def hash_md5(values: np.ndarray, components: int) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    if values.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got shape {values.shape}")
    components = _positive_int64(components, "components")
    if values.shape[0] > np.iinfo(np.int64).max // components:
        raise OverflowError("output size exceeds the native int64 range")
    flat = values.ravel(order="C")
    strings = np.asarray(
        [None if value is None else str(value) for value in flat], dtype=object
    )
    codes, unique = pd.factorize(strings, use_na_sentinel=True)
    encoded = [value.encode("utf-8") for value in unique]
    lengths = np.fromiter((len(value) for value in encoded), dtype=np.int64)
    starts = np.empty(len(unique), dtype=np.int64)
    cursor = 0
    for i, value in enumerate(encoded):
        starts[i] = cursor
        cursor += len(value)
    blob = np.frombuffer(b"".join(encoded) or b"\0", dtype=np.uint8)
    buckets = np.empty(len(unique), dtype=np.int64)
    result = np.empty((values.shape[0], components), dtype=np.int64)
    if values.shape[0] == 0:
        return result
    if not len(unique):
        result.fill(0)
        return result
    native = lib()
    rows = values.shape[0]
    key_workers = _workers(len(unique), _HASH_PARALLEL_THRESHOLD, _HASH_PARALLEL_MARGIN)
    row_workers = _workers(rows, _ACCUMULATE_PARALLEL_THRESHOLD, _ACCUMULATE_PARALLEL_MARGIN)

    if key_workers > 1:
        native.mce_hash_md5_buckets_parallel(
            addr(blob), addr(starts), addr(lengths),
            addr(_MD5_SHIFTS), addr(_MD5_CONSTANTS), addr(buckets),
            len(unique), components, key_workers, _CHUNK,
        )
    else:
        native.mce_hash_md5_buckets(
            addr(blob), addr(starts), addr(lengths),
            addr(_MD5_SHIFTS), addr(_MD5_CONSTANTS), addr(buckets),
            len(unique), components,
        )

    codes_2d = codes.reshape(values.shape)
    if row_workers > 1:
        native.mce_hash_accumulate_parallel(
            addr(codes_2d), addr(buckets), addr(result),
            rows, values.shape[1], components, row_workers, _CHUNK,
        )
    else:
        native.mce_hash_accumulate(
            addr(codes_2d), addr(buckets), addr(result),
            rows, values.shape[1], components,
        )
    return result


def _workers(size: int, threshold: int, margin: int) -> int:
    """Worker count for one phase: 1 below the measured, margin-padded threshold."""
    if size >= max(threshold * margin, 2 * _CHUNK):
        return _MAX_WORKERS
    return 1


def ordinal_apply(
    positions: np.ndarray, lookup: np.ndarray, unknown: float, missing: float
) -> np.ndarray:
    positions = _int64_buffer(positions, "positions")
    lookup = _float64_buffer(lookup, "lookup")
    result = np.empty(positions.size, dtype=np.float64)
    if not positions.size:
        return result
    native_lookup = lookup if lookup.size else np.zeros(1, dtype=np.float64)
    workers = _workers(positions.size, _APPLY_PARALLEL_THRESHOLD, _APPLY_PARALLEL_MARGIN)
    if workers > 1:
        lib().mce_ordinal_apply_parallel(
            addr(positions), addr(native_lookup), addr(result), positions.size, lookup.size,
            unknown, missing, workers, _CHUNK,
        )
    else:
        lib().mce_ordinal_apply(
            addr(positions), addr(native_lookup), addr(result), positions.size, lookup.size,
            unknown, missing, _APPLY_PARALLEL_THRESHOLD,
        )
    return result


def target_stats(codes: np.ndarray, target: np.ndarray, groups: int):
    codes = _int64_buffer(codes, "codes")
    target = _float64_buffer(target, "target")
    groups = _positive_int64(groups, "groups")
    if codes.ndim != 1 or target.ndim != 1:
        raise ValueError("codes and target must be one-dimensional")
    if codes.size != target.size:
        raise ValueError("codes and target must have the same length")
    counts = np.empty(groups, dtype=np.int64)
    sums = np.empty(groups, dtype=np.float64)
    if not codes.size:
        counts.fill(0)
        sums.fill(0.0)
        return counts, sums
    lib().mce_target_stats(
        addr(codes), addr(target), addr(counts), addr(sums), codes.size, groups
    )
    return counts, sums


def target_apply(codes: np.ndarray, mapping: np.ndarray, default: float) -> np.ndarray:
    codes = _int64_buffer(codes, "codes")
    mapping = _float64_buffer(mapping, "mapping")
    if codes.ndim != 1 or mapping.ndim != 1:
        raise ValueError("codes and mapping must be one-dimensional")
    result = np.empty(codes.size, dtype=np.float64)
    if not codes.size:
        return result
    native_mapping = mapping if mapping.size else np.zeros(1, dtype=np.float64)
    workers = _workers(codes.size, _APPLY_PARALLEL_THRESHOLD, _APPLY_PARALLEL_MARGIN)
    if workers > 1:
        lib().mce_target_apply_parallel(
            addr(codes), addr(native_mapping), addr(result), codes.size, mapping.size,
            default, workers, _CHUNK,
        )
    else:
        lib().mce_target_apply(
            addr(codes), addr(native_mapping), addr(result), codes.size, mapping.size, default,
            _APPLY_PARALLEL_THRESHOLD,
        )
    return result
