"""Serial vs parallel comparison for the shipped kernels and thresholds.

Uses only public module knobs (the threshold constants, `_MAX_WORKERS`) and the
real kernels, so it measures what ships. Recorded numbers in the README come
from this script. Run with `pixi run bench-parallel`.

The host is shared, so every figure is the minimum of five runs, median of
three such minima.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np  # noqa: E402

from mojo_category_encoders import _lib  # noqa: E402

_THRESHOLDS = (
    "_HASH_PARALLEL_THRESHOLD",
    "_ACCUMULATE_PARALLEL_THRESHOLD",
    "_APPLY_PARALLEL_THRESHOLD",
    "_MAX_WORKERS",
)


class knob:
    """Temporarily set lib knobs, restoring them afterwards."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.saved[name] = getattr(_lib, name)
            setattr(_lib, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(_lib, name, value)


def measure(fn):
    best = []
    for _ in range(3):
        runs = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - start)
        best.append(min(runs))
    return statistics.median(best)


def serial_knobs():
    return knob(
        _HASH_PARALLEL_THRESHOLD=10**12,
        _ACCUMULATE_PARALLEL_THRESHOLD=10**12,
        _APPLY_PARALLEL_THRESHOLD=10**12,
    )


def parallel_knobs(workers):
    return knob(
        _HASH_PARALLEL_THRESHOLD=1,
        _ACCUMULATE_PARALLEL_THRESHOLD=1,
        _APPLY_PARALLEL_THRESHOLD=1,
        _MAX_WORKERS=workers,
    )


def bench_hash():
    # 60k distinct 7-char keys: passes the shipped key threshold (32 768).
    keys = [f"v{i:07d}" for i in range(60_000)]
    values = np.array([[k, k[::-1], "same"] for k in keys], dtype=object)
    # 1M rows over only 1 000 distinct keys: hash pass stays serial under the
    # shipped threshold, accumulation is past 1 048 576 and runs parallel.
    small_alphabet = np.array(
        [[f"c{i % 1000}"] for i in range(1_048_576)], dtype=object
    )

    print("### hash_md5, accumulation-dominant: 1M rows x 1 col, 1k distinct keys (ms)")
    print("| path | hash pass (60k keys) | accumulate 1M rows |")
    print("|---|---:|---:|")
    with serial_knobs():
        serial_keys = measure(lambda: _lib.hash_md5(values, 32))
        serial_acc = measure(lambda: _lib.hash_md5(small_alphabet, 32))
    print(f"| serial | {serial_keys * 1e3:.1f} | {serial_acc * 1e3:.1f} |")
    for workers in (2, 4, 8):
        with parallel_knobs(workers):
            keyed = measure(lambda: _lib.hash_md5(values, 32))
            acc = measure(lambda: _lib.hash_md5(small_alphabet, 32))
        print(f"| {workers}w | {keyed * 1e3:.1f} | {acc * 1e3:.1f} |")


def bench_apply():
    lookup = np.arange(1000, dtype=np.float64)
    print()
    print("### ordinal_apply (ms per call)")
    print("| n | serial | 2w | 4w (cap) | 8w | 4w vs serial |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in (16_384, 65_536, 262_144, 1_048_576, 4_194_304, 8_388_608):
        positions = np.arange(n, dtype=np.int64) % 1000
        with serial_knobs():
            serial = measure(lambda: _lib.ordinal_apply(positions, lookup, -1.0, -2.0))
        times = {}
        for workers in (2, 4, 8):
            with parallel_knobs(workers):
                times[workers] = measure(lambda: _lib.ordinal_apply(positions, lookup, -1.0, -2.0))
        print(
            f"| {n} | {serial * 1e3:.3f} | {times[2] * 1e3:.3f} | "
            f"{times[4] * 1e3:.3f} | {times[8] * 1e3:.3f} | {serial / times[4]:.2f}x |"
        )


def main():
    load = open("/proc/loadavg").read().split()[0]
    print(f"host: {os.cpu_count()} logical CPUs, load average {load}")
    print()
    bench_hash()
    bench_apply()


if __name__ == "__main__":
    main()
