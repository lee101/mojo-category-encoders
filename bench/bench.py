"""Benchmarks against category-encoders 2.10.0 on identical data."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import category_encoders as upstream
import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_category_encoders as mojo  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def categorical_data(rows: int, columns: int, cardinality: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {
        f"c{i}": np.array([f"value_{v}" for v in rng.integers(0, cardinality, rows)], dtype=object)
        for i in range(columns)
    }
    data["number"] = rng.normal(size=rows)
    return pd.DataFrame(data), rng.normal(size=rows)


CASES = []


def case(name):
    def decorate(fn):
        CASES.append((name, fn))
        return fn
    return decorate


@case("OrdinalEncoder.transform (1M x 4, k=1k)")
def ordinal_case():
    X, _ = categorical_data(1_000_000, 4, 1_000)
    ours = mojo.OrdinalEncoder().fit(X)
    theirs = upstream.OrdinalEncoder().fit(X)
    return lambda: ours.transform(X), lambda: theirs.transform(X)


@case("TargetEncoder.fit_transform (1M x 3, k=1k)")
def target_fit_case():
    X, y = categorical_data(1_000_000, 3, 1_000)
    return (
        lambda: mojo.TargetEncoder(min_samples_leaf=20, smoothing=10).fit_transform(X, y),
        lambda: upstream.TargetEncoder(min_samples_leaf=20, smoothing=10).fit_transform(X, y),
    )


@case("TargetEncoder.transform (1M x 3, k=1k)")
def target_transform_case():
    X, y = categorical_data(1_000_000, 3, 1_000)
    ours = mojo.TargetEncoder().fit(X, y)
    theirs = upstream.TargetEncoder().fit(X, y)
    return lambda: ours.transform(X), lambda: theirs.transform(X)


@case("HashingEncoder MD5 (250k x 4, 32 buckets)")
def hashing_case():
    X, _ = categorical_data(250_000, 4, 50_000)
    cols = [f"c{i}" for i in range(4)]
    return (
        lambda: mojo.HashingEncoder(
            cols=cols, n_components=32, max_process=1
        ).fit_transform(X),
        lambda: upstream.HashingEncoder(
            cols=cols, n_components=32, max_process=1
        ).fit_transform(X),
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(
        f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; "
        f"Python {platform.python_version()}; category-encoders {upstream.__version__}"
    )
    print()
    print("| case | Mojo (ms) | upstream (ms) | upstream / Mojo | result |")
    print("|---|---:|---:|---:|---|")
    for name, setup in CASES:
        ours, theirs = setup()
        ours()
        theirs()
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        ratio = upstream_time / mojo_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.1f} | {upstream_time * 1000:.1f} | "
            f"{ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
