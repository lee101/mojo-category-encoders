# mojo-category-encoders

`mojo-category-encoders` is a Mojo implementation of the compute-heavy parts of
[`category-encoders`](https://contrib.scikit-learn.org/category_encoders/). It
provides estimator-style Python classes with the upstream names and constructor
signatures, backed by one compiled shared library.

The parity baseline is `category-encoders` 2.10.0. This repository is independent
and is not affiliated with that project.

## Coverage

The covered API is:

- `OrdinalEncoder`: learned or supplied mappings, ordered categoricals,
  `index_start`, inverse transform, missing/unknown policies, invariant dropping,
  DataFrame and NumPy output.
- `TargetEncoder`: continuous, binary, and label-encoded targets; logistic
  smoothing; missing/unknown policies; invariant dropping. Target aggregation
  and application run in Mojo.
- `HashingEncoder`: exact upstream MD5 buckets, including its `str(value)` UTF-8
  conversion, `None` behavior, Unicode, and multi-block inputs. Other algorithms
  with public constructors in Python's `hashlib` use a parity-preserving Python
  fallback.
- Common sklearn behavior: `fit`, `transform`, `fit_transform`, cloning,
  `get_feature_names_in`, and `get_feature_names_out`.

Not covered: hierarchical target encoding, encoders outside the three named
above, and upstream's multiprocessing implementation for hashing. `max_process`
and `process_creation_method` remain accepted for signature compatibility, but
the accelerated MD5 kernel is single-process. Non-MD5 hash methods are correct
but are not accelerated.

## Install

The repository pins the Mojo nightly used to build the shared library.

```bash
pixi install
pixi run build
```

Run the parity suite with:

```bash
pixi run test
```

## Usage

```python
import pandas as pd
from mojo_category_encoders import TargetEncoder

X = pd.DataFrame({"city": ["Oslo", "Lima", "Oslo", "Tokyo"]})
y = [1.0, 0.0, 0.8, 0.3]

encoder = TargetEncoder(
    cols=["city"],
    min_samples_leaf=1,
    smoothing=2,
).fit(X, y)

encoded = encoder.transform(pd.DataFrame({"city": ["Oslo", "Lima", "new"]}))
print(encoded)
```

The final row receives the training target mean, matching upstream's default
`handle_unknown="value"` behavior.

## Benchmarks

Measured on this machine with `pixi run bench`; values are the best of three
warmed runs from the final publication pass.
The machine was an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64, Python
3.13.14, with category-encoders 2.10.0. Hashing used `max_process=1` on both
sides so the comparison measures the kernels rather than process-pool startup.

| case | Mojo (ms) | upstream (ms) | upstream / Mojo | result |
|---|---:|---:|---:|---|
| OrdinalEncoder.transform (1M x 4, k=1k) | 531.8 | 686.8 | 1.29x | faster |
| TargetEncoder.fit_transform (1M x 3, k=1k) | 832.8 | 2664.0 | 3.20x | faster |
| TargetEncoder.transform (1M x 3, k=1k) | 353.3 | 618.2 | 1.75x | faster |
| HashingEncoder MD5 (250k x 4, 32 buckets) | 504.5 | 4068.0 | 8.06x | faster |

These are end-to-end estimator timings, not isolated native-kernel timings.
Performance varies by workload and system load; the table reports the final
run directly.

## Parallelism

`parallelize` from `max.algorithm` is used, never `std.algorithm` or
`std.threading` (those do not compile on the pinned toolchain). Callers below a
size threshold stay on the plain serial export, and the worker count is capped at
4. Thresholds are constants in `python/mojo_category_encoders/_lib.py`.

### What the launch costs

A `parallelize` call with two or more workers costs a flat ~5ms before any work
runs, on this host. Measured with no-op chunks: 5.03ms (1 chunk), 4.74ms (4),
5.96ms (256); 2, 4, 8 and 16 workers were all 5.3-6.1ms. That cost, not the work,
sets the threshold: only kernels whose serial form costs well over 5ms benefit.

### Before and after, per kernel

Measured 2026-11-11 on a 72-core Xeon E5-2697 v4, load average 13-26 (the host is
shared). `pixi run bench-parallel`.

| kernel | size | before | after | path before -> after |
|---|---:|---:|---:|---|
| MD5 hash pass, distinct keys | 32 768 | 30.2ms | 10.2ms (8w) | Python threads -> `parallelize` |
| MD5 hash pass, distinct keys | 262 144 | 282.6ms | 96.6ms (4w) | Python threads -> `parallelize` |
| MD5 row accumulation | 1M rows | 267ms | 265ms | Python threads -> `parallelize` |
| ordinal/target apply | 262 144 | 0.41ms | 0.41ms (serial) | Python threads removed |
| ordinal/target apply | 1 048 576 | 1.78ms | 1.78ms (serial) | Python threads removed |
| ordinal/target apply | 4 194 304 | 16.8ms | 10.8ms (4w) | Python threads -> `parallelize` |
| ordinal/target apply | 8 388 608 | 34.5ms | 19.4ms (4w) | Python threads -> `parallelize` |

### Thread-count sweep (1, 2, 4, cap)

| kernel | size | 1w | 2w | 4w | 8w |
|---|---:|---:|---:|---:|---:|
| ordinal apply | 4 194 304 | 16.8ms | 14.7ms | 10.8ms | 9.0ms |
| ordinal apply | 8 388 608 | 34.5ms | 25.4ms | 19.4ms | 12.7ms |
| MD5 hash pass | 262 144 keys | 282.6ms | 152.6ms | 96.6ms | 58.8ms |

The gain does **not** flatten at 4 for these two kernels when measured alone: 8
threads was still 1.2-1.5x better than 4. The cap stays at 4 because the host is
shared with production and the rules for this phase cap at 4 unless more is
proven on a quiet machine; it was not. The accumulation stage did flatten: 1.19x
at 4 threads and 1.19x at 8.

### What was reverted

- **Threaded application stages.** The previous code split the apply kernels
  across a `ThreadPoolExecutor` and switched them on at 262 144 elements. That
  *lost* to serial at every size real callers pass: 4-thread at 262k took 5.26ms
  against 0.41ms serial (0.08x), and at 1M 5.27ms against 1.78ms (0.34x). The
  threaded path is gone; the kernel now uses `parallelize` only from 16 777 216
  elements (an 8M measured crossover, doubled as margin).
- **Threaded row accumulation.** Its 65 536-row threshold was below the crossover.
  Through the public API it measured 265ms at 1M rows against 267ms serial, i.e.
  no win. It is disabled in the shipped thresholds. The export
  `mce_hash_accumulate_parallel` still exists and is tested for correctness, but
  the sizing logic was not settled, so it is not called.
- **Python-side threading for hashing.** Replaced by `parallelize`. It also made
  hashing slower end to end in this phase: the 250k x 4 hashing case went from
  456.5ms to 504.5ms. Same machine, same harness, load average roughly 25 before
  and 17 after, so treat that as a tentative regression rather than a clean
  measurement; the isolated hash pass is 2.6x faster, but the end-to-end case
  still projects slower than before.

Results are identical to the serial path. `tests/test_parity.py` compares serial
and parallel output on the same input for ordinal apply, target apply and MD5
hashing at 1, 2, 4 and 8 workers, and checks the parallel hash against public
`hashlib`.

No GPU path is included. The ordinal and target application kernels are
memory-bound indexed lookups with well under two operations per byte moved, so
host/device transfer cannot be justified. MD5 has higher arithmetic intensity,
but the optimized end-to-end CPU path is already 8.06x faster than upstream in
the measured workload; it was deliberately left alone rather than adding a GPU
path to a case already more than 5x ahead.

## How it works

Python handles estimator state, pandas indexes, arbitrary Python category
objects, and sklearn-compatible validation. Categories are converted to compact
contiguous buffers before crossing the FFI boundary. Mojo receives only integer
addresses through a C ABI, reconstructs pointers with mutable origins, and
writes into NumPy-owned output arrays.

Ordinal and target codes are contiguous `int64`; target values and mappings are
contiguous `float64`; hashing receives one packed UTF-8 byte buffer plus `int64`
offset and length arrays. Outputs use row-major layout. Target and ordinal
application use native-width SIMD loads and stores with scalar remainder loops.
Application buffers are split into 64K chunks; sizes below the measured threshold
stay on a serial export, and larger ones use `max.algorithm.parallelize` at up to
4 workers. The MD5 key pass is likewise parallel above its measured threshold.
All exported functions live in `src/kernels.mojo`, so one
`mojo build --emit shared-lib` invocation creates
`dist/libmojo-category-encoders.so`.
