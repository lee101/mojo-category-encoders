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
| OrdinalEncoder.transform (1M x 4, k=1k) | 750.6 | 930.7 | 1.24x | faster |
| TargetEncoder.fit_transform (1M x 3, k=1k) | 3799.9 | 3248.0 | 0.85x | slower |
| TargetEncoder.transform (1M x 3, k=1k) | 694.0 | 826.0 | 1.19x | faster |
| HashingEncoder MD5 (250k x 4, 32 buckets) | 1925.3 | 5028.8 | 2.61x | faster |

These are end-to-end estimator timings, not isolated native-kernel timings.
Performance varies by workload and system load; the table reports the final
run directly, including the slower target fit case.

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
Hash output initialization is also SIMD, while independent hashing and row
accumulation switch to host-threaded native slices only above their measured
size thresholds. The MD5 implementation computes the exact 128-bit digest
bucket required by upstream without allocating per row. All exported functions
live in `src/kernels.mojo`, so one `mojo build --emit shared-lib` invocation creates
`dist/libmojo-category-encoders.so`.
