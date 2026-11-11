"""GPU gate: flops per byte, device round-trip cost, and the measured verdict.

`bench/analysis.md` does not exist in this repository (checked in the working
tree and in every commit), so the two counters it was meant to carry, bytes
moved per unit of work and device round-trip time, are computed by the script
itself, next to the CPU measurements they are compared against.

Run with `pixi run bench-gpu`. Every figure is a minimum of five runs, median of
three such minima, as in `bench/bench_parallel.py`.
"""

from __future__ import annotations

import os
import statistics
import time
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from bench_parallel import parallel_knobs, serial_knobs  # noqa: E402
from gpu_probe import gpu_free_mib, probe  # noqa: E402
from mojo_category_encoders import _lib  # noqa: E402

GATE_FLOPS_PER_BYTE = 2.0
GATE_FREE_MIB = 4000


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


def intensity():
    print("### Arithmetic intensity (measured bytes, operations per byte)")
    print("| kernel | work | bytes moved per call | ops per byte | gate |")
    print("|---|---|---:|---:|---|")

    keys = np.array([f"v{i:07d}" for i in range(60_000)], dtype=object)
    values = np.array([[k, k[::-1], "same"] for k in keys], dtype=object)
    # Must match `_lib.hash_md5`: packed UTF-8 blob, per-key start and length,
    # per-key bucket out, plus the 1M-row code and counts buffers that travel
    # in the same call to the public API.
    flat = values.ravel(order="C")
    encoded = [str(value).encode("utf-8") for value in flat]
    unique = list(dict.fromkeys(encoded))
    blob = sum(len(value) for value in unique)
    keys_bytes = blob + 8 * len(unique) + 8 * len(unique)
    rows, columns = 1_048_576, 1
    payload = keys_bytes + 8 * rows * columns + 8 * rows * 32
    # 13 ops per key per 4 bytes of block: the MD5 round body (4 boolean ops,
    # 3 additions) runs 64 times over a 64-byte block, plus 4 state adds.
    md5_ops = 13 * 64 * sum(max(1, (len(v) + 9 + 63) // 64) for v in unique)
    print(
        f"| MD5 hash | 60k keys + 1M x 1 accumulate | {payload} "
        f"| {md5_ops / payload:.2f} | {'candidate' if md5_ops / payload >= GATE_FLOPS_PER_BYTE else 'CPU only: memory-bound'} |"
    )

    lookup = np.arange(1000, dtype=np.float64)
    for n in (4_194_304,):
        payload = 8 * n + 8 * lookup.size + 8 * n
        print(
            f"| ordinal apply | {n} elements | {payload} | {3 * n / payload:.2f} "
            f"| {'candidate' if 3 * n / payload >= GATE_FLOPS_PER_BYTE else 'CPU only: memory-bound'} |"
        )
    rows = 1_048_576
    payload = 8 * rows + 8 * 32 + 8 * rows * 32
    print(
        f"| hash accumulate | 1M rows x 1, 32 buckets | {payload} | {2 * rows / payload:.2f} "
        f"| {'candidate' if 2 * rows / payload >= GATE_FLOPS_PER_BYTE else 'CPU only: memory-bound'} |"
    )
    print()


def cpu_times():
    print("### CPU kernel time at real caller sizes (ms)")
    print("| kernel | size | serial | parallel 4w |")
    print("|---|---:|---:|---:|")
    keys = [f"v{i:07d}" for i in range(60_000)]
    values = np.array([[k, k[::-1], "same"] for k in keys], dtype=object)
    small_alphabet = np.array([[f"c{i % 1000}"] for i in range(1_048_576)], dtype=object)
    results = {}
    for label, data in (("60k distinct keys", values), ("1M rows, 1k keys", small_alphabet)):
        with serial_knobs():
            serial = measure(lambda: _lib.hash_md5(data, 32))
        with parallel_knobs(4):
            par = measure(lambda: _lib.hash_md5(data, 32))
        results[label] = (serial, par)
        print(f"| hash_md5 | {label} | {serial * 1e3:.1f} | {par * 1e3:.1f} |")
    lookup = np.arange(1000, dtype=np.float64)
    for n in (1_048_576, 4_194_304, 8_388_608):
        positions = np.arange(n, dtype=np.int64) % 1000
        with serial_knobs():
            serial = measure(lambda: _lib.ordinal_apply(positions, lookup, -1.0, -2.0))
        with parallel_knobs(4):
            par = measure(lambda: _lib.ordinal_apply(positions, lookup, -1.0, -2.0))
        results[f"ordinal_apply {n}"] = (serial, par)
        print(f"| ordinal_apply | {n} | {serial * 1e3:.1f} | {par * 1e3:.1f} |")
    print()
    return results


def main():
    load = open("/proc/loadavg").read().split()[0]
    print(f"host: {os.cpu_count()} logical CPUs, load average {load}")
    free = gpu_free_mib()
    print(f"gpu: {free[0] if free else 0} MiB free" if free else "gpu: none visible")
    print(f"gate: need >= {GATE_FREE_MIB} MiB free and >= {GATE_FLOPS_PER_BYTE} ops/byte")
    print()
    intensity()
    times = cpu_times()
    trip = probe()
    if trip is None:
        print("round trip: no usable GPU device, GPU path skipped")
        return
    print("### Device round trip, fresh 512 KiB copy in and out per run (ms)")
    print("| copy in | launch call | copy out | round trip |")
    print("|---:|---:|---:|---:|")
    print(
        f"| {trip['copy_in_ms']:.3f} | {trip['launch_call_ms']:.3f} "
        f"| {trip['copy_out_ms']:.3f} | {trip['round_trip_ms']:.3f} |"
    )
    print()
    serial_md5, _ = times["60k distinct keys"]
    print(
        f"Round trip for the MD5 call: {trip['round_trip_ms']:.2f}ms against "
        f"{serial_md5 * 1e3:.1f}ms of CPU work at the measured caller size "
        f"({trip['round_trip_ms'] / (serial_md5 * 1e3):.3f}x of the CPU time before "
        "any hashing happens)."
    )


if __name__ == "__main__":
    main()
