"""Device round-trip probe for the GPU gate recorded in the README.

`python bench/bench_gpu.py` and `tests/test_gpu.py` both use this module.

It copies 512 KiB to the device, launches a no-op kernel and copies the result
back. That round trip is the fixed cost a GPU path would pay on every call,
whatever the arithmetic, so it is the floor any device kernel has to beat. Each
timed round trip copies fresh data in, launches, syncs and copies out, so it
measures device work rather than a no-op after a warm-up.

After the first call in a process the runtime already holds a device context, so
a probe that warmed up a `synchronize()` and then timed only the sync would
report microseconds. That is why the copies are inside the timed region.

CPU-only hosts: `probe()` returns None, and the test that uses it skips.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

_PROBE = r"""
from std.time import perf_counter_ns
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]

def noop_kernel(a: FPtr, n: Int64):
    var i = Int64(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        a.unsafe_store(i, a.unsafe_load(i))

def main() raises:
    var ctx = DeviceContext()
    var n = 65536
    var hbuf = ctx.enqueue_create_host_buffer[DType.float64](n)
    for i in range(n):
        hbuf[i] = Float64(i)
    var dev = ctx.enqueue_create_buffer[DType.float64](n)
    # One untimed round trip: context, buffers, and host pages created.
    ctx.enqueue_copy(dev, hbuf)
    ctx.enqueue_function[noop_kernel](dev, Int64(n), grid_dim=512, block_dim=128)
    ctx.synchronize()
    ctx.enqueue_copy(hbuf, dev)
    ctx.synchronize()
    var down = Float64(0)
    var up = Float64(0)
    var launch = Float64(0)
    var total = Float64(0)
    for _ in range(20):
        var t0 = UInt64(perf_counter_ns())
        ctx.enqueue_copy(dev, hbuf)
        ctx.synchronize()
        var t = UInt64(perf_counter_ns())
        down += Float64(t.value() - t0.value()) / 1e6

        t0 = UInt64(perf_counter_ns())
        ctx.enqueue_function[noop_kernel](dev, Int64(n), grid_dim=512, block_dim=128)
        t = UInt64(perf_counter_ns())
        launch += Float64(t.value() - t0.value()) / 1e6

        t0 = UInt64(perf_counter_ns())
        ctx.enqueue_copy(hbuf, dev)
        ctx.synchronize()
        t = UInt64(perf_counter_ns())
        up += Float64(t.value() - t0.value()) / 1e6
        total += Float64(t.value() - t0.value()) / 1e6
    print(down / 20.0, up / 20.0, launch / 20.0, total / 20.0)
"""


def gpu_free_mib():
    """Free memory per visible GPU, or [] when there is no GPU tooling at all."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        return [int(line.strip()) for line in out if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def probe(timeout=300):
    """Time the copy-in / launch / copy-out phases of a 512 KiB round trip.

    Returns per-phase milliseconds, or None when no device accepts the work
    (no GPU, or the runtime rejects `DeviceContext()`); callers treat None as
    "skip".
    """
    if not gpu_free_mib():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rt_probe.mojo")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_PROBE)
        try:
            out = subprocess.run(
                ["mojo", "run", path], capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    if out.returncode != 0:
        return None
    parts = out.stdout.split()
    if len(parts) != 4:
        return None
    down, up, launch, total = (float(part) for part in parts)
    return {
        "copy_in_ms": down,
        "launch_call_ms": launch,
        "copy_out_ms": up,
        "round_trip_ms": down + launch + up,
    }
