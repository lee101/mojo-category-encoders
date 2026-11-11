from __future__ import annotations

import pytest

from bench.gpu_probe import gpu_free_mib, probe

GATE_FREE_MIB = 4000


@pytest.fixture(scope="session")
def device_probe():
    """Measured device round trip, or skip when this host has no usable GPU.

    This fixture is the one place that decides whether a GPU test may run. On a
    CPU-only host (no device, no `nvidia-smi`, a device under the memory gate, or
    a runtime that cannot build a `DeviceContext`) it skips, so the suite stays
    green without a GPU.
    """
    free = gpu_free_mib()
    if not free:
        pytest.skip("no GPU visible: GPU path skipped")
    if max(free) < GATE_FREE_MIB:
        pytest.skip(f"GPU has {max(free)} MiB free, under the {GATE_FREE_MIB} MiB gate: skipped")
    result = probe()
    if result is None:
        pytest.skip("device probe could not run: runtime has no usable GPU, skipped")
    return result


@pytest.mark.gpu
def test_device_round_trip_completes(device_probe, capsys):
    # Guards the probe the README's "GPU path not shipped" verdict is measured
    # against: copy 512 KiB in, launch a kernel, copy it back.
    assert device_probe["round_trip_ms"] > 0
    assert device_probe["copy_in_ms"] >= 0
    assert device_probe["copy_out_ms"] >= 0
    with capsys.disabled():
        print(f"\ndevice round trip: {device_probe}")
    assert device_probe["round_trip_ms"] > 0


def test_probe_reports_no_device_without_nvidia_smi(monkeypatch):
    # The CPU-only skip path itself is tested on every host.
    monkeypatch.setenv("PATH", "/nonexistent")
    assert gpu_free_mib() == []
    assert probe() is None
