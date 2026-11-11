import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a working GPU device; skips on CPU-only hosts")
