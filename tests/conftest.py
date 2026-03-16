# conftest.py
"""
Shared pytest configuration.

Markers:
  unit        — fast, no I/O, no GPU, no API keys
  integration — requires external services
  cuda        — requires NVIDIA GPU
  mps         — requires Apple Silicon
  cpu         — runs on any platform (CPU fallback)
"""
import pytest

from hooks.types import RunConfig, RunResults


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: requires live external services")
    config.addinivalue_line("markers", "cuda: requires NVIDIA GPU")
    config.addinivalue_line("markers", "mps: requires Apple Silicon MPS")
    config.addinivalue_line("markers", "cpu: runs on any platform")
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring mnemebrain-lite")


def pytest_collection_modifyitems(config, items):
    """Skip cuda/mps tests when hardware not available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps  = torch.backends.mps.is_available()
    except ImportError:
        has_cuda = has_mps = False

    skip_cuda = pytest.mark.skip(reason="NVIDIA GPU not available")
    skip_mps  = pytest.mark.skip(reason="Apple Silicon MPS not available")

    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not has_mps:
            item.add_marker(skip_mps)


@pytest.fixture
def sample_config():
    return RunConfig(
        matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
        scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
        device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
    )


@pytest.fixture
def sample_results():
    return RunResults(
        val_bpb=1.872, steps=1450, peak_vram_mb=4200.0, final_loss=2.34,
        mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.2,
    )


@pytest.fixture
def diverged_results():
    return RunResults(
        val_bpb=float("nan"), steps=342, peak_vram_mb=4200.0, final_loss=100.0,
        mfu=0.10, diverged=True, loss_trend="diverging", grad_norm_max=9.1,
    )


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp directory."""
    monkeypatch.setattr("hooks.artifacts.RESULTS_DIR", tmp_path)
    return tmp_path
