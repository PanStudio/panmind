"""Shared fixtures for PanMind tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_panmind_dir():
    """Create temporary PanMind directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        panmind_path = Path(tmpdir) / ".panmind"
        panmind_path.mkdir()
        yield panmind_path


@pytest.fixture(autouse=True)
def _no_model_calls(monkeypatch):
    """Tests never spawn the model CLI; tests that exercise it inject a fake client."""
    monkeypatch.setenv("PANMIND_LLM", "0")
    # A lab shell's trust setting must not leak into provenance tests.
    monkeypatch.delenv("PANMIND_TRUST_HEADLESS", raising=False)
