"""Shared fixtures for PanMind tests."""

import asyncio
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.client import Client


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


@asynccontextmanager
async def held_client(server, **kwargs) -> AsyncIterator[Client]:
    """An open ``mcp`` client whose task group lives in a task of its own.

    pytest-asyncio sets a fixture up and tears it down in different tasks,
    which an anyio cancel scope refuses, so the client is entered and
    exited inside one dedicated task instead.
    """
    opened: asyncio.Future[Client] = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def hold() -> None:
        try:
            async with Client(server, **kwargs) as client:
                opened.set_result(client)
                await release.wait()
        except BaseException as exc:
            if not opened.done():
                opened.set_exception(exc)
            raise

    task = asyncio.create_task(hold())
    try:
        yield await opened
    finally:
        release.set()
        await task
