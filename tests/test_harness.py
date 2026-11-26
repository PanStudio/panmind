"""BehavioralHarness over PanMindService."""

from pathlib import Path

import pytest

from panmind.harness import BehavioralHarness


@pytest.fixture()
def panmind_dir(tmp_path: Path) -> Path:
    return tmp_path / "panmind"


def test_context_is_the_rendered_persona_profile(panmind_dir):
    ctx = BehavioralHarness("developer", panmind_dir).get_context()
    assert "'developer'" in ctx
    assert "Epistemic style" in ctx


def test_decorator_injects_system_and_preserves_metadata(panmind_dir):
    captured: list[str] = []
    harness = BehavioralHarness("developer", panmind_dir, observe_every=1000)

    @harness
    def agent(messages, system=""):
        """Doc."""
        captured.append(system)
        return 42

    assert agent([]) == 42
    assert agent.__name__ == "agent" and agent.__doc__ == "Doc."
    assert "Epistemic style" in captured[0]


def test_invalid_persona_is_rejected(panmind_dir):
    from panmind.security import SecurityError

    with pytest.raises(SecurityError):
        BehavioralHarness("../x", panmind_dir)


def test_fresh_harness_does_not_recommend_negotiation(panmind_dir):
    harness = BehavioralHarness("developer", panmind_dir)
    assert harness.should_negotiate() is False
    assert harness.get_drift_summary() is None


def test_repeated_observe_of_a_growing_conversation_is_idempotent(panmind_dir):
    pytest.importorskip("panmind.ingest")
    harness = BehavioralHarness("developer", panmind_dir)
    convo = [{"role": "user", "content": "Refactor this."},
             {"role": "assistant", "content": "Done. I used a comprehension."},
             {"role": "user", "content": "Thanks. Now add a test."}]
    first = harness.observe(convo)
    assert first["observations_added"] >= 1
    assert harness.observe(convo)["observations_added"] == 0
