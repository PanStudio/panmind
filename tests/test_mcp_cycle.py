"""End-to-end MCP protocol cycle through a real in-memory ``mcp`` client.

Every call here goes through the actual MCP tool-call wire shape (arguments
as a plain dict in, a CallToolResult out), not PanMindService directly, so a
gap between what server.py exposes and what service.py actually does would
show up here even if each side's own unit tests pass. server.py and
service.py are core's files; defects found here get reported to core as
BUGs rather than fixed in this repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.client import Client

from panmind.server import create_server
from panmind.taxonomy import list_dimensions

from .conftest import held_client

# A reply the heuristic classifier reads as thorough/verbose, paired with a
# correction endorsement.py maps straight to communication_register=terse
# (the same "too long, be terse" pattern T1's lab driver scripts).
VERBOSE_REPLY = (
    "Let me walk you through this in detail. First, I'll explain the background "
    "context, then the reasoning, then the implementation approach, then caveats, "
    "then alternatives, then a summary of everything discussed so far in depth."
)
TERSE_CORRECTION = "too long, be terse"

# Empirically converges in 6-12 turns against the current calibration
# (C8: prior strength 25, JS threshold 0.0125). Generous headroom so a
# future recalibration doesn't make this test flaky.
MAX_CORRECTION_TURNS = 25

TRAVERSAL_PERSONA_NAMES = ["../../etc/passwd", "..", ".", "a/b", "~root"]


ERAS = ["legacy", "2026-07-28"]


async def _client(panmind_dir: Path, mode: str = "auto") -> Client:
    server = await create_server(panmind_dir, import_root=panmind_dir.parent)
    return Client(server, mode=mode)


def _turn(assistant_text: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": user_text},
    ]


async def _drive_to_proposal(
    client: Client, persona: str, session_id: str
) -> tuple[str, list[dict[str, str]]]:
    """Send scripted correction turns until observe_interaction reports a
    proposal, or fail the test if it never does within MAX_CORRECTION_TURNS."""
    messages: list[dict[str, str]] = []
    for _ in range(MAX_CORRECTION_TURNS):
        messages += _turn(VERBOSE_REPLY, TERSE_CORRECTION)
        result = await client.call_tool(
            "observe_interaction",
            {"messages": messages, "persona_name": persona, "session_id": session_id},
        )
        assert not result.is_error, result.content
        sc = result.structured_content
        if sc["proposal_id"]:
            return sc["proposal_id"], messages
    pytest.fail(
        f"no proposal after {MAX_CORRECTION_TURNS} corrected turns for {persona!r}"
    )


class TestFullMCPCycle:
    """The happy-path cycle: list -> observe -> drift report -> accept ->
    context reflects the accepted target -> export -> import."""

    @pytest.mark.parametrize("era", ERAS)
    async def test_full_cycle_accept(self, tmp_path, era):
        panmind_dir = tmp_path / "panmind"
        async with await _client(panmind_dir, era) as client:
            personas = (await client.call_tool("list_personas", {})).structured_content
            assert "developer" in personas["personas"]

            proposal_id, _ = await _drive_to_proposal(client, "developer", "cycle-accept")

            report = (
                await client.call_tool("get_drift_report", {"persona_name": "developer"})
            ).structured_content
            assert report["proposal_id"] == proposal_id
            assert "communication_register" in report["per_dimension"]
            assert "communication_register" in report["endorsed"]

            accepted = (
                await client.call_tool(
                    "negotiate_update",
                    {
                        "proposal_id": proposal_id,
                        "decision": "accept",
                        "persona_name": "developer",
                    },
                )
            ).structured_content
            assert accepted["decision"] == "accepted"
            assert "communication_register" in accepted["dimensions"]
            assert accepted["commit"], "accept must git-commit the profile change"

            context = (
                await client.call_tool(
                    "get_behavioral_context", {"persona_name": "developer"}
                )
            ).structured_content
            # The renderer describes the "terse" state in prose ("be
            # concise: minimal words, maximum information density") rather
            # than printing the state name, so check the rendered text
            # loosely here and confirm the actual target precisely below,
            # via export_profile's structured dominant state.
            assert "concise" in context["behavioral_context"].lower()

            exported_check = (
                await client.call_tool(
                    "export_profile", {"persona_name": "developer", "format": "json"}
                )
            ).structured_content
            register = exported_check["dimensions"]["communication_register"]
            assert register["dominant_state"] == "terse", register

            # A second drift report on the same ledger must not re-propose
            # the dimension just accepted: the resolved profile now equals
            # the target, so there's nothing left to drift toward.
            settled = (
                await client.call_tool("get_drift_report", {"persona_name": "developer"})
            ).structured_content
            assert settled["proposal_id"] is None
            assert settled["negotiation_recommended"] is False

            exported = (
                await client.call_tool(
                    "export_profile",
                    {"persona_name": "developer", "format": "soulspec"},
                )
            ).structured_content
            assert exported["content"].startswith("# developer")

            import_dir = tmp_path / "import_src"
            import_dir.mkdir()
            soul_file = import_dir / "soul.md"
            soul_file.write_text(exported["content"])
            imported = (
                await client.call_tool(
                    "import_profile",
                    {
                        "source_path": "import_src/soul.md",
                        "persona_name": "developer-clone",
                    },
                )
            ).structured_content
            assert imported["persona"] == "developer-clone"
            assert imported["source"] == "soul.md"
            assert imported["saved_to"] == "personas/developer-clone.yaml"
            assert set(imported["distributions"]) == set(list_dimensions())
            assert (panmind_dir / "personas" / "developer-clone.yaml").is_file()


class TestRejectPathAndCooldown:
    async def test_reject_records_decision_and_suppresses_reproposal(self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        async with await _client(panmind_dir) as client:
            proposal_id, _ = await _drive_to_proposal(
                client, "researcher", "cycle-reject"
            )

            rejected = (
                await client.call_tool(
                    "negotiate_update",
                    {
                        "proposal_id": proposal_id,
                        "decision": "reject",
                        "persona_name": "researcher",
                        "reason": "test rejection",
                    },
                )
            ).structured_content
            assert rejected["decision"] == "rejected"
            assert rejected["reason"] == "test rejection"
            assert "communication_register" in rejected["dimensions"]

            # Re-evaluating the same ledger right after a reject must not
            # immediately produce a new proposal for the rejected
            # dimension. Whether that's a time-based cooldown or the
            # rejection's pseudo-counts pulling the posterior back toward
            # declared is an implementation detail; what must hold is that
            # rejecting doesn't just get silently overridden next call.
            after = (
                await client.call_tool(
                    "get_drift_report", {"persona_name": "researcher"}
                )
            ).structured_content
            assert after["negotiation_recommended"] is False
            assert after["proposal_id"] is None

    async def test_unknown_proposal_id_errors_cleanly(self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        async with await _client(panmind_dir) as client:
            result = await client.call_tool(
                "negotiate_update",
                {
                    "proposal_id": "0123456789ab",
                    "decision": "accept",
                    "persona_name": "developer",
                },
            )
            assert result.is_error
            assert "get_drift_report" in result.content[0].text

    async def test_unknown_decision_errors_cleanly(self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        async with await _client(panmind_dir) as client:
            result = await client.call_tool(
                "negotiate_update",
                {
                    "proposal_id": "irrelevant",
                    "decision": "maybe",
                    "persona_name": "developer",
                },
            )
            assert result.is_error


class TestPathTraversalPersonaNames:
    """Every tool that accepts a persona_name must reject a traversal
    attempt cleanly (isError) and must never write outside
    panmind_dir while doing it."""

    @pytest.fixture
    async def sandboxed_client(self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        server = await create_server(panmind_dir, import_root=tmp_path)
        async with held_client(server) as client:
            yield client, panmind_dir, tmp_path

    @staticmethod
    def _assert_contained(tmp_path: Path, panmind_dir: Path) -> None:
        """Every file under tmp_path must be inside panmind_dir (or the
        fixed import_src scratch dir this class never uses), proving a
        malicious persona_name never escaped PANMIND_DIR."""
        for p in tmp_path.rglob("*"):
            if p.is_file():
                assert p.is_relative_to(panmind_dir), f"escaped write: {p}"

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_get_behavioral_context(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        result = await client.call_tool(
            "get_behavioral_context", {"persona_name": bad_name}
        )
        assert result.is_error
        self._assert_contained(tmp_path, panmind_dir)

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_observe_interaction(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        result = await client.call_tool(
            "observe_interaction",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "persona_name": bad_name,
            },
        )
        assert result.is_error
        self._assert_contained(tmp_path, panmind_dir)

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_get_drift_report(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        result = await client.call_tool(
            "get_drift_report", {"persona_name": bad_name}
        )
        assert result.is_error
        self._assert_contained(tmp_path, panmind_dir)

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_negotiate_update(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        result = await client.call_tool(
            "negotiate_update",
            {"proposal_id": "x", "decision": "accept", "persona_name": bad_name},
        )
        assert result.is_error
        self._assert_contained(tmp_path, panmind_dir)

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_export_profile(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        result = await client.call_tool(
            "export_profile", {"persona_name": bad_name}
        )
        assert result.is_error
        self._assert_contained(tmp_path, panmind_dir)

    @pytest.mark.parametrize("bad_name", TRAVERSAL_PERSONA_NAMES)
    async def test_import_profile(self, sandboxed_client, bad_name):
        client, panmind_dir, tmp_path = sandboxed_client
        source_dir = tmp_path / "import_src"
        source_dir.mkdir(exist_ok=True)
        source_file = source_dir / f"src-{abs(hash(bad_name))}.md"
        source_file.write_text("# a persona\n\nDirect and confident.\n")
        result = await client.call_tool(
            "import_profile",
            {"source_path": f"import_src/{source_file.name}", "persona_name": bad_name},
        )
        assert result.is_error
        # import_profile legitimately writes outside panmind_dir (the
        # source file the test itself just planted), so contain the check
        # to panmind_dir's own subtree rather than reusing _assert_contained.
        for p in panmind_dir.rglob("*"):
            if p.is_file():
                assert "etc" not in p.parts and "passwd" not in p.name

    async def test_empty_persona_name_falls_back_to_default(self, sandboxed_client):
        """Not a traversal case, but the boundary right next to it: an
        empty string must resolve to the configured default, not error."""
        client, _panmind_dir, _tmp_path = sandboxed_client
        result = await client.call_tool("get_behavioral_context", {"persona_name": ""})
        sc = result.structured_content
        assert sc["persona"] == "default"


# ---------------------------------------------------------------------------
# import_profile containment
# ---------------------------------------------------------------------------

PERSONA_TEXT = "# a persona\n\nDirect and confident. Keep answers short.\n"


async def test_null_assistant_content_does_not_drop_the_conversation(tmp_path):
    """OpenAI-style tool-call messages carry "content": null."""
    server = await create_server(tmp_path / "panmind", import_root=tmp_path)
    async with Client(server) as client:
        result = await client.call_tool("observe_interaction", {"messages": [
            {"role": "user", "content": "fix the test"},
            {"role": "assistant", "content": None, "tool_calls": [{}]},
            {"role": "assistant", "content": "Fixed."},
            {"role": "user", "content": "thanks"},
        ]})
    assert not result.is_error, result.content
    assert result.structured_content["observations_added"] >= 1


class TestImportContainment:
    """source_path resolves inside the import root, symlinks included, and
    results name the file without disclosing where anything lives."""

    @pytest.fixture
    async def rooted(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text(PERSONA_TEXT)
        server = await create_server(tmp_path / "panmind", import_root=root)
        async with held_client(server) as client:
            yield client, root, outside, tmp_path / "panmind"

    @staticmethod
    async def _import(client: Client, source: str):
        return await client.call_tool(
            "import_profile", {"source_path": source, "persona_name": "imported"})

    async def test_file_inside_root_imports_without_absolute_paths(self, rooted):
        client, root, _outside, panmind_dir = rooted
        (root / "docs").mkdir()
        (root / "docs" / "CLAUDE.md").write_text(PERSONA_TEXT)
        result = await self._import(client, "docs/CLAUDE.md")
        assert not result.is_error, result.content
        sc = result.structured_content
        assert sc["source"] == "CLAUDE.md"
        assert sc["saved_to"] == "personas/imported.yaml"
        assert str(root) not in result.content[0].text
        assert (panmind_dir / "personas" / "imported.yaml").is_file()

    @pytest.mark.parametrize("source", ["../outside/secret.md", "{outside}/secret.md"])
    async def test_paths_outside_root_are_refused(self, rooted, source):
        client, _root, outside, panmind_dir = rooted
        result = await self._import(client, source.format(outside=outside))
        assert result.is_error
        assert "import root" in result.content[0].text
        assert not (panmind_dir / "personas" / "imported.yaml").exists()

    async def test_symlink_escaping_root_is_refused(self, rooted):
        client, root, outside, panmind_dir = rooted
        (root / "link.md").symlink_to(outside / "secret.md")
        (root / "linkdir").symlink_to(outside)
        for source in ("link.md", "linkdir/secret.md"):
            result = await self._import(client, source)
            assert result.is_error, source
        assert not (panmind_dir / "personas" / "imported.yaml").exists()

    async def test_oversize_file_is_refused(self, rooted):
        client, root, _outside, panmind_dir = rooted
        (root / "big.md").write_text("# big\n" + "x" * (256 * 1024))
        result = await self._import(client, "big.md")
        assert result.is_error
        assert "262144" in result.content[0].text
        assert not (panmind_dir / "personas" / "imported.yaml").exists()

    async def test_directory_is_refused(self, rooted):
        client, root, _outside, _panmind_dir = rooted
        (root / "sub").mkdir()
        assert (await self._import(client, "sub")).is_error

    async def test_env_var_sets_root(self, tmp_path, monkeypatch):
        root = tmp_path / "envroot"
        root.mkdir()
        (root / "soul.md").write_text(PERSONA_TEXT)
        monkeypatch.setenv("PANMIND_IMPORT_ROOT", str(root))
        server = await create_server(tmp_path / "panmind")
        async with Client(server) as client:
            assert not (await self._import(client, "soul.md")).is_error


# ---------------------------------------------------------------------------
# Elicitation for negotiate_update
# ---------------------------------------------------------------------------

MODERN = "2026-07-28"


def _pending_proposal(panmind_dir: Path) -> str:
    from panmind.service import PanMindService
    from panmind.store import TurnObservation

    service = PanMindService(panmind_dir)
    service.record([
        TurnObservation(persona="developer", session_id="s", turn_id=f"t{i}",
                        timestamp=float(i), source="heuristic",
                        labels={"communication_register": {"terse": 0.8, "moderate": 0.2}},
                        endorsement=1.0)
        for i in range(60)
    ], "developer")
    proposal_id = service.drift_report("developer")["proposal_id"]
    assert proposal_id
    return proposal_id


def _user_profile(panmind_dir: Path) -> Path:
    return panmind_dir / "personas" / "developer_user.yaml"


async def _no_elicitation(context, params):  # pragma: no cover - never called
    raise AssertionError("unexpected elicitation")


class TestNegotiateElicitation:
    @pytest.fixture
    async def setup(self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        server = await create_server(panmind_dir, import_root=tmp_path)
        return server, panmind_dir, _pending_proposal(panmind_dir)

    @staticmethod
    async def _first_round(client: Client, proposal_id: str):
        from mcp_types import InputRequiredResult

        first = await client.session.call_tool(
            "negotiate_update",
            {"proposal_id": proposal_id, "persona_name": "developer"},
            allow_input_required=True)
        assert isinstance(first, InputRequiredResult), first
        [(key, request)] = first.input_requests.items()
        return first, key, request

    async def test_modern_round_trip_asks_then_applies(self, setup):
        from mcp_types import ElicitResult

        server, panmind_dir, proposal_id = setup
        async with Client(server, mode=MODERN,
                          elicitation_callback=_no_elicitation) as client:
            first, key, request = await self._first_round(client, proposal_id)
            assert request.method == "elicitation/create"
            assert proposal_id in request.params.message
            schema = request.params.requested_schema
            assert schema["properties"]["decision"]["enum"] == ["accept", "reject"]
            dims = schema["properties"]["dimensions"]
            assert dims["type"] == "array"
            assert dims["items"]["enum"] == ["communication_register"]
            assert schema["required"] == ["decision"]
            assert not _user_profile(panmind_dir).exists()

            done = await client.session.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer"},
                input_responses={key: ElicitResult(
                    action="accept",
                    content={"decision": "accept",
                             "dimensions": ["communication_register"]})},
                request_state=first.request_state)
            assert not done.is_error, done.content
            assert done.structured_content["decision"] == "accepted"
            assert done.structured_content["commit"]
            assert _user_profile(panmind_dir).exists()

            # The same state handed back a second time is refused.
            from mcp.shared.exceptions import MCPError

            with pytest.raises(MCPError, match="Invalid or expired requestState"):
                await client.session.call_tool(
                    "negotiate_update",
                    {"proposal_id": proposal_id, "persona_name": "developer"},
                    input_responses={key: ElicitResult(
                        action="accept", content={"decision": "reject"})},
                    request_state=first.request_state)

    async def test_tampered_state_is_refused(self, setup):
        from mcp.shared.exceptions import MCPError
        from mcp_types import ElicitResult

        server, panmind_dir, proposal_id = setup
        async with Client(server, mode=MODERN,
                          elicitation_callback=_no_elicitation) as client:
            first, key, _request = await self._first_round(client, proposal_id)
            state = first.request_state
            tampered = state[:-2] + ("AA" if state[-2:] != "AA" else "BB")
            with pytest.raises(MCPError, match="Invalid or expired requestState"):
                await client.session.call_tool(
                    "negotiate_update",
                    {"proposal_id": proposal_id, "persona_name": "developer"},
                    input_responses={key: ElicitResult(
                        action="accept", content={"decision": "accept"})},
                    request_state=tampered)
            # Bound to the arguments: the same state for another proposal fails.
            with pytest.raises(MCPError, match="Invalid or expired requestState"):
                await client.session.call_tool(
                    "negotiate_update",
                    {"proposal_id": "abcdef012345", "persona_name": "developer"},
                    input_responses={key: ElicitResult(
                        action="accept", content={"decision": "accept"})},
                    request_state=state)
        assert not _user_profile(panmind_dir).exists()

    @pytest.mark.parametrize("action", ["decline", "cancel"])
    async def test_decline_and_cancel_change_nothing(self, setup, action):
        from mcp_types import ElicitResult

        server, panmind_dir, proposal_id = setup
        proposals = panmind_dir / "proposals" / "developer.json"
        before = proposals.read_text()
        async with Client(server, mode=MODERN,
                          elicitation_callback=_no_elicitation) as client:
            first, key, _request = await self._first_round(client, proposal_id)
            done = await client.session.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer"},
                input_responses={key: ElicitResult(action=action)},
                request_state=first.request_state)
        assert not done.is_error, done.content
        assert done.structured_content["decision"] == {
            "decline": "declined", "cancel": "cancelled"}[action]
        assert done.structured_content["commit"] is None
        assert proposals.read_text() == before
        assert not _user_profile(panmind_dir).exists()

    @pytest.mark.parametrize("era", ERAS)
    async def test_client_without_elicitation_gets_argument_path(self, setup, era):
        server, panmind_dir, proposal_id = setup
        async with Client(server, mode=era) as client:
            missing = await client.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer"})
            assert missing.is_error
            assert "'accept' or 'reject'" in missing.content[0].text
            done = await client.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer",
                 "decision": "reject", "reason": "not now"})
        assert done.structured_content["decision"] == "rejected"
        assert not _user_profile(panmind_dir).exists()

    async def test_legacy_client_answers_elicitation_create(self, setup):
        from mcp_types import ElicitResult

        server, panmind_dir, proposal_id = setup
        asked = []

        async def answer(context, params):
            asked.append(params)
            return ElicitResult(action="accept", content={
                "decision": "reject", "dimensions": ["communication_register"],
                "reason": "keep it"})

        async with Client(server, mode="legacy", elicitation_callback=answer) as client:
            done = await client.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer"})
        assert not done.is_error, done.content
        assert len(asked) == 1 and proposal_id in asked[0].message
        assert done.structured_content["decision"] == "rejected"
        assert done.structured_content["reason"] == "keep it"
        assert not _user_profile(panmind_dir).exists()

    async def test_stale_proposal_is_not_asked_about(self, setup):
        server, _panmind_dir, _proposal_id = setup
        async with Client(server, mode=MODERN,
                          elicitation_callback=_no_elicitation) as client:
            result = await client.call_tool(
                "negotiate_update",
                {"proposal_id": "0123456789ab", "persona_name": "developer"})
        assert result.is_error
        assert "get_drift_report" in result.content[0].text

    async def test_unknown_accepted_dimension_errors_before_eliciting(self, setup):
        server, _panmind_dir, proposal_id = setup
        async with Client(server, mode=MODERN,
                          elicitation_callback=_no_elicitation) as client:
            result = await client.call_tool(
                "negotiate_update",
                {"proposal_id": proposal_id, "persona_name": "developer",
                 "accepted_dimensions": ["communication_register", "narration"]})
        assert result.is_error
        assert "narration" in result.content[0].text


# ---------------------------------------------------------------------------
# Skills over MCP
# ---------------------------------------------------------------------------


class TestSkills:
    @staticmethod
    async def _rpc(client: Client, method: str, params: dict | None = None):
        from typing import Any, Literal

        from mcp import types
        from mcp_types import RequestParams
        from pydantic import ConfigDict, TypeAdapter

        class Params(RequestParams):
            model_config = ConfigDict(extra="allow")

        class Raw(types.Request[Params | None, Any]):
            method: Literal["skills/list", "skills/get"]
            params: Params | None = None

        return await client.session.send_request(
            Raw(method=method, params=Params(**(params or {}))),
            TypeAdapter(dict[str, Any]))

    async def test_manifests_match_served_bytes(self, tmp_path):
        import hashlib

        server = await create_server(tmp_path / "panmind", import_root=tmp_path)
        # mode="auto" discovers; it settles on 2026-07-28 here.
        async with Client(server) as client:
            assert client.protocol_version == MODERN
            caps = client.server_capabilities
            assert "io.modelcontextprotocol/skills" in (caps.extensions or {})
            assert caps.resources is not None
            listed = await self._rpc(client, "skills/list")
            assert listed["resultType"] == "complete"
            assert listed["ttlMs"] > 0 and listed["cacheScope"] == "public"
            names = {s["frontmatter"]["name"] for s in listed["skills"]}
            assert names == {"panmind", "panmind-observer"}
            for skill in listed["skills"]:
                [entry] = skill["resources"]
                read = await client.read_resource(entry["uri"])
                body = read.contents[0].text.encode("utf-8")
                assert entry["digest"] == "sha256:" + hashlib.sha256(body).hexdigest()
                assert entry["size"] == len(body)
                assert read.contents[0].mime_type == "text/markdown"
                got = await self._rpc(client, "skills/get", {"uri": skill["uri"]})
                assert got["skill"] == skill

    async def test_unknown_skill_is_invalid_params(self, tmp_path):
        from mcp.shared.exceptions import MCPError

        server = await create_server(tmp_path / "panmind", import_root=tmp_path)
        async with Client(server, mode=MODERN) as client:
            with pytest.raises(MCPError) as err:
                await self._rpc(client, "skills/get",
                                {"uri": "skill://panmind/nope/SKILL.md"})
        assert err.value.error.code == -32602

    async def test_legacy_reads_skill_files_as_resources(self, tmp_path):
        server = await create_server(tmp_path / "panmind", import_root=tmp_path)
        async with Client(server, mode="legacy") as client:
            resources = await client.list_resources()
            uris = {str(r.uri) for r in resources.resources}
            assert "skill://panmind/panmind/SKILL.md" in uris
            read = await client.read_resource("skill://panmind/panmind/SKILL.md")
            assert read.contents[0].text.startswith("---\nname: panmind\n")


# ---------------------------------------------------------------------------
# Schemas, bounds and listing
# ---------------------------------------------------------------------------


class TestBoundsAndListing:
    @pytest.mark.parametrize("era", ["legacy", "auto"])
    async def test_listing_publishes_enums_and_capabilities(self, tmp_path, era):
        async with await _client(tmp_path / "panmind", era) as client:
            assert client.protocol_version == (
                MODERN if era == "auto" else "2025-11-25")
            tools = {t.name: t for t in (await client.list_tools()).tools}
            caps = client.server_capabilities
        assert caps.tools is not None and not caps.tools.list_changed
        negotiate = tools["negotiate_update"].input_schema["properties"]
        assert negotiate["decision"]["anyOf"][0]["enum"] == ["accept", "reject"]
        dims = negotiate["accepted_dimensions"]["anyOf"][0]["items"]["enum"]
        assert dims == list_dimensions()
        export = tools["export_profile"].input_schema["properties"]
        assert export["format"]["enum"] == ["yaml", "json", "soulspec"]
        persona = tools["get_drift_report"].input_schema["properties"]["persona_name"]
        assert persona["pattern"] == "^$|^[a-zA-Z0-9_-]{1,50}$"
        messages = tools["observe_interaction"].input_schema["properties"]["messages"]
        assert messages["maxItems"] == 1000
        assert all(t.title and t.icons for t in tools.values())

    async def test_list_personas_pages(self, tmp_path):
        async with await _client(tmp_path / "panmind") as client:
            seen, cursor = [], None
            while True:
                args = {"limit": 2} | ({"cursor": cursor} if cursor else {})
                page = (await client.call_tool("list_personas", args)).structured_content
                seen += page["personas"]
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            assert seen == ["developer", "researcher", "writer"]
            bad = await client.call_tool("list_personas", {"cursor": "bogus"})
        assert bad.is_error and "next_cursor" in bad.content[0].text

    async def test_invalid_enum_and_oversize_messages_are_errors(self, tmp_path):
        async with await _client(tmp_path / "panmind") as client:
            fmt = await client.call_tool("export_profile", {"format": "toml"})
            many = await client.call_tool(
                "observe_interaction",
                {"messages": [{"role": "user", "content": "hi"}] * 1001})
            big = await client.call_tool(
                "observe_interaction",
                {"messages": [{"role": "user", "content": "x" * 100_001}]})
        assert fmt.is_error and many.is_error and big.is_error

    async def test_observe_interaction_leaves_profiles_alone_below_threshold(
            self, tmp_path):
        panmind_dir = tmp_path / "panmind"
        async with await _client(panmind_dir) as client:
            before = {p.name: p.read_bytes()
                      for p in (panmind_dir / "personas").glob("*.yaml")}
            result = await client.call_tool("observe_interaction", {
                "messages": _turn(VERBOSE_REPLY, "thanks, that works"),
                "persona_name": "developer", "session_id": "quiet"})
        assert not result.is_error
        assert result.structured_content["auto_accepted"] == []
        after = {p.name: p.read_bytes()
                 for p in (panmind_dir / "personas").glob("*.yaml")}
        assert after == before

    def test_long_text_is_capped_with_marker(self):
        from panmind.server import MAX_TEXT_CHARS, _cap

        text, truncated = _cap("y" * (MAX_TEXT_CHARS + 10))
        assert truncated and len(text) == MAX_TEXT_CHARS
        assert text.endswith(f"[truncated by PanMind at {MAX_TEXT_CHARS} characters]")
        assert _cap("short") == ("short", False)


class TestSingleUseRequestState:
    @staticmethod
    def _ctx(token: str):
        from types import SimpleNamespace

        return SimpleNamespace(method="tools/call", params={"requestState": token})

    async def test_only_accepted_tokens_are_remembered(self):
        from mcp.shared.exceptions import MCPError

        from panmind.server import SingleUseRequestState

        guard = SingleUseRequestState(ttl=60)
        calls = []

        async def boundary(ctx):
            calls.append(ctx.params["requestState"])
            if ctx.params["requestState"] == "forged":
                raise MCPError(-32602, "Invalid or expired requestState",
                               data={"reason": "invalid_request_state"})
            return "ok"

        for _ in range(2):
            with pytest.raises(MCPError):
                await guard(self._ctx("forged"), boundary)
        assert calls == ["forged", "forged"]
        assert await guard(self._ctx("good"), boundary) == "ok"
        with pytest.raises(MCPError, match="Invalid or expired"):
            await guard(self._ctx("good"), boundary)
        assert calls == ["forged", "forged", "good"]

    async def test_a_full_cache_refuses_new_rounds_before_running_them(self):
        from mcp.shared.exceptions import MCPError

        from panmind.server import SingleUseRequestState

        guard = SingleUseRequestState(ttl=60, max_entries=1)
        calls = []

        async def boundary(ctx):
            calls.append(ctx.params["requestState"])
            return "ok"

        await guard(self._ctx("a"), boundary)
        with pytest.raises(MCPError):
            await guard(self._ctx("b"), boundary)
        assert calls == ["a"]

    async def test_non_mrtr_methods_cannot_populate_replay_cache(self):
        from types import SimpleNamespace

        from panmind.server import SingleUseRequestState

        guard = SingleUseRequestState(ttl=60, max_entries=1)

        async def passthrough(ctx):
            return ctx.method

        for i, method in enumerate(("ping", "tools/list", "skills/list")):
            ctx = SimpleNamespace(method=method, params={"requestState": f"junk-{i}"})
            assert await guard(ctx, passthrough) == method

        assert await guard(self._ctx("real"), passthrough) == "tools/call"
