"""Serve the packaged Agent Skills over MCP (``io.modelcontextprotocol/skills``).

``panmind/skills/<name>/SKILL.md`` is the one copy of each skill's text. The
server publishes it as ``skill://panmind/<name>/SKILL.md`` through
``skills/list``, ``skills/get`` and ``resources/read``; the Claude Code plugin
and the standalone skill carry derived copies that the test suite checks
against it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

import yaml
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding, ResourceBinding
from mcp.server.mcpserver.resources import TextResource
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, RequestParams
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
SKILL_TTL_MS = 300_000


class ListSkillsParams(RequestParams):
    cursor: str | None = None


class GetSkillParams(RequestParams):
    uri: str


def skill_texts() -> dict[str, bytes]:
    """``{skill name: SKILL.md bytes}`` for every packaged skill."""
    root = files("panmind") / "skills"
    return {
        entry.name: (entry / "SKILL.md").read_bytes()
        for entry in sorted(root.iterdir(), key=lambda e: e.name)
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    }


def skill_uri(name: str) -> str:
    return f"skill://panmind/{name}/SKILL.md"


def _frontmatter(name: str, raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"skill {name}: SKILL.md has no frontmatter")
    meta = yaml.safe_load(text.split("---\n", 2)[1])
    if not isinstance(meta, dict) or meta.get("name") != name:
        raise ValueError(f"skill {name}: frontmatter name must be {name!r}")
    if not meta.get("description"):
        raise ValueError(f"skill {name}: frontmatter has no description")
    return meta


class SkillsExtension(Extension):
    """Atomic skill entries whose manifests are computed from the bytes served."""

    identifier = SKILLS_EXTENSION

    def __init__(self) -> None:
        self.texts = skill_texts()
        self.entries: dict[str, dict[str, Any]] = {}
        for name, raw in self.texts.items():
            uri = skill_uri(name)
            self.entries[uri] = {
                "uri": uri,
                "frontmatter": _frontmatter(name, raw),
                "resources": [{
                    "uri": uri,
                    "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }],
            }

    def resources(self) -> Sequence[ResourceBinding]:
        return [
            ResourceBinding(TextResource(
                uri=skill_uri(name),
                name=f"{name}/SKILL.md",
                title=f"{name} skill",
                description=self.entries[skill_uri(name)]["frontmatter"]["description"],
                mime_type="text/markdown",
                text=raw.decode("utf-8"),
            ))
            for name, raw in self.texts.items()
        ]

    def methods(self) -> Sequence[MethodBinding]:
        # The extension is declared only in server/discover, so its methods
        # answer only there; handshake-era clients still read the files
        # through resources/read.
        return (
            MethodBinding("skills/list", ListSkillsParams, self._list,
                          protocol_versions=frozenset(MODERN_PROTOCOL_VERSIONS)),
            MethodBinding("skills/get", GetSkillParams, self._get,
                          protocol_versions=frozenset(MODERN_PROTOCOL_VERSIONS)),
        )

    async def _list(self, _ctx: ServerRequestContext[Any, Any],
                    params: ListSkillsParams) -> dict[str, Any]:
        if params.cursor is not None:
            raise MCPError(INVALID_PARAMS, "unknown skills/list cursor")
        return {"resultType": "complete", "skills": list(self.entries.values()),
                "ttlMs": SKILL_TTL_MS, "cacheScope": "public"}

    async def _get(self, _ctx: ServerRequestContext[Any, Any],
                   params: GetSkillParams) -> dict[str, Any]:
        entry = self.entries.get(params.uri)
        if entry is None:
            raise MCPError(INVALID_PARAMS, f"unknown skill {params.uri!r}",
                           data={"skills": sorted(self.entries)})
        return {"resultType": "complete", "skill": entry,
                "ttlMs": SKILL_TTL_MS, "cacheScope": "public"}
