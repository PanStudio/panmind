"""PanMind MCP server: seven tools over PanMindService, plus the packaged skills.

Built on the ``mcp`` 2.x ``MCPServer``, which serves 2026-07-28 statelessly
beside the ``initialize`` handshake (2024-11-05 through 2025-11-25). Tools are
plain functions, so the SDK runs them in a worker thread and their file
locks, git commits and model-CLI calls never block the event loop. Failures
raise ``ToolError`` and reach the client as ``isError: true`` results.
"""

import base64
import binascii
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NoReturn,
    NotRequired,
    TypedDict,
    TypeVar,
)

from mcp.server.caching import CacheHint
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import (
    CancelledElicitation,
    Context,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    RequestStateBoundary,
    RequestStateSecurity,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import Icon, ToolAnnotations
from mcp_types import INVALID_PARAMS
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, create_model

from . import __version__
from .security import (
    SAFE_PERSONA_NAME_PATTERN,
    SecurityError,
    read_import_source,
    sanitize_error_message,
)
from .service import DimensionReport, PanMindService
from .skills_extension import SKILL_TTL_MS, SkillsExtension
from .store import Proposal
from .taxonomy import list_dimensions

logger = logging.getLogger(__name__)
_F = TypeVar("_F", bound=Callable[..., Any])

if TYPE_CHECKING:
    Dimension = str
else:
    Dimension = Literal[tuple(list_dimensions())]

MAX_MESSAGES = 1000
MAX_CONTENT_CHARS = 100_000
MAX_TEXT_CHARS = 64 * 1024
REQUEST_STATE_TTL = 300.0
TRUNCATION_MARKER = "\n\n[truncated by PanMind at {limit} characters]"

INSTRUCTIONS = (
    "PanMind keeps a behavioral profile per persona as probability distributions "
    "over nine dimensions in PANMIND_DIR, a git repository. Read the profile with "
    "get_behavioral_context. observe_interaction records answered assistant turns "
    "and may auto-accept a small, credible change, which commits the user profile. "
    "When it reports negotiation_recommended, call get_drift_report and show the "
    "proposal to the user, then call negotiate_update with their decision; omit "
    "decision and a client with form elicitation asks the user directly. Only "
    "accepted changes and auto-accepts alter a profile, and every change is "
    "git-committed."
)

_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#1f2937"/>'
    '<path d="M14 44 L24 28 L32 36 L40 20 L50 44" fill="none" stroke="#34d399" '
    'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
ICONS = [Icon(
    src="data:image/svg+xml;base64," + base64.b64encode(_ICON_SVG.encode()).decode(),
    mime_type="image/svg+xml", sizes=["any"],
)]

PersonaName = Annotated[str, Field(
    pattern=f"^$|{SAFE_PERSONA_NAME_PATTERN.pattern}",
    max_length=50,
    description="Persona name, e.g. 'developer'. Omit to use the default persona "
    "(PANMIND_DIR/default_persona, else 'default'). Unknown names resolve to the "
    "base identity.",
)]


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Annotated[str, Field(max_length=32)] = "text"
    text: Annotated[str | None, Field(max_length=MAX_CONTENT_CHARS)] = None


class Message(BaseModel):
    """One conversation message; only user and assistant text is used."""

    model_config = ConfigDict(extra="ignore")
    role: Annotated[str, Field(max_length=32,
                               description="'user' or 'assistant'; others are ignored")]
    content: (Annotated[str, Field(max_length=MAX_CONTENT_CHARS)]
              | Annotated[list[ContentBlock], Field(max_length=200)]
              | None) = Field(
        default="",
        description="Text, or a list of content blocks whose text blocks are read")


class NegotiationChoice(BaseModel):
    """What the human decided about a proposal."""

    decision: Literal["accept", "reject"] = Field(
        title="Decision",
        description="accept writes the proposed targets to your profile and commits "
        "them; reject keeps your profile and pauses these proposals for 3 days")
    dimensions: list[str] | None = None
    reason: str = Field(default="", max_length=500, title="Reason (optional)")


# -- results ---------------------------------------------------------------


class ListPersonasResult(TypedDict):
    personas: list[str]
    next_cursor: str | None


class GetBehavioralContextResult(TypedDict):
    persona: str
    behavioral_context: str
    truncated: bool
    observation_count: int
    specialization_level: int


class ObserveInteractionResult(TypedDict):
    persona: str
    observations_added: int
    turns: int
    negotiation_recommended: bool
    proposal_id: str | None
    proposal_tier: str | None
    auto_accepted: list[str]


class GetDriftReportResult(TypedDict):
    persona: str
    turns: int
    negotiation_recommended: bool
    proposal_id: str | None
    proposal_tier: str | None
    summary: str
    per_dimension: dict[str, str]
    endorsed: dict[str, DimensionReport]
    fingerprint: dict[str, DimensionReport]
    auto_accepted: list[str]
    cooling_down: list[str]
    explicit_only: list[str]


class NegotiateUpdateResult(TypedDict):
    persona: str
    proposal_id: str
    decision: Literal["accepted", "rejected", "declined", "cancelled"]
    dimensions: list[str]
    commit: str | None
    reason: str | None


class ImportProfileResult(TypedDict):
    persona: str
    source: str
    saved_to: str
    commit: str | None
    distributions: dict[str, dict[str, str | float]]


class ExportProfileResult(TypedDict):
    persona: str
    format: Literal["yaml", "json", "soulspec"]
    content: NotRequired[str]
    truncated: NotRequired[bool]
    agent_id: NotRequired[str]
    level: NotRequired[str]
    dimensions: NotRequired[dict[str, dict[str, Any]]]


# -- helpers ---------------------------------------------------------------


def _fail(tool: str, exc: Exception, hint: str = "") -> NoReturn:
    logger.warning("%s failed: %s", tool, exc)
    message = sanitize_error_message(exc)
    if isinstance(exc, TimeoutError):
        hint = hint or "Another PanMind process holds the directory; retry shortly."
    raise ToolError(f"{message}. {hint}".strip() if hint else message) from exc


@contextmanager
def _errors(tool: str, hint: str = "") -> Iterator[None]:
    try:
        yield
    except ToolError:
        raise
    except Exception as exc:
        _fail(tool, exc, hint)


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    marker = TRUNCATION_MARKER.format(limit=MAX_TEXT_CHARS)
    return text[:MAX_TEXT_CHARS - len(marker)] + marker, True


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"personas:{offset}".encode()).decode()


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        kind, _, offset = raw.partition(":")
        if kind == "personas" and offset.isdigit():
            return int(offset)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        pass
    raise ToolError("invalid cursor. Pass next_cursor from the previous "
                    "list_personas page, or omit it for the first page.")


def _dominant(distribution: dict[str, float]) -> str:
    state = max(distribution, key=lambda s: distribution[s])
    return f"{state} {distribution[state]:.2f}"


def _supports_form_elicitation(ctx: Context[Any, Any]) -> bool:
    caps = ctx.client_capabilities
    elicitation = caps.elicitation if caps is not None else None
    # A bare `elicitation: {}` predates modes and means form support.
    return elicitation is not None and (elicitation.form is not None
                                        or elicitation.url is None)


def _confirmation_form(persona: str, proposal: Proposal,
                       offered: list[str]) -> Elicit[NegotiationChoice]:
    lines = [
        f"PanMind proposes updating the '{persona}' profile "
        f"(proposal {proposal.id}, {proposal.tier}, "
        f"{proposal.observation_count} observed turns):",
    ]
    for dim in offered:
        change = proposal.changes[dim]
        lines.append(f"- {dim}: {_dominant(change.declared)} -> "
                     f"{_dominant(change.target)} ({change.tier})")
    lines.append("Accept the dimensions you want applied, or reject them.")
    # An explicit enum, because pydantic renders a one-member Literal as
    # `const`, which elicitation's multi-select schema does not admit.
    picks = Annotated[list[Literal[tuple(offered)]],  # type: ignore[valid-type,misc]
                      WithJsonSchema({"type": "array", "minItems": 1,
                                      "items": {"type": "string", "enum": offered}})]
    schema = create_model(
        "NegotiationChoice",
        __base__=NegotiationChoice,
        dimensions=(picks, Field(
            default=offered, min_length=1, title="Dimensions",
            description="Which proposed changes this decision applies to")),
    )
    return Elicit("\n".join(lines), schema)


def _invalid_request_state() -> MCPError:
    return MCPError(code=INVALID_PARAMS, message="Invalid or expired requestState",
                    data={"reason": "invalid_request_state"})


_MRTR_METHODS = frozenset({"tools/call", "prompts/get", "resources/read"})


@dataclass
class SingleUseRequestState:
    """Refuse a sealed ``requestState`` the server has already accepted.

    Runs outside ``RequestStateBoundary``, so it sees the sealed wire token,
    which is unique per round (fresh AEAD nonce). Only inspects the MRTR
    methods that ``RequestStateBoundary`` validates, so non-MRTR requests
    carrying an arbitrary ``requestState`` cannot populate the replay cache.
    A token is held while its request runs, so a concurrent replay is refused
    too, and is remembered only once the boundary has accepted it, so forged
    tokens cost nothing. Entries outlive the token's TTL, after which the
    boundary refuses it anyway. When ``max_entries`` live tokens are
    remembered, new ones are refused rather than forgetting a live one.
    """

    ttl: float
    max_entries: int = 10_000

    def __post_init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._held: set[str] = set()
        self._lock = threading.Lock()

    async def __call__(self, ctx: ServerRequestContext[Any, Any],
                       call_next: CallNext) -> HandlerResult:
        if ctx.method not in _MRTR_METHODS:
            return await call_next(ctx)
        token = ctx.params.get("requestState") if ctx.params else None
        if not isinstance(token, str):
            return await call_next(ctx)
        key = hashlib.sha256(token.encode()).hexdigest()
        with self._lock:
            if key in self._seen or key in self._held:
                logger.warning("requestState replay refused on %s", ctx.method)
                raise _invalid_request_state()
            self._prune()
            if len(self._seen) + len(self._held) >= self.max_entries:
                logger.warning("requestState replay cache full; refusing new rounds")
                raise _invalid_request_state()
            self._held.add(key)
        accepted = False
        try:
            result = await call_next(ctx)
            accepted = True
            return result
        except MCPError as exc:
            data = exc.error.data
            accepted = not (isinstance(data, dict)
                            and data.get("reason") == "invalid_request_state")
            raise
        except BaseException:
            # The boundary let the token through before the handler failed.
            accepted = True
            raise
        finally:
            with self._lock:
                self._held.discard(key)
                if accepted:
                    self._seen[key] = time.monotonic()

    def _prune(self) -> None:
        now = time.monotonic()
        # Insertion order is time order, so expired entries lead.
        while self._seen:
            old, seen_at = next(iter(self._seen.items()))
            if now - seen_at < self.ttl:
                break
            del self._seen[old]


def _import_root() -> Path:
    return Path(os.environ.get("PANMIND_IMPORT_ROOT") or Path.cwd()).resolve()


# -- server ----------------------------------------------------------------


async def create_server(panmind_dir: Path | None = None,
                        import_root: Path | None = None) -> MCPServer:
    """Create the PanMind MCP server over ``panmind_dir`` (default ``~/.panmind``).

    ``import_profile`` reads only regular files under ``import_root``
    (default: ``PANMIND_IMPORT_ROOT``, else the working directory at launch).
    """
    if panmind_dir is None:
        panmind_dir = Path.home() / ".panmind"
    panmind_dir.mkdir(parents=True, exist_ok=True)
    root = (import_root or _import_root()).resolve()
    service = PanMindService(panmind_dir)

    mcp = MCPServer(
        "panmind",
        title="PanMind",
        version=__version__,
        description="Learns an agent's behavioral preferences from observed "
        "turns and negotiates profile updates with the user.",
        website_url="https://github.com/PanStudio/panmind",
        instructions=INSTRUCTIONS,
        icons=ICONS,
        extensions=[SkillsExtension()],
        # A fixed tool and resource set: no subscriptions/listen, so the
        # modern era advertises no listChanged or subscribe.
        subscriptions=False,
        request_state_security=RequestStateSecurity.ephemeral(ttl=REQUEST_STATE_TTL),
        cache_hints={
            "tools/list": CacheHint(ttl_ms=3_600_000, scope="public"),
            "resources/list": CacheHint(ttl_ms=SKILL_TTL_MS, scope="public"),
            "resources/templates/list": CacheHint(ttl_ms=SKILL_TTL_MS, scope="public"),
            "resources/read": CacheHint(ttl_ms=SKILL_TTL_MS, scope="public"),
        },
    )
    boundary = next(i for i, m in enumerate(mcp.middleware)
                    if isinstance(m, RequestStateBoundary))
    mcp.middleware.insert(boundary, SingleUseRequestState(ttl=REQUEST_STATE_TTL * 2))

    def tool(name: str, title: str, description: str, *, read_only: bool,
             destructive: bool, idempotent: bool,
             open_world: bool = False) -> Callable[[_F], _F]:
        return mcp.tool(
            name=name, title=title, description=description, icons=ICONS,
            annotations=ToolAnnotations(
                title=title, read_only_hint=read_only, destructive_hint=destructive,
                idempotent_hint=idempotent, open_world_hint=open_world),
        )

    @tool("list_personas", "List Personas",
          "List persona names in PANMIND_DIR/personas, sorted, a page at a time. "
          "Pass next_cursor back as cursor for the next page.",
          read_only=True, destructive=False, idempotent=True)
    def list_personas(
        cursor: Annotated[str | None, Field(
            max_length=64, description="next_cursor from the previous page")] = None,
        limit: Annotated[int, Field(ge=1, le=500, description="Page size")] = 100,
    ) -> ListPersonasResult:
        start = _decode_cursor(cursor)
        with _errors("list_personas"):
            names = service.list_personas()
        end = start + limit
        return {"personas": names[start:end],
                "next_cursor": _encode_cursor(end) if end < len(names) else None}

    @tool("get_behavioral_context", "Get Behavioral Context",
          "Get a persona's resolved behavioral profile as system-prompt text, "
          f"at most {MAX_TEXT_CHARS} characters. Reads only.",
          read_only=True, destructive=False, idempotent=True)
    def get_behavioral_context(
        persona_name: PersonaName = "",
        model: Annotated[str, Field(
            max_length=128,
            description="Model id of the calling agent, e.g. claude-opus-5. When "
            "given, the context counters that model's own observed tendencies; "
            "otherwise it covers the most observed models")] = "",
    ) -> GetBehavioralContextResult:
        with _errors("get_behavioral_context"):
            ctx = service.context(persona_name or None, model or None)
        text, truncated = _cap(ctx["behavioral_context"])
        return {"persona": ctx["persona"], "behavioral_context": text,
                "truncated": truncated,
                "observation_count": ctx["observation_count"],
                "specialization_level": ctx["specialization_level"]}

    @tool("observe_interaction", "Observe Interaction",
          "Record the answered assistant turns of a conversation in the persona's "
          "observation ledger (PANMIND_DIR/observations) and re-evaluate drift. "
          "May record a pending proposal, and when the evidence shows a small, "
          "credible change it auto-accepts it, which updates the persona's user "
          "profile (personas/<persona>_user.yaml) and git-commits it. Always "
          "rewrites rendered/<persona>.md. An assistant turn is recorded once a "
          "user message follows it; re-send the growing conversation with the "
          "same session_id and recorded turns are skipped.",
          read_only=False, destructive=True, idempotent=False)
    def observe_interaction(
        messages: Annotated[list[Message], Field(
            min_length=1, max_length=MAX_MESSAGES,
            description="Conversation messages in order, as role and content")],
        persona_name: PersonaName = "",
        session_id: Annotated[str, Field(
            max_length=128,
            description="Stable id for this conversation, so repeated calls "
            "do not count the same turns twice")] = "",
    ) -> ObserveInteractionResult:
        raw = [m.model_dump(exclude_none=True) for m in messages]
        with _errors("observe_interaction"):
            return service.observe_messages(persona_name or None, raw,
                                            session_id or None)

    @tool("get_drift_report", "Get Drift Report",
          "Get a persona's drift report: endorsed drift, per-model fingerprint "
          "diagnostics and the pending proposal's id. Records that proposal in "
          "PANMIND_DIR/proposals so the id stays valid for negotiate_update; "
          "never changes a profile, and repeated calls on the same ledger "
          "return the same proposal. Call when negotiation_recommended is true.",
          read_only=False, destructive=False, idempotent=True)
    def get_drift_report(persona_name: PersonaName = "") -> GetDriftReportResult:
        with _errors("get_drift_report"):
            return service.drift_report(persona_name or None)

    def resolve_choice(
        ctx: Context[Any, Any],
        proposal_id: str,
        decision: Literal["accept", "reject"] | None = None,
        persona_name: str = "",
        reason: str = "",
        accepted_dimensions: list[str] | None = None,
    ) -> Elicit[NegotiationChoice] | NegotiationChoice:
        if decision is not None:
            return NegotiationChoice(decision=decision, reason=reason,
                                     dimensions=accepted_dimensions)
        if not _supports_form_elicitation(ctx):
            raise ToolError("decision is required: pass 'accept' or 'reject' "
                            "after showing the user the proposal from "
                            "get_drift_report.")
        with _errors("negotiate_update"):
            persona = service.persona(persona_name or None)
            proposal = service.negotiator.proposals.get(persona, proposal_id)
        if proposal is None or proposal.status != "pending":
            state = "unknown" if proposal is None else proposal.status
            raise ToolError(f"proposal {proposal_id} for persona '{persona}' is "
                            f"{state}. Call get_drift_report for the current "
                            "proposal id.")
        if accepted_dimensions:
            unknown = sorted(set(accepted_dimensions) - set(proposal.changes))
            if unknown:
                raise ToolError(
                    f"accepted_dimensions not in proposal: {unknown}. "
                    f"Proposal {proposal_id} covers: {sorted(proposal.changes)}."
                )
        offered = [d for d in proposal.changes
                   if not accepted_dimensions or d in accepted_dimensions]
        return _confirmation_form(persona, proposal, offered)

    @tool("negotiate_update", "Negotiate Behavioral Update",
          "Accept or reject a pending proposal from get_drift_report. Accept writes "
          "the proposed distributions to personas/<persona>_user.yaml, git-commits "
          "them and rewrites rendered/<persona>.md; reject records the decision in "
          "PANMIND_DIR/proposals (git-committed) and pauses proposals on those "
          "dimensions for 3 days. Pass the user's decision, or omit it and a "
          "client that supports form elicitation asks the user; declining or "
          "cancelling that form changes nothing.",
          read_only=False, destructive=True, idempotent=False)
    # decision, reason and accepted_dimensions are read by resolve_choice,
    # which folds them (or the user's form answer) into choice.
    def negotiate_update(
        choice: Annotated[ElicitationResult[NegotiationChoice],
                          Resolve(resolve_choice)],
        proposal_id: Annotated[str, Field(
            min_length=1, max_length=64, pattern=r"^[0-9a-f]+$",
            description="proposal_id from get_drift_report")],
        decision: Annotated[Literal["accept", "reject"] | None, Field(  # noqa: ARG001
            description="The user's decision. Omit to ask the user through the "
            "client's form, when it supports elicitation")] = None,
        persona_name: PersonaName = "",
        reason: Annotated[str, Field(  # noqa: ARG001
            max_length=500, description="Why, for a rejection")] = "",
        accepted_dimensions: Annotated[list[Dimension] | None, Field(  # noqa: ARG001
            min_length=1, max_length=len(list_dimensions()),
            description="Dimensions to decide on (default: all in the proposal)",
        )] = None,
    ) -> NegotiateUpdateResult:
        persona = persona_name or None
        if isinstance(choice, DeclinedElicitation | CancelledElicitation):
            with _errors("negotiate_update"):
                resolved = service.persona(persona)
            return {"persona": resolved, "proposal_id": proposal_id,
                    "decision": "declined" if isinstance(choice, DeclinedElicitation)
                    else "cancelled",
                    "dimensions": [], "commit": None, "reason": None}
        picked = choice.data
        hint = "Call get_drift_report for the current proposal id."
        try:
            if picked.decision == "accept":
                report = service.accept(persona, proposal_id, picked.dimensions)
            else:
                report = service.reject(persona, proposal_id, picked.reason,
                                        picked.dimensions)
        except KeyError:
            raise ToolError(f"unknown proposal {proposal_id}. {hint}") from None
        except (ValueError, SecurityError, TimeoutError, OSError) as exc:
            _fail("negotiate_update", exc, hint)
        return {"persona": report["persona"], "proposal_id": report["proposal_id"],
                "decision": report["status"],  # type: ignore[typeddict-item]
                "dimensions": report["dimensions"], "commit": report["commit"],
                "reason": report["reason"]}

    @tool("import_profile", "Import Profile",
          "Import a personality file (CLAUDE.md, soul.md, AGENTS.md and the like) "
          "as a persona profile. source_path must name a regular file of at most "
          "256 KiB inside the import root (PANMIND_IMPORT_ROOT, else the server's "
          "launch directory); paths or symlinks leading outside it are refused. "
          "Writes personas/<persona>.yaml, replacing a persona of the same name "
          "(the old version stays in git history), git-commits it and rewrites "
          "rendered/<persona>.md. Unless PANMIND_LLM=0, the file's text is sent "
          "to the configured model CLI for projection.",
          read_only=False, destructive=True, idempotent=False, open_world=True)
    def import_profile(
        source_path: Annotated[str, Field(
            min_length=1, max_length=1024,
            description="File to import, relative to the import root")],
        persona_name: Annotated[str, Field(
            pattern=f"^$|{SAFE_PERSONA_NAME_PATTERN.pattern}",
            max_length=50,
            description="Name for the persona (default: derived from the file)")] = "",
    ) -> ImportProfileResult:
        with _errors("import_profile"):
            name, content = read_import_source(source_path, root)
            return service.import_profile(  # type: ignore[return-value]
                Path(name), persona_name or None, content=content)

    @tool("export_profile", "Export Profile",
          "Export a persona's resolved profile as yaml or json dimensions, or as "
          f"SoulSpec markdown (at most {MAX_TEXT_CHARS} characters). Reads only.",
          read_only=True, destructive=False, idempotent=True)
    def export_profile(
        persona_name: PersonaName = "",
        format: Annotated[Literal["yaml", "json", "soulspec"], Field(  # noqa: A002
            description="Export format")] = "yaml",
        dimensions: Annotated[list[Dimension] | None, Field(
            min_length=1, max_length=len(list_dimensions()),
            description="Dimensions to include (default: all)")] = None,
    ) -> ExportProfileResult:
        with _errors("export_profile"):
            exported = service.export_profile(persona_name or None, format, dimensions)
        if "content" in exported:
            exported["content"], exported["truncated"] = _cap(exported["content"])
        return exported  # type: ignore[return-value]

    logger.info("PanMind MCP server created (config: %s, import root: %s)",
                panmind_dir, root)
    return mcp
