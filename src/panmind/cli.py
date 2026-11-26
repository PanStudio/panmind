"""Command line interface: the MCP server by default, plus inspection commands.

Every subcommand is a thin layer over PanMindService. ``--panmind-dir`` works both
before the subcommand and after it; the later one wins.
"""

import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml

from . import __version__
from .bootstrap import BootstrapManager
from .security import SecurityError
from .service import PanMindService

logger = logging.getLogger(__name__)

def _default_panmind_dir() -> Path:
    return Path(os.getenv("PANMIND_DIR", str(Path.home() / ".panmind")))


def panmind_dir_option[F: Callable[..., Any]](func: F) -> F:
    return click.option(
        "--panmind-dir", "panmind_dir", default=None,
        type=click.Path(path_type=Path),
        help="PanMind directory (default: the group option, PANMIND_DIR, ~/.panmind)",
    )(func)


def _service(panmind_dir: Path | None) -> PanMindService:
    ctx = click.get_current_context()
    root = panmind_dir or ctx.find_root().obj["panmind_dir"]
    return PanMindService(root)


@click.group(invoke_without_command=True)
@click.option(
    "--panmind-dir", default=_default_panmind_dir,
    type=click.Path(path_type=Path),
    help="PanMind directory (default: ~/.panmind, env: PANMIND_DIR)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.version_option(version=__version__, prog_name="panmind")
@click.pass_context
def main(ctx: click.Context, panmind_dir: Path, verbose: bool) -> None:
    """PanMind: learn an agent's behavioral preferences and negotiate updates.

    Run without a subcommand to start the MCP server over stdio.
    """
    level = logging.DEBUG if verbose else getattr(
        logging, os.getenv("PANMIND_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["panmind_dir"] = panmind_dir
    if ctx.invoked_subcommand is not None:
        return
    try:
        BootstrapManager(panmind_dir).update_last_boot()
        asyncio.run(_run_server(panmind_dir))
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise SystemExit(1) from e


async def _run_server(panmind_dir: Path) -> None:
    from .server import create_server

    server = await create_server(panmind_dir)
    await server.run_async()


@main.command("status")
@panmind_dir_option
def status_command(panmind_dir: Path | None) -> None:
    """Show each persona's resolved profile, evidence and pending proposal."""
    service = _service(panmind_dir)
    click.echo(f"PanMind directory: {service.panmind_dir}")
    click.echo(f"Default persona: {service.persona(None)}")
    for persona in service.list_personas():
        report = service.drift_report(persona)
        pending = (f"{report['proposal_id']} ({report['proposal_tier']})"
                   if report["proposal_id"] else "none")
        click.echo(f"\n{persona}: {report['turns']} turns, pending proposal: {pending}")
        for dim, r in report["endorsed"].items():
            declared, observed = r["dominant_declared"], r["dominant_observed"]
            tier = f", {r['tier']}" if r["tier"] else ""
            explicit = ("  declared: explicit signals only"
                        if dim in report["explicit_only"] else "")
            click.echo(
                f"  {dim:<24} declared {declared} {r['declared'][declared]:.2f} "
                f"-> observed {observed} {r['posterior_mean'][observed]:.2f}  "
                f"(JS {r['divergence']:.4f}, credible {r['credibility']:.0%}{tier})"
                + explicit
            )


@main.command("negotiate")
@click.argument("persona", required=False)
@click.option("--accept", "-y", "--yes", "accept", is_flag=True,
              help="Accept the pending proposal without prompting.")
@click.option("--reject", "reject", is_flag=True,
              help="Reject the pending proposal without prompting.")
@click.option("--reason", default="", help="Reason recorded with a rejection.")
@panmind_dir_option
def negotiate_command(persona: str | None, accept: bool, reject: bool,
                      reason: str, panmind_dir: Path | None) -> None:
    """Show the drift report for PERSONA and accept or reject its proposal."""
    if accept and reject:
        raise click.UsageError("--accept and --reject are mutually exclusive")
    service = _service(panmind_dir)
    report = service.drift_report(persona)
    click.echo(report["summary"])
    click.echo("")
    for dim, line in report["per_dimension"].items():
        click.echo(f"  {dim:<24} {line}")

    proposal_id = report["proposal_id"]
    if proposal_id is None:
        return
    if accept:
        action = "A"
    elif reject:
        action = "R"
    else:
        action = click.prompt("[A]ccept  [R]eject  [Q]uit", default="Q",
                              show_default=False).strip().upper()
    if action == "A":
        decided = service.accept(report["persona"], proposal_id)
        click.echo(f"Accepted {', '.join(decided['dimensions'])}; "
                   f"commit {decided['commit'] or 'not recorded'}.")
    elif action == "R":
        decided = service.reject(report["persona"], proposal_id, reason)
        click.echo(f"Rejected {', '.join(decided['dimensions'])}.")
    else:
        click.echo(f"No action taken. Proposal {proposal_id} stays pending.")


@main.command("ingest")
@click.option("--persona", default=None, help="Persona (default: default_persona)")
@click.option("--session-id", required=True, help="Claude Code session id")
@click.option("--transcript", required=True, type=click.Path(path_type=Path),
              help="Path to the session transcript JSONL")
@click.option("--final", is_flag=True,
              help="Session ended: also run the batch LLM labeler.")
@panmind_dir_option
def ingest_command(persona: str | None, session_id: str, transcript: Path,
                   final: bool, panmind_dir: Path | None) -> None:
    """Label a session transcript's agent turns into the observation ledger.

    Idempotent, so it is safe on every Stop. A missing transcript is a no-op.
    """
    if not transcript.is_file():
        logger.debug("transcript %s not there yet; nothing to ingest", transcript)
        return
    try:
        report = _service(panmind_dir).ingest(persona, transcript, session_id, final)
    except (RuntimeError, SecurityError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(report))


@main.command("render")
@click.argument("persona", required=False)
@click.option("--model", default=None,
              help="Print the context for this model id instead of writing the file.")
@panmind_dir_option
def render_command(persona: str | None, model: str | None,
                   panmind_dir: Path | None) -> None:
    """Write rendered/<persona>.md, the context injected at SessionStart."""
    service = _service(panmind_dir)
    if model:
        click.echo(service.render_text(persona, model), nl=False)
    else:
        click.echo(str(service.render(persona)))


@main.group("persona")
def persona_group() -> None:
    """Persona settings."""


@persona_group.command("default")
@click.argument("name", required=False)
@panmind_dir_option
def persona_default_command(name: str | None, panmind_dir: Path | None) -> None:
    """Show the default persona, or set it to NAME."""
    service = _service(panmind_dir)
    if name is None:
        click.echo(service.persona(None))
        return
    try:
        click.echo(f"Default persona: {service.set_default_persona(name)}")
    except SecurityError as e:
        raise click.BadParameter(str(e), param_hint="NAME") from e


@main.command("export")
@click.argument("persona", required=False)
@click.option("--format", "fmt", type=click.Choice(["yaml", "json", "soulspec"]),
              default="yaml", help="Export format (default: yaml)")
@click.option("--dimensions", "dims", default=None,
              help="Comma-separated dimension names to export")
@panmind_dir_option
def export_command(persona: str | None, fmt: str, dims: str | None,
                   panmind_dir: Path | None) -> None:
    """Export PERSONA's resolved behavioral profile."""
    dim_list = [d.strip() for d in dims.split(",")] if dims else None
    try:
        exported = _service(panmind_dir).export_profile(persona, fmt, dim_list)
    except (ValueError, SecurityError) as e:
        raise click.ClickException(str(e)) from e
    if fmt == "soulspec":
        click.echo(exported["content"])
    elif fmt == "json":
        click.echo(json.dumps(exported, indent=2))
    else:
        click.echo(yaml.safe_dump(exported, default_flow_style=False, sort_keys=False))


@main.command("init")
@click.argument("persona", required=False)
@click.option("--home", type=click.Path(path_type=Path), default=None,
              help="Claude home to read declared sources from (default: ~/.claude)")
@click.option("--project-dir", "project_dir", type=click.Path(path_type=Path),
              default=None,
              help="Project dir for a local output style (default: cwd)")
@panmind_dir_option
def init_command(persona: str | None, home: Path | None, project_dir: Path | None,
                 panmind_dir: Path | None) -> None:
    """Bootstrap PanMind and onboard PERSONA (default: developer) from CLAUDE.md
    and the active output style.

    Imports both declared sources into PERSONA's user level as authoritative,
    sets it as the default persona, and renders its context. Safe to rerun.
    """
    from .onboard import onboard

    try:
        result = onboard(_service(panmind_dir), persona, home,
                         project_dir or Path.cwd())
    except (ValueError, SecurityError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Persona: {result['persona']} (default)")
    click.echo("Imported from: " + ", ".join(result["sources"]))
    for dim, summary in result["distributions"].items():
        click.echo(f"  {dim:<24} {summary['dominant']} "
                   f"(entropy {summary['entropy']:.2f})")
    if result["kept_accepted"]:
        click.echo("Kept your accepted " + ", ".join(result["kept_accepted"]))
    click.echo(f"Saved to: {result['saved_to']}")
    click.echo(f"Rendered to: {result['rendered_to']}")


@main.command("import")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--persona", default=None,
              help="Name for the imported persona (default: from filename)")
@click.option("--format", "fmt",
              type=click.Choice(["auto", "claude", "soul", "agents", "gemini",
                                 "generic"]),
              default="auto", help="Personality file format (default: auto-detect)")
@panmind_dir_option
def import_command(path: Path, persona: str | None, fmt: str,
                   panmind_dir: Path | None) -> None:
    """Import a personality file into PanMind behavioral distributions."""
    try:
        result = _service(panmind_dir).import_profile(path, persona, fmt)
    except (ValueError, SecurityError) as e:
        raise click.ClickException(f"Import failed: {e}") from e
    click.echo(f"Imported from: {path}")
    click.echo(f"Persona: {result['persona']}")
    for dim, summary in result["distributions"].items():
        click.echo(f"  {dim:<24} {summary['dominant']} "
                   f"(entropy {summary['entropy']:.2f})")
    click.echo(f"Saved to: {result['saved_to']}")


if __name__ == "__main__":
    main()
