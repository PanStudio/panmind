# Contributing to PanMind

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Running the test suite

```bash
uv run --frozen pytest
```

## Linting and type checking

```bash
uv run ruff check .
uv run mypy src
```

## Skills

`src/panmind/skills/<name>/SKILL.md` is the one source for PanMind's skills;
the server serves those files over MCP. The copies other installs read are
derived from it, and `tests/test_plugin_structure.py` fails when they drift:

- `panmind-skill/SKILL.md` is the canonical `panmind` skill, byte for byte.
- `panmind-plugin/skills/panmind/SKILL.md` is the canonical skill with
  `mcp__panmind__` replaced by `mcp__plugin_panmind_panmind__`, the name a
  plugin-loaded server's tools carry.
- `panmind-plugin/agents/panmind-observer.md` is the canonical
  `panmind-observer` skill, byte for byte.

Edit the canonical file, then regenerate the copies:

```bash
cp src/panmind/skills/panmind/SKILL.md panmind-skill/SKILL.md
sed 's/mcp__panmind__/mcp__plugin_panmind_panmind__/g' \
  src/panmind/skills/panmind/SKILL.md > panmind-plugin/skills/panmind/SKILL.md
cp src/panmind/skills/panmind-observer/SKILL.md panmind-plugin/agents/panmind-observer.md
```

## Pre-commit hooks

Install the git hooks so ruff and the standard hygiene checks run automatically before each commit:

```bash
uvx pre-commit install
```
