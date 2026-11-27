# Changelog

## Unreleased

- The MCP server moved from FastMCP to the `mcp` 2.x SDK. It serves the
  2026-07-28 protocol (stateless, `server/discover`) beside the `initialize`
  handshake of 2024-11-05 through 2025-11-25, publishes tool listings as
  cacheable, and declares an icon, a title and server instructions.
- Tool failures are `isError` results with a hint; results no longer carry a
  `status` field. Every input is bounded in its schema (persona names, message
  count and size, proposal ids, dimensions as enums, formats), every output
  has a schema, and long text fields are capped at 64 KiB with a `truncated`
  flag.
- `list_personas` pages with `cursor` and `limit`.
- `observe_interaction` is marked destructive: a silent auto-accept rewrites
  the user profile, git-commits it and re-renders `rendered/<persona>.md`.
- `import_profile` reads only regular files of at most 256 KiB inside the
  import root (`PANMIND_IMPORT_ROOT`, else the launch directory), refuses
  traversal and symlink escapes, and returns relative paths.
- `negotiate_update` without a `decision` asks the user through form
  elicitation when the client supports it (`input_required` on 2026-07-28,
  `elicitation/create` before). The request state is sealed, bound to the
  call's arguments, expires after five minutes and is single-use. Declining
  or cancelling changes nothing.
- The `panmind` and `panmind-observer` skills are served over MCP as
  `skill://` resources and through the `io.modelcontextprotocol/skills`
  extension. `src/panmind/skills/` is their one source; the plugin and
  standalone copies are checked against it.
- File locks and git calls give up after 30 seconds instead of hanging.
- `.mcp.json` declares `"type": "stdio"` and passes `PANMIND_DIR` through; an
  empty `PANMIND_DIR` means the default `~/.panmind`.
- Added `server.json` for the MCP registry.

## 0.5.0

- Renamed Helios to PanMind under PanStudio, Apache-2.0.
