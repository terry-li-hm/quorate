# Local workspace mode

## Outcome

`quorate quick`, `quorate brainstorm`, and `quorate council` gain an opt-in
`--local --workspace <directory>` path. The name describes workspace grounding,
not local inference: the selected text still leaves the machine and is sent to
the Codex and Gemini subscription services.

The mode is deliberately separate from Quorate's normal provider router. It
uses exactly two seats:

- Codex CLI with an ephemeral, read-only sandbox.
- Gemini through Quorate's established Antigravity CLI route in sandboxed plan
  mode.

Neither seat may fall back to an API or a third model family.

## Data boundary

Both flags are required together. Before invoking either client, Quorate:

1. resolves the workspace to an existing directory;
2. rejects a workspace inside a configured `QUORATE_PROTECTED_ROOTS` entry;
3. rejects a workspace that contains a configured protected root;
4. rejects symbolic links among files selected for the snapshot;
5. rejects common credential filenames and key-file suffixes;
6. reads only regular UTF-8 text files, excluding generated and dependency
   directories; and
7. refuses a snapshot above 200 files, 64 KiB per text file, or 128 KiB of
   aggregate text.

The clients never receive the workspace path and never run inside the user
workspace. Quorate embeds the validated relative paths and file contents in the
prompt, labels them as untrusted data, and launches each client from an isolated
temporary directory. The ordinary question, context, and persona file guards
remain in force.

This protects configured local roots and limits ambient filesystem authority.
It does not make private material suitable for external disclosure. A local
workspace must still be public, synthetic, or deliberately sanitized.

## Mode behavior

- `quick` asks both seats independently and requires both to succeed.
- `brainstorm` uses the two seats for generation and cross-pollination, then
  Codex curates with Gemini as the contract fallback.
- `council` uses both seats for the blind and debate phases, Codex as judge, and
  Gemini as critic. Judge fallback is Gemini and critic fallback is Codex.
- Existing non-local model rosters, provider fallbacks, output envelopes, run
  logs, and command behavior remain unchanged.

## Acceptance criteria

- Unit tests cover flag pairing, protected-root overlap in both directions,
  symlink rejection, binary exclusion, snapshot limits, direct client routing,
  strict two-seat quorum, and injected runner behavior in all three modes.
- The complete Quorate test suite passes.
- A synthetic workspace token is recovered through both Codex and
  Gemini-through-Antigravity using the production local runner.
- The installed `quorate` wrapper passes its data-boundary regression and a
  synthetic `--local` smoke test.
- The final diff contains only the local-mode implementation, tests, and
  documentation.
