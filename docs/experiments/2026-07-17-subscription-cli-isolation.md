# Subscription CLI isolation audit

## Finding

Kimi Code already received an allowlisted environment, but Codex and Antigravity inherited every credential injected for Quorate's API routes. Claude removed only `ANTHROPIC_API_KEY`, leaving unrelated provider credentials available to its subprocess. Codex, Claude, and Antigravity also started from the caller's working directory rather than an empty workspace.

No credential disclosure was observed. The issue was unnecessary ambient authority: a tool call or prompt injection could potentially reach secrets and local context that the model did not need.

## Change

All four subscription CLIs now receive the same allowlisted runtime environment, retaining only basic process, certificate, locale, and client configuration paths. They no longer inherit provider keys. Every route starts in a temporary workspace. Claude's tool set is empty, Codex runs ephemerally with a read-only sandbox and no repository rules, Antigravity retains sandboxed plan mode, and Kimi Code retains its empty skills directory. K3 continues to fail closed instead of falling through to an unrelated API route.

## Verification

A live four-route probe asked K3, Fable 5, GPT-5.6 Sol, and Gemini 3.5 Flash for one exact token. All four returned correctly through `kimi-code`, `claude-print`, `codex-exec`, and `antigravity-cli` respectively, without fallback diagnostics. Unit tests verify that unrelated provider secrets are absent from the child environment and that every subscription route is priced at zero marginal API cost.

The clients still receive their home and client-configuration paths because subscription authentication requires them. Tool restrictions and temporary workspaces provide the corresponding filesystem boundary.
