---
title: Configure Claude Code
description: Set Claude Code client variables for claude-code-proxy without mixing them with CCP proxy configuration.
---

Claude Code reads its API connection when the process starts. These variables belong to **Claude Code**, not to the proxy server.

## Minimal client contract

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gpt-6-sol[1m] \
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m] \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
  claude
```

| Claude Code variable | Purpose |
| --- | --- |
| `ANTHROPIC_BASE_URL` | Sends Anthropic API requests to the local proxy. |
| `ANTHROPIC_AUTH_TOKEN` | Satisfies Claude Code's client credential requirement. The proxy does not use it for upstream auth. |
| `ANTHROPIC_MODEL` | Selects the main request model and therefore the provider. |
| `ANTHROPIC_SMALL_FAST_MODEL` | Selects the model for title generation, token-related background work, and other small requests. Use a model the proxy routes. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Reduces background traffic sent to the subscription provider. |
| `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1` | Prevents Claude Code from retrying a partially completed stream as non-streaming, which can duplicate tool calls. |

The proxy always makes streaming upstream requests. It can still accumulate a non-streaming Anthropic response when the client requests one.

## Compaction settings

A trailing `[1m]` tells Claude Code to use its larger local context policy. The proxy strips the suffix before upstream routing. It does not enlarge the provider's context window.

For example, `CLAUDE_CODE_AUTO_COMPACT_WINDOW=272000` is a conservative starting point for some Codex models. Choose a safe value for the actual model and account limit; `[1m]` does not grant a one-million-token context. `DISABLE_AUTO_COMPACT=1` disables automatic compaction while preserving manual `/compact`, but the session can then hit the upstream limit.

## Persistent Claude Code settings

If every Claude Code session should use the proxy, put client variables in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:18765",
    "ANTHROPIC_AUTH_TOKEN": "unused",
    "ANTHROPIC_MODEL": "gpt-6-sol[1m]",
    "ANTHROPIC_SMALL_FAST_MODEL": "gpt-6-luna[1m]",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": 272000,
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": 1
  }
}
```

Use process-level variables or a wrapper when you also launch Claude Code directly against Anthropic. See [Switching models and backends](switching-models-and-backends.md).

## Install the proxy-speed command

The repository includes an explicitly user-invoked command at `integrations/claude-code/skills/proxy-speed/SKILL.md`. It adds `/proxy-speed off|fast|ultrafast`; no hook, shell helper, or `settings.json` change is required. Use it with a proxy version that implements session speed controls and a session routed to Codex.

For a new user-scoped installation, run this from the repository root. An existing command is left untouched so you can inspect and compare it before replacement:

```sh
skill_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/proxy-speed"
if [ -e "$skill_dir/SKILL.md" ]; then
  printf 'Already exists; inspect before replacing: %s\n' "$skill_dir/SKILL.md"
else
  mkdir -p "$skill_dir"
  cp -n integrations/claude-code/skills/proxy-speed/SKILL.md "$skill_dir/SKILL.md"
fi
```

Alternatively, put the same file at `.claude/skills/proxy-speed/SKILL.md` in the project where you use Claude Code. Install in one scope rather than creating competing copies. Restart the client after installation. If project discovery needs an explicit directory, launch with `claude --add-dir /absolute/path/to/project`.

```text
/proxy-speed fast
/proxy-speed ultrafast
/proxy-speed off
```

Wait for the proxy's local confirmation before assuming a preference was saved. If Claude Code reports that the command is unavailable, it did not run: an unexpanded `/proxy-speed fast` is ordinary text, not a fallback control protocol. The isolated compatibility tests verify Claude Code 2.1.316 on Linux in both normal and `--bare` mode with a temporary project and `--add-dir`. The normal-client test uses `gpt-6.1-sol`, including its Responses Lite tier. Other client versions and platforms are not guaranteed by these tests.

`fast` and `ultrafast` are sent literally to the API on ordinary Codex generation requests; the command does not change the model or effort and does not validate API tier support locally. See [per-session speed behavior](models-and-routing.md#per-session-proxy-speed) for precedence, agent inheritance, lifecycle, errors, and special-path limitations.

To uninstall, inspect the installed file, then remove only the copy you installed and restart Claude Code. Removing the file does not clear an existing proxy preference. Use `off` first to suppress acceleration, or allow the session preference to expire; restarting the proxy clears all in-memory session preferences and should be a deliberate operational action.

## Proxy settings are separate

`CCP_*`, `PORT`, and `config.json` configure the **claude-code-proxy server process**. They control the listener, provider endpoints, transport, credentials, and diagnostics. They do not belong in Claude Code's client environment unless the same shell also starts the proxy.

See [Configuration](../reference/configuration.md) for the canonical server setting table.
