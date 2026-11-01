# claude-code2codex-proxy

Use **Claude Code with your ChatGPT Codex subscription**, keeping its tools,
skills, hooks, subagents, and editor integrations.

[claude-code2codex-proxy](https://github.com/comiam/claude-code2codex-proxy)
is an independent fork of [raine/claude-code-proxy](https://github.com/raine/claude-code-proxy).
The executable remains `claude-code-proxy`.

Compatibility is supported only for Codex. Kimi, Grok, OpenCode Go, and Cursor
Agent remain available without compatibility guarantees; use them at your own risk.

<img src="meta/claude-code-screenshot-2026-07.webp" alt="Claude Code running through claude-code-proxy" />

## Quick start with Codex

You need **Claude Code**, a recent stable **Rust toolchain with Cargo**, and a
**ChatGPT Plus or Pro account with Codex access**. An OpenAI Platform API key is
not a substitute for the ChatGPT login. You do not need the native Codex CLI.

**Windows:** follow the [Windows / PowerShell guide](docs/windows.md) for native
installation, client setup, `/proxy-speed`, and updates. The shell examples in
this README use macOS/Linux syntax.

### 1. Install this fork

Clone the fork and build it on macOS or Linux:

```sh
git clone https://github.com/comiam/claude-code2codex-proxy.git
cd claude-code2codex-proxy
cargo install --locked --path . --root "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
claude-code-proxy --version
```

Keep `~/.local/bin` on your shell's `PATH`. The upstream Homebrew package,
release installer, and release binaries install the original project, not this
fork's changes.

### 2. Sign in to ChatGPT

```sh
claude-code-proxy codex auth login
claude-code-proxy codex auth status
```

For a remote or headless machine, use device-code login instead:

```sh
claude-code-proxy codex auth device
```

Sign in through the proxy even if you already use the Codex CLI; their logins
are separate.

### 3. Start the proxy

In one terminal:

```sh
claude-code-proxy serve
```

The default address is `http://127.0.0.1:18765`. Leave this process running.
The interactive monitor shows sessions, requests, errors, token usage, cached
input tokens, and throughput.

For plain output, use `claude-code-proxy serve --no-monitor`. This still runs
in the foreground; use a service manager or terminal multiplexer if needed.
You can attach a separate dashboard with `claude-code-proxy monitor`.
Quitting an attached dashboard does not stop the server.

### 4. Launch Claude Code

In your project directory, in another terminal:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gpt-6.1-sol \
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna \
ENABLE_TOOL_SEARCH=false \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
  claude
```

`unused` is a placeholder for Claude Code, not an OpenAI key. Codex uses the
ChatGPT account you signed in to above. `ENABLE_TOOL_SEARCH=false` disables
Claude Code's deferred tool discovery; your configured MCP tools can still be
used. Disabling non-streaming fallback helps avoid duplicate tool calls after
an interrupted response.

Install the optional command below to switch Codex speed without changing models.

## Change speed in a conversation

### Install `/proxy-speed` once

From this repository's root, install the bundled command for your Claude Code
user profile. This leaves any existing copy untouched:

```sh
skill_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/proxy-speed"
if [ -e "$skill_dir/SKILL.md" ] || [ -L "$skill_dir/SKILL.md" ]; then
  printf 'Already exists; inspect before replacing: %s\n' "$skill_dir/SKILL.md"
else
  mkdir -p "$skill_dir"
  cp -n integrations/claude-code/skills/proxy-speed/SKILL.md "$skill_dir/SKILL.md"
fi
```

Restart Claude Code after installation. Alternatively, install the same file
at `.claude/skills/proxy-speed/SKILL.md` in the project where you use Claude
Code. Choose one scope, not both.

### Use it inside Claude Code

```text
/proxy-speed fast
/proxy-speed ultrafast
/proxy-speed off
```

| Command | Effect |
| --- | --- |
| `/proxy-speed fast` | Request Codex's `fast` tier for subsequent ordinary replies. |
| `/proxy-speed ultrafast` | Request Codex's `ultrafast` tier for subsequent ordinary replies. |
| `/proxy-speed off` | Stop requesting an accelerated tier, even if a model suffix or server default enables one. |

Wait for a confirmation such as **`Proxy speed: fast`**. The command itself
makes no Codex generation and does not change the selected model or reasoning
effort. It is separate from Claude Code's built-in `/fast`.

Keep these rules in mind:

- **Account support is decided by Codex.** A confirmation means the preference
  was saved, not that the API accepts the tier or guarantees a speedup. There
  is no silent downgrade if the API rejects your choice. Use
  `/proxy-speed off` and retry, or choose another supported mode.
- **Only this session changes.** Its subagents inherit the choice; other
  conversations and other providers are unaffected. Run the command in the
  main conversation. A reply already in progress keeps its previous setting.
- **The setting is temporary.** Select it again after 30 minutes idle or a
  proxy restart or capacity eviction, and when resuming a conversation if needed. `off` suppresses
  acceleration only while that session preference remains active; it does
  not erase server defaults.
- **Search and compaction keep their own settings.** The command controls
  ordinary Codex replies in Claude Code, not standalone web searches,
  `/compact`, or other clients using the optional OpenAI-compatible APIs.

The bundled command has been tested with Claude Code **2.1.316 on Linux**.
If Claude Code says the command is unavailable, check the installation scope
and restart the client; typing an unavailable command does not activate it.
See the [command guide](integrations/claude-code/README.md) for installation
and removal details.

## Models and reasoning effort

List the models supported by your installed proxy:

```sh
claude-code-proxy models
```

Useful Codex choices include:

| Model | Suggested use |
| --- | --- |
| `gpt-6-luna` | Small background tasks and lightweight work. |
| `gpt-6.1-terra` | Routine coding and exploration. |
| `gpt-6.1-sol` | General coding, debugging, and complex tasks. |
| `gpt-6-astra` | An optional choice for demanding tasks, subject to account access. |

In an existing Claude Code conversation:

```text
/model gpt-6.1-terra
/model gpt-6.1-sol
```

Use Claude Code's `/effort` command to choose reasoning effort independently
of the model and speed tier. Supported Codex effort values are `low`,
`medium`, `high`, `xhigh`, and `max`; availability can depend on the model.
For example, lower effort can suit a small edit while higher effort can suit
a difficult investigation.

### Speed through model names or server defaults

If you do not install `/proxy-speed`, model names can also request acceleration:

```text
/model gpt-6.1-sol-fast
/model gpt-6.1-sol-ultrafast
```

Every registered Codex model has a `-fast` form. The `-ultrafast` form is
currently registered for `gpt-6.1-sol` and `gpt-6-astra`, not Terra or Luna.

**The `-fast` suffix requests `priority`, whereas `/proxy-speed fast`
requests `fast` literally. They are not identical settings.** Account support
and provider limits still apply.

For a server-wide default, set the variable when starting the **proxy**, not
Claude Code:

```sh
CCP_CODEX_SERVICE_TIER=priority claude-code-proxy serve
```

An explicit `/proxy-speed` choice wins over the server default and model
suffixes for ordinary replies. Without a session choice, the server default
wins over suffixes. Leave the variable unset for standard service by default.

## Save your Claude Code configuration

If you want Claude Code to use this proxy by default, **merge** the following
into your existing `~/.claude/settings.json` (or the `settings.json` under your
`CLAUDE_CONFIG_DIR`). Do not replace unrelated permissions, hooks, or settings.

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:18765",
    "ANTHROPIC_AUTH_TOKEN": "unused",
    "ANTHROPIC_MODEL": "gpt-6.1-sol",
    "ANTHROPIC_SMALL_FAST_MODEL": "gpt-6-luna",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-6-luna",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-6.1-terra",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-6.1-sol",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "gpt-6.1-sol",
    "ENABLE_TOOL_SEARCH": "false",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1"
  }
}
```

The role defaults make Claude Code's Haiku/Sonnet/Opus/Fable selections use
these explicit Codex models. They do not select Anthropic models. Adjust them
to models available on your account and restart Claude Code after editing.

If you also use Claude Code directly with Anthropic, keep the per-launch
configuration from the quick start instead of changing your global settings.
Server options such as `CCP_CODEX_SERVICE_TIER` belong to the proxy process,
not this client configuration.

## Long conversations, `/compact`, and resume

Use Claude Code's normal `/compact`, optionally with a focus:

```text
/compact Keep the current task, decisions, changed files, and remaining checks.
```

Automatic compaction remains available. To resume later, launch
`claude --resume` with the same proxy environment, or simply use that command
if you saved the configuration above.

For a larger local context budget, add `[1m]` to the model name and set an
explicit compaction window appropriate for your model and account. For example,
these are **additional client settings** to merge into the `env` object above:

```json
{
  "ANTHROPIC_MODEL": "gpt-6.1-sol[1m]",
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "272000"
}
```

`272000` is a conservative example, not a universal Codex limit. `[1m]` does
not grant a one-million-token context window; keep the compaction window
within your account's actual limit. Leave automatic compaction enabled unless
you deliberately want to manage the limit yourself.

## Update an existing installation

First check which executable your shell actually uses:

```sh
command -v claude-code-proxy
```

From a clean checkout of this fork, pull your chosen branch and reinstall.
For the `~/.local/bin` installation shown above:

```sh
git pull --ff-only
cargo install --locked --path . --root "$HOME/.local" --force
```

If your executable is elsewhere, use the matching installation location or
adjust `PATH`; installing another copy does not update the one already in use.
Stop and restart the running proxy when you are ready to use the new build.
A running process does not switch to the new executable automatically. Check
the bundled `/proxy-speed` file against your installed copy when updating it.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Claude Code cannot connect | Keep `serve` running and check that `ANTHROPIC_BASE_URL` matches its address and port. |
| Login or token error | Run `claude-code-proxy codex auth status`; sign in again if needed. |
| Unknown or unavailable model | Run `claude-code-proxy models`; local support does not guarantee access on your ChatGPT account. |
| `/proxy-speed` unavailable | Check the installed skill path and restart Claude Code. |
| A speed tier is rejected | Run `/proxy-speed off`, then retry the task. No new conversation is required. |
| Changes seem absent after an update | Check `command -v claude-code-proxy` and restart the correct proxy process. |
| TLS decryption errors or repeated disconnects behind a VPN/proxy | Check and update the network tunnel before assuming Codex or this proxy is at fault. |

See [Troubleshooting](docs/src/content/docs/using/troubleshooting.md) for more.

## Other providers and optional APIs

| Provider | Account | Guide |
| --- | --- | --- |
| Codex | ChatGPT Plus or Pro with Codex access | [Codex](docs/src/content/docs/providers/codex.md) |
| Kimi | kimi.com with Kimi Code access | [Kimi](docs/src/content/docs/providers/kimi.md) |
| Grok | grok.com | [Grok](docs/src/content/docs/providers/grok.md) |
| OpenCode Go | OpenCode Go subscription | [OpenCode Go](docs/src/content/docs/providers/opencode-go.md) |
| Cursor Agent | Cursor account | [Cursor Agent](docs/src/content/docs/providers/cursor-agent.md) |

Optional OpenAI-compatible Responses and Chat Completions routes can be enabled
with `CCP_CODEX_RESPONSES_API=1` when starting the proxy. Image generation and
editing use `CCP_CODEX_IMAGES_API=1` and the same ChatGPT login; they consume
that account's image quota. See the [HTTP API guide](docs/src/content/docs/reference/http-api.md)
for usage and limits.

> [!WARNING]
> The proxy accepts requests without client authentication and binds to
> `127.0.0.1` by default. Keep it local. If you change the bind address, protect
> access with a firewall and authentication; anyone admitted can consume your
> subscription quota. Provider terms, model access, limits, and account
> enforcement still apply. This is an unofficial client.

## Documentation

Use the documentation in **this checkout** for fork-specific behavior:

- [Windows / PowerShell setup](docs/windows.md)
- [Configure Claude Code](docs/src/content/docs/using/configure-claude-code.md)
- [Models and routing](docs/src/content/docs/using/models-and-routing.md)
- [Monitor TUI](docs/src/content/docs/using/monitor-tui.md)
- [Command reference](docs/src/content/docs/reference/command-reference.md)
- [Configuration](docs/src/content/docs/reference/configuration.md)
- [Compatibility and limitations](docs/src/content/docs/reference/compatibility-and-limitations.md)

The [upstream documentation site](https://claude-code-proxy.raine.dev) covers
the original project and may differ from this fork.

## License

[MIT](LICENSE). Original project by [raine](https://github.com/raine) and
contributors; see [LICENSE](LICENSE) for attribution.
