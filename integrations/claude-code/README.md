# Claude Code `/proxy-speed`

This optional user-invoked command saves a speed preference for the current
Claude Code session in claude-code-proxy. It does not change the selected model
or reasoning effort and is separate from Claude Code's built-in `/fast` toggle.
Use a proxy build with session speed controls and route the session to Codex.

## Install

The supplied file is `skills/proxy-speed/SKILL.md`. Install it in one scope:

- User: `${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/proxy-speed/SKILL.md`.
- Project: `.claude/skills/proxy-speed/SKILL.md` in the project used by Claude Code.

Inspect any existing target before replacing it. From the repository root, this
creates a user-scoped copy only if the target does not already exist:

```sh
skill_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/proxy-speed"
if [ -e "$skill_dir/SKILL.md" ]; then
  printf 'Already exists; inspect before replacing: %s\n' "$skill_dir/SKILL.md"
else
  mkdir -p "$skill_dir"
  cp -n integrations/claude-code/skills/proxy-speed/SKILL.md "$skill_dir/SKILL.md"
fi
```

Restart Claude Code after installation. Project discovery may require an
explicit `--add-dir /absolute/path/to/project`. The isolated compatibility test
covers normal client context as well as `--bare --add-dir` on Claude Code
2.1.316, Linux. The normal-client test uses `gpt-6.1-sol` and verifies its Lite
wire tier. Other versions and platforms have not been established by these tests.

No hook, shell helper, new HTTP endpoint, or `settings.json` modification is
required. If the client reports that `/proxy-speed` is unavailable, the command
did not run. Plain slash-command text is not interpreted as proxy control.

## Use

```text
/proxy-speed off
/proxy-speed fast
/proxy-speed ultrafast
```

Wait for the local proxy acknowledgment. It confirms the saved preference, not
API support or measured acceleration. The control request itself does not call
Codex. If the API rejects a tier, select another mode or `off` and retry; a
new client process or clearing failed input is not required.

| Mode | Ordinary Codex Messages generation |
| --- | --- |
| `off` | Omits `service_tier`, overriding tier settings and model suffixes. |
| `fast` | Sends the literal `service_tier: "fast"`. |
| `ultrafast` | Sends the literal `service_tier: "ultrafast"`. |

The API decides whether a tier is accepted. The command does not check the local
model capability catalog, convert `fast` to `priority`, downgrade `ultrafast`, or
retry an API rejection at another tier. A session choice takes precedence over
global settings and model suffixes. Without one, global settings take precedence
over suffixes; the `-fast` suffix requests `priority`.

Only the main conversation can change the preference. Agent requests with the
same session ID inherit it; other sessions and other providers do not. A new
value applies to subsequently accepted requests, while in-flight requests and
their retries keep their captured value. A tier change can cause the normal
continuation checks to send full context.

Preferences are stored only in memory and disappear on 30 minutes idle,
capacity eviction, or proxy restart. Old control messages in history do not
restore them. In particular, `off` does not remove global configuration: that
configuration becomes effective again when the explicit session preference
expires.

The command controls ordinary Codex generations through Anthropic Messages,
including both full Responses and Responses Lite models such as `gpt-6.1-sol`.
Both paths forward the selected literal tier. The session command does not
control standalone search, compaction, or native Responses and Chat Completions
routes.

## Verify without a real provider

From the repository root:

```sh
cargo test --offline --locked --test proxy_speed_command -- --ignored --nocapture
```

The opt-in test uses the installed Claude Code, temporary project/profile
files, dummy credentials, and a loopback stub. It does not install into the
active profile or send inference requests to a real provider. Set
`CCP_CLAUDE_BIN=/absolute/path/to/claude` only if the executable is not available
on the test process's `PATH`.

## Uninstall or roll back

Inspect the installed target, then remove only the copy installed above and
restart Claude Code. Do not remove another command or a directory containing
unrelated files. Removing the skill does not clear a preference already stored
in the proxy. Use `off` first to suppress acceleration, or allow the preference
to expire. Restarting the proxy clears its session store and should be a
deliberate operational action.

For the complete client setup, see
[Configure Claude Code](../../docs/src/content/docs/using/configure-claude-code.md).
For precedence and provider boundaries, see
[Models and routing](../../docs/src/content/docs/using/models-and-routing.md).
