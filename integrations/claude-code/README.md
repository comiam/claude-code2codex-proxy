# Claude Code integrations

- [`/proxy-speed`](#proxy-speed-command) changes the current session's speed preference.
- [Codex quota status line](#codex-quota-status-line) displays account quota in Claude Code, with Linux/WSL and native Windows helpers.

## `/proxy-speed` command

This optional user-invoked command saves a speed preference for the current
Claude Code session in claude-code-proxy. It does not change the selected model
or reasoning effort and is separate from Claude Code's built-in `/fast` toggle.
Use a proxy build with session speed controls and route the session to Codex.

### Install

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

### Use

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

### Verify without a real provider

From the repository root:

```sh
cargo test --offline --locked --test proxy_speed_command -- --ignored --nocapture
```

The opt-in test uses the installed Claude Code, temporary project/profile
files, dummy credentials, and a loopback stub. It does not install into the
active profile or send inference requests to a real provider. Set
`CCP_CLAUDE_BIN=/absolute/path/to/claude` only if the executable is not available
on the test process's `PATH`.

### Uninstall or roll back

Inspect the installed target, then remove only the copy installed above and
restart Claude Code. Do not remove another command or a directory containing
unrelated files. Removing the skill does not clear a preference already stored
in the proxy. Use `off` first to suppress acceleration, or allow the preference
to expire. Restarting the proxy clears its session store and should be a
deliberate operational action.

## Codex quota status line

[`codex-status.sh`](codex-status.sh) (Linux/WSL) and
[`codex-status.ps1`](codex-status.ps1) (native Windows) display the remaining
percentage and reset countdown for Codex's 5-hour and 7-day account quota
windows. Exhausted windows
show `RATE LIMITED` and, when available, their reset time in the machine's local
timezone (or `TZ`). The helper reads `http://127.0.0.1:18765/usage`; it does not
read credentials or call a provider. It is independent of `/proxy-speed`.

### Linux / WSL requirements and installation

Use **Bash, curl, jq, awk, and GNU `date`** on Linux or WSL. The script uses
GNU `date -d`; macOS requires GNU coreutils with its `date` available under that
name on Claude Code's `PATH`. Native PowerShell and macOS's BSD `date` are not
compatible with this helper.

From the repository root:

```sh
claude_config="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
if [ -e "$claude_config/codex-status.sh" ] || [ -L "$claude_config/codex-status.sh" ]; then
  printf 'Already exists; inspect before replacing: %s\n' "$claude_config/codex-status.sh"
else
  mkdir -p "$claude_config" &&
    cp -n integrations/claude-code/codex-status.sh "$claude_config/codex-status.sh" &&
    chmod +x "$claude_config/codex-status.sh"
fi
```

Merge this into the user profile's `settings.json`, preserving unrelated keys:

```json
{
  "statusLine": {
    "type": "command",
    "command": "~/.claude/codex-status.sh",
    "refreshInterval": 10
  }
}
```

The default settings path is `~/.claude/settings.json`. If you use
`CLAUDE_CONFIG_DIR`, edit its `settings.json` and replace `command` with the
installed script's absolute path. Paths containing spaces need shell quotes
inside the JSON string. A profile has one `statusLine` command; preserve or
combine an existing status line rather than adding a duplicate JSON key.

The interval re-runs the helper every ten seconds, including while idle, in
addition to event-driven updates. It reads the proxy's latest in-memory quota
snapshot and updates countdowns; it does not refresh quota upstream. A proxy
restart clears the snapshot. Percentages remain the last observed values until
Codex supplies another event; expired windows show `awaiting refresh`.

### Check and customize

With the proxy running, test your installed copy:

```sh
claude_config="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
"$claude_config/codex-status.sh" </dev/null
```

For a different proxy port, set `CODEX_USAGE_URL` in the Claude Code process's
environment or use this status-line command:

```json
{
  "statusLine": {
    "type": "command",
    "command": "CODEX_USAGE_URL=http://127.0.0.1:18766/usage ~/.claude/codex-status.sh",
    "refreshInterval": 10
  }
}
```

The helper bypasses configured HTTP proxies for this local request. It uses a
0.3-second connection timeout and a one-second total timeout.

- `jq not installed`: install `jq` and ensure Claude Code can find it on `PATH`.
- `quota unavailable`: check that the proxy is running and the URL is correct.
- `quota waiting`: no usable quota data is available yet. Normal Codex activity
  can populate it; availability depends on upstream quota events and transport.
- `awaiting refresh`: a reported reset has passed; wait for a newer snapshot
  rather than treating the old window as a current measurement.

### Native Windows installation

Use [`codex-status.ps1`](codex-status.ps1) with Windows PowerShell 5.1 and .NET
on Windows 10/11. It needs no Bash, curl, jq, or GNU tools. The HTTP request
bypasses proxies and has a one-second timeout. The script respects your existing
PowerShell execution policy; no bypass is included.

Follow the [Windows installation guide](../../docs/windows.md#optional-codex-quota-status-line)
to copy the `.ps1` into your profile and configure `statusLine` to run
`powershell.exe -NoProfile -NonInteractive -File` with the installed path.
The Bash installation above is for Linux/WSL. Native Windows execution has not
been verified from Linux.

### Update or remove

Inspect and compare an installed copy before replacing it with the bundled
script. Cargo installation updates the proxy binary, not this helper.
To disable it, remove only the `statusLine` setting added above, or restore your
previous command. After inspecting the target, remove only the installed
`codex-status.sh` or `codex-status.ps1` if it is no longer used. No proxy restart
is needed.

For the complete client setup, see
[Configure Claude Code](../../docs/src/content/docs/using/configure-claude-code.md).
For precedence and provider boundaries, see
[Models and routing](../../docs/src/content/docs/using/models-and-routing.md).
