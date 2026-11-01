# Windows / PowerShell guide

Run Claude Code with your ChatGPT Codex subscription through **this fork** of
claude-code-proxy on native Windows.

All shell examples below are for **PowerShell**, not Command Prompt (`cmd.exe`),
Git Bash, or WSL. Windows Terminal is a convenient way to keep separate tabs
for the proxy and Claude Code. No WSL installation is required.

> **Verification scope:** These are native Windows setup instructions, not a
> report of a Windows end-to-end test. This fork's bundled `/proxy-speed`
> integration has been tested with Claude Code 2.1.316 on Linux; Windows client
> compatibility has not been established by that test.

## 1. Install the prerequisites

You need:

- A **ChatGPT Plus or Pro account with Codex access**. An OpenAI Platform API
  key does not replace this subscription login.
- **Rust and Cargo**, installed with [rustup](https://www.rust-lang.org/tools/install).
  Download and run `rustup-init.exe`, using the stable MSVC toolchain.
- **Visual Studio C++ Build Tools** and a **Windows SDK**. Let rustup install
  prerequisites when offered, or use the Visual Studio Installer and select
  **Desktop development with C++**, including MSVC and a Windows SDK. You do
  not need the full Visual Studio IDE. See the
  [official MSVC prerequisites](https://rust-lang.github.io/rustup/installation/windows-msvc.html).
- **Claude Code**, using its native Windows installation.
- **Git for Windows** if you want to clone and update with Git. It is also
  recommended for Claude Code's Bash tool; current Claude Code can use
  PowerShell without Git Bash. Get it from [Git for Windows](https://git-scm.com/downloads/win).

If Claude Code is not installed, its
[official PowerShell installer](https://code.claude.com/docs/en/setup) is:

```powershell
irm https://claude.ai/install.ps1 | iex
```

This downloads and runs Anthropic's installer. The native installation does
not require Node.js or npm. You do not need the native Codex CLI either.

After installing prerequisites, open a **new PowerShell window** so that PATH
changes take effect, then check:

```powershell
rustc --version
cargo --version
claude --version
```

If you installed Git, check `git --version` too. If `claude` is not found,
check whether its native executable exists:

```powershell
Test-Path "$env:USERPROFILE\.local\bin\claude.exe"
```

If it exists, add `%USERPROFILE%\.local\bin` to your **user Path** in Windows
Environment Variables, then open a new terminal. Do not replace your entire
Path with that one entry.

## 2. Build and install this fork

Clone the fork with Git, or download and extract a source ZIP from
[its GitHub repository](https://github.com/comiam/claude-code2codex-proxy):

```powershell
git clone https://github.com/comiam/claude-code2codex-proxy.git
```

The upstream release installer and binaries do not contain this fork's changes.

Open PowerShell in the extracted or cloned repository root: the folder must
contain `Cargo.toml` and `integrations`. For example, adjust this path to your
checkout:

```powershell
Set-Location 'C:\src\claude-code2codex-proxy'
cargo install --locked --path . --root "$env:USERPROFILE\.cargo"
```

Continue only after Cargo reports a successful installation. The expected
executable for this command is
`%USERPROFILE%\.cargo\bin\claude-code-proxy.exe`.

Make that location first on PATH in the **current** PowerShell window, then
check what will run:

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
Get-Command claude-code-proxy | Select-Object Source
claude-code-proxy --version
```

For future terminals, make sure `%USERPROFILE%\.cargo\bin` is on your user
Path; rustup normally adds it. If another proxy installation appears first,
resolve that PATH conflict rather than assuming the newly built copy is used.

If compilation reports that `link.exe` or a Windows SDK is missing, install
or repair the C++ workload from step 1, open a new terminal, and retry. An
upstream prebuilt binary is not a substitute for building this fork.

## 3. Sign in to ChatGPT

```powershell
claude-code-proxy codex auth login
claude-code-proxy codex auth status
```

Complete the browser login with the ChatGPT account that has Codex access.
If browser login is inconvenient, use device-code login instead:

```powershell
claude-code-proxy codex auth device
```

The proxy has its own login, separate from any native Codex CLI installation.
Do not paste a ChatGPT token or an OpenAI key into the client settings below.

## 4. Start the proxy in the first terminal

```powershell
claude-code-proxy serve --no-monitor
```

Keep this terminal open. The server listens on `http://127.0.0.1:18765`.
`Ctrl+C` stops it. `--no-monitor` keeps the output plain; it does not start a
background service.

For the interactive dashboard, either omit `--no-monitor` or attach from a
separate terminal with `claude-code-proxy monitor`. Quitting an attached
monitor does not stop the server.

Keep the listener on loopback. You do not need to open an inbound firewall
port or expose port 18765 to your LAN or the internet. The proxy does not
authenticate callers, and anyone with access can consume your subscription.

## 5. Launch Claude Code in a second terminal

Open **another PowerShell tab**, go to the project you want to work on, and
set the client environment before launching Claude Code:

```powershell
Set-Location 'C:\src\my-project'
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:18765'
$env:ANTHROPIC_AUTH_TOKEN = 'unused'
$env:ANTHROPIC_MODEL = 'gpt-6.1-sol'
$env:ANTHROPIC_SMALL_FAST_MODEL = 'gpt-6-luna'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'gpt-6-luna'
$env:ENABLE_TOOL_SEARCH = 'false'
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
$env:CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK = '1'
claude
```

Replace the example project path with your own. `unused` is a placeholder for
Claude Code, not a real credential; Codex uses the account from step 3.
`ENABLE_TOOL_SEARCH=false` disables deferred tool discovery, not your
configured MCP tools. Disabling non-streaming fallback helps avoid duplicate
tool calls after an interrupted stream.

Send a normal prompt. Your request should appear in the proxy terminal or
monitor. These `$env:` assignments apply only to this PowerShell window and
processes launched from it. Repeat them in a new window, or save the settings
as described below. Do not use Bash-style `NAME=value command` syntax here.

## 6. Install the optional `/proxy-speed` command

In PowerShell, return to **this fork's repository root** and run:

```powershell
$claudeConfig = if ($env:CLAUDE_CONFIG_DIR) {
    $env:CLAUDE_CONFIG_DIR
} else {
    Join-Path $env:USERPROFILE '.claude'
}
$skillDir = Join-Path $claudeConfig 'skills\proxy-speed'
$skillFile = Join-Path $skillDir 'SKILL.md'
if (Test-Path -LiteralPath $skillFile) {
    Write-Host "Already exists; inspect before replacing: $skillFile"
} else {
    New-Item -ItemType Directory -Path $skillDir -Force | Out-Null
    Copy-Item -LiteralPath '.\integrations\claude-code\skills\proxy-speed\SKILL.md' -Destination $skillFile -ErrorAction Stop
}
```

This installs the command for your user without deliberately replacing an
existing copy. The default destination is
`%USERPROFILE%\.claude\skills\proxy-speed\SKILL.md`; a custom
`CLAUDE_CONFIG_DIR` is respected. Alternatively, copy the file to
`.claude\skills\proxy-speed\SKILL.md` under the project where you use Claude
Code. Install in one scope, not both.

**Restart Claude Code**, then use these commands inside the conversation,
not in PowerShell:

```text
/proxy-speed fast
/proxy-speed ultrafast
/proxy-speed off
```

Wait for a confirmation such as `Proxy speed: fast`.

- The command changes speed for this session's ordinary Codex replies without
  changing the model or reasoning effort. Its subagents inherit the choice.
- Confirmation means the preference was saved; Codex decides whether your
  account and model accept the tier. Acceleration is not guaranteed. If a tier
  is rejected, use `/proxy-speed off` and retry; no new conversation is needed.
- The setting is temporary. Reapply it after 30 minutes idle, a proxy restart,
  or when resuming if needed. Other conversations do not inherit it.
- `off` suppresses acceleration even when enabled by a model suffix or server
  default, but does not permanently erase those defaults.
- The command is not Claude Code's built-in `/fast`, and it does not control
  standalone search or compaction.
- `/proxy-speed fast` requests `fast` literally; the `-fast` model
  suffix requests `priority`. They are not identical settings.

If Claude Code says the command is unavailable, it did not run. Recheck the
installation path, scope, and client version; do not assume an unavailable
slash command changed your proxy settings.

## 7. Save the client settings for future launches

To avoid setting `$env:` variables each time, merge the following `env` block
into `%USERPROFILE%\.claude\settings.json`, or `settings.json` under your
custom `CLAUDE_CONFIG_DIR`. Create the file if absent, but **inspect an
existing file and preserve its other settings**, permissions, and hooks.

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

Restart Claude Code after saving. With the proxy running, future launches can
be just `claude` or `claude --resume` from your project directory. Choose
models your ChatGPT account can access. For larger context and compaction
settings, see [Long conversations](../README.md#long-conversations-compact-and-resume).

These are **Claude Code client settings**. Server options such as
`CCP_CODEX_SERVICE_TIER` must instead be set in the PowerShell window that
starts the proxy. For example, to request priority service by default:

```powershell
$env:CCP_CODEX_SERVICE_TIER = 'priority'
claude-code-proxy serve --no-monitor
```

An explicit `/proxy-speed` preference takes precedence for ordinary replies.
Omit the variable for standard service by default. To remove it from the
current server terminal before restarting, use
`Remove-Item Env:CCP_CODEX_SERVICE_TIER -ErrorAction SilentlyContinue`.

### Optional Codex quota status line

[`codex-status.ps1`](../integrations/claude-code/codex-status.ps1) shows remaining
Codex account quota for the 5-hour and 7-day windows, countdowns, and local reset
times. It uses Windows PowerShell 5.1 and .NET on Windows 10/11, without Bash,
curl, jq, or extra modules. It reads the local proxy's `/usage` endpoint and
does not make Codex inference requests.

From the repository root in PowerShell, install it without replacing an
existing copy:

```powershell
$claudeConfig = if ($env:CLAUDE_CONFIG_DIR) {
    $env:CLAUDE_CONFIG_DIR
} else {
    Join-Path $env:USERPROFILE '.claude'
}
$statusFile = Join-Path $claudeConfig 'codex-status.ps1'
if (Get-Item -LiteralPath $statusFile -Force -ErrorAction SilentlyContinue) {
    Write-Host "Already exists; inspect before replacing: $statusFile"
} else {
    New-Item -ItemType Directory -Path $claudeConfig -Force | Out-Null
    Copy-Item -LiteralPath '.\integrations\claude-code\codex-status.ps1' -Destination $statusFile -ErrorAction Stop
}
$statusFile.Replace('\', '/')
```

The last line prints the installed path with forward slashes. Use that path in
`-File` below, replacing `C:/Users/YOUR_NAME/.claude/codex-status.ps1`. Keep the
single quotes around the path, including when it contains spaces.

Merge this **top-level** field into the profile's `settings.json`, alongside
`env`, preserving other settings and any status-line command you want to keep:

```json
{
  "statusLine": {
    "type": "command",
    "command": "powershell.exe -NoProfile -NonInteractive -File 'C:/Users/YOUR_NAME/.claude/codex-status.ps1'",
    "refreshInterval": 10
  }
}
```

The default settings file is `%USERPROFILE%\.claude\settings.json`; with
`CLAUDE_CONFIG_DIR`, use that directory's `settings.json` and script path.
The quoted forward-slash path works with Git Bash or PowerShell as Claude Code's
command shell. The script respects the existing PowerShell execution policy.
If local scripts are blocked, follow your system's approved script policy or
use WSL; no policy bypass is included.

Check the installed helper in PowerShell:

```powershell
powershell.exe -NoProfile -NonInteractive -File $statusFile
```

For a different proxy address, set `CODEX_USAGE_URL` before launching Claude Code
or add it to the client's existing `env` settings. The default is
`http://127.0.0.1:18765/usage`; requests bypass HTTP proxies and have a one-second
timeout. The ten-second refresh interval reads the last quota snapshot, not
fresh quota from Codex. `quota waiting` means no usable snapshot is available;
`quota unavailable` means `/usage` could not be read. Expired windows show
`awaiting refresh`.

Cargo installation does not update this helper. Compare an installed copy
before replacing it. To disable it, remove the added `statusLine` field or
restore your previous command. Native Windows execution has not been verified
from Linux. For Claude Code inside WSL, use the
[Linux helper](../README.md#codex-quota-in-the-claude-code-status-line) instead.

## 8. Update the fork

Check which installation you are using:

```powershell
Get-Command claude-code-proxy -All | Select-Object Source
```

**Stop the proxy with `Ctrl+C` and close all attached monitor instances
before reinstalling.** Windows may prevent replacing an executable while it
is running. This also clears temporary session speed preferences.

From a clean Git checkout of your chosen fork branch:

```powershell
git pull --ff-only
cargo install --locked --path . --root "$env:USERPROFILE\.cargo" --force
```

Only run the installation after a successful pull. If you downloaded a ZIP,
get an updated archive from this fork, extract it to a separate directory,
and run the Cargo command from that new root; `git pull` does not work in a
ZIP checkout. Keep your old directory if it contains local work.

If your executable lives outside `.cargo\bin`, adjust the installation root
or PATH to match the intended executable. Restart the server after successful
installation. Compare the bundled `/proxy-speed` skill with your installed
copy, inspect before replacing it, and restart Claude Code if you update it.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `cargo` or `claude` is not recognized | Open a new terminal and check user Path. Cargo normally uses `%USERPROFILE%\.cargo\bin`; native Claude Code normally uses `%USERPROFILE%\.local\bin`. |
| `link.exe` or Windows SDK is missing | Install or repair the Visual Studio C++ workload and SDK, then reopen PowerShell. |
| Executable replacement fails or access is denied during an update | Close all running proxy and monitor processes before retrying. Do not start by disabling antivirus or protections. |
| PowerShell blocks a `claude.ps1` launcher | Use the native Claude Code installer and `claude.exe`; do not lower the global execution policy just for this guide. |
| Claude Code cannot find Git Bash | If you want Bash, locate Git with `where.exe git` and set `CLAUDE_CODE_GIT_BASH_PATH` to that installation's `bin\bash.exe`. A typical value is `C:\Program Files\Git\bin\bash.exe`; verify the path exists. Current Claude Code can also use PowerShell without Git Bash. |
| Claude Code cannot connect | Keep the proxy running and check the client URL and port. Use `Test-NetConnection -ComputerName 127.0.0.1 -Port 18765` to check the local listener. |
| Port 18765 is occupied | Inspect the listener with `Get-NetTCPConnection -LocalPort 18765 -State Listen`. Do not kill an unidentified process. If you use `serve --port 18766`, change the client URL to match. |
| Login or model error | Run `claude-code-proxy codex auth status` and `claude-code-proxy models`. Local model support does not guarantee account access. |
| `/proxy-speed` is unavailable | Check the file path, restart Claude Code, and remember that Windows compatibility has not been established by the Linux-only integration test. |
| A speed tier is rejected | Run `/proxy-speed off` inside Claude Code, then retry the task. |
| Disconnects or TLS errors behind a VPN/proxy | Check the network tunnel and its updates; do not assume the model provider caused the error. |

### Prefer WSL?

Install and run **both the proxy and Claude Code inside the same WSL Linux
distribution**, then follow the [main README](../README.md). Use Linux paths
and installations there, not the Windows commands in this guide.
