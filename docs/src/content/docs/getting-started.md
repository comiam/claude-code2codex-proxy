---
title: Getting started
description: Install claude-code-proxy, authenticate with Codex, start the server, and open one working Claude Code session.
---

This path gets one Codex-backed Claude Code session working. This fork supports compatibility only with Codex; other providers are available without guarantees and at your own risk. See [Choosing a provider](providers/choosing-a-provider.md).

## 1. Install

On macOS or Linux, install a stable Rust toolchain and build the fork:

```sh
git clone https://github.com/comiam/claude-code2codex-proxy.git
cd claude-code2codex-proxy
cargo install --locked --path . --root "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
```

For Windows, follow the [PowerShell guide](../../../windows.md).

## 2. Sign in to Codex

Use a **ChatGPT Plus or Pro account**, not an OpenAI API account:

```sh
claude-code-proxy codex auth login
```

For SSH or another headless environment, use `claude-code-proxy codex auth device` instead.

## 3. Start the proxy

Keep this process running:

```sh
claude-code-proxy serve
```

It listens on `127.0.0.1:18765`. In an interactive terminal it opens the monitor TUI. The listener has no client authentication: keep it local or protect access externally, since clients can consume your account quota and send data through your account.

## 4. Start Claude Code

Open another terminal and run:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gpt-6.1-sol[1m] \
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m] \
ENABLE_TOOL_SEARCH=false \
CLAUDE_CODE_AUTO_COMPACT_WINDOW=272000 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
  claude
```

Send a prompt. The request appears in the monitor and the response streams into Claude Code.

The `[1m]` suffix is a Claude Code compaction hint, not a guarantee of a one-million-token context. The proxy removes it before the upstream request. Adjust the example 272,000-token compaction window to the actual model and account limit. Disabling ToolSearch does not disable normal configured MCP tools.

For persistent settings, provider switching, and model discovery, continue to [Configure Claude Code](using/configure-claude-code.md) and [Models and routing](using/models-and-routing.md).
