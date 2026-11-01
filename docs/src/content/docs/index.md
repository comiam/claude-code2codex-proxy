# claude-code2codex-proxy

[claude-code2codex-proxy](https://github.com/comiam/claude-code2codex-proxy) lets Claude Code use a ChatGPT Codex subscription through a local proxy. The executable is named `claude-code-proxy`.

This fork supports compatibility only with Codex. Kimi, Grok, OpenCode Go and Cursor Agent remain available without guarantees and at your own risk.

Claude Code retains its tools, skills, hooks, subagents and editor integrations. The proxy translates requests and provides a monitor for sessions, errors and usage. The local listener is unauthenticated: protect access to your account quota and data.

![Claude Code running through claude-code-proxy](../../../public/claude-code-screenshot.webp)

Start with [Codex setup](getting-started.md), review [provider choices](providers/choosing-a-provider.md), or read [how requests flow](how-it-works.md).
