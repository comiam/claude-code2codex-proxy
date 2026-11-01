---
title: Models and routing
description: Understand provider model patterns, aliases, Codex fast mode, context hints, small models, model listing, and Claude Code gateway discovery.
---

The model ID in each request selects its provider. This fork supports compatibility only with Codex; other providers are available without guarantees and at your own risk.

## Routing patterns

| Pattern | Provider |
| --- | --- |
| Registered `gpt-*` IDs and their `-fast` and `-ultrafast` forms | Codex |
| `kimi-for-coding`, `kimi-k2.6`, `k2.6` | Kimi |
| `grok-composer-2.5-fast`, `grok-4.5`, `grok-4.6` | Grok |
| Non-conflicting registered OpenCode Go IDs and every `opencode-go/<model-id>` | OpenCode Go |
| `cursor`, Cursor legacy aliases, `cursor:<id>`, `cursor-plan:<id>`, `cursor-ask:<id>` | Cursor Agent |
| Anthropic-style aliases such as `haiku`, `sonnet`, `opus`, `fable`, and registered `claude-*` aliases | The `aliasProvider`, Codex by default |

An unknown ID returns HTTP 400 with the supported provider catalog. There is no implicit fallback for arbitrary model names.

## Prefer the live catalog

The model list changes faster than documentation. Ask the installed CLI:

```sh
claude-code-proxy models
claude-code-proxy models --full
```

The compact command abbreviates Cursor's dynamic aliases. `--full` prints every alias discovered through the installed Cursor Agent catalog.

The HTTP equivalent is:

```sh
curl http://127.0.0.1:18765/v1/models
```

## Codex fast mode

Every registered Codex model also has a local `-fast` form. The proxy removes `-fast` from the upstream model and requests the priority service tier.

Models that offer the ultrafast tier also have a local `-ultrafast` form, which requests the ultrafast service tier. Today those are `gpt-6-astra-ultrafast` and `gpt-6.1-sol-ultrafast`. An `-ultrafast` name on any other model is not a registered ID.

Without a `/proxy-speed` session override, Claude Code requests on `/v1/messages` use a configured `codex.serviceTier` or `CCP_CODEX_SERVICE_TIER` override ahead of both suffixes. When the configured tier is `ultrafast` and the model does not offer it, the proxy requests priority instead.

### Per-session `/proxy-speed`

Install the supplied [Claude Code command](configure-claude-code.md#install-the-proxy-speed-command) to select a tier without changing the model or reasoning effort:

```text
/proxy-speed off
/proxy-speed fast
/proxy-speed ultrafast
```

| Command | Ordinary Codex Messages generation |
| --- | --- |
| `off` | Omits `service_tier`, even when a global setting or model suffix requests acceleration. |
| `fast` | Sends the literal `service_tier: "fast"`. |
| `ultrafast` | Sends the literal `service_tier: "ultrafast"`. |

These explicit session choices take precedence over global tier settings and model suffixes. Unlike the `-fast` suffix, the command sends `fast` rather than `priority`. The command does not check the local tier-capability catalog or silently downgrade the requested mode. The API decides whether to accept the value; a rejection is not retried at another tier.

The proxy confirms that the preference was saved locally. This is not confirmation of API support or measured acceleration, and the control request itself does not call Codex. The new value applies to subsequently accepted generation requests; an already-running request and its retries retain their captured value.

The preference belongs to the current session, including agent requests carrying that session ID. Only the main conversation can change it. Other sessions, other providers, and the native Responses and Chat Completions routes are unaffected. Both full Responses and Responses Lite ordinary generations forward the selected tier, including `gpt-6.1-sol`. The command does not control standalone search or compaction.

Preferences are in memory and expire after 30 minutes idle, on capacity eviction, or when the proxy restarts. Historical control messages do not restore them. Without a session choice, global settings and suffixes determine the tier; explicit `off` suppresses them only while its session preference exists. A tier change can require full context on the next request.

This is a custom proxy command, not Claude Code's built-in `/fast` toggle.

## The `[1m]` hint

A trailing `[1m]` affects Claude Code's local compaction policy. The proxy strips it before matching and forwarding the model. Use it only when the upstream model and account can accept the resulting context, and set a safe `CLAUDE_CODE_AUTO_COMPACT_WINDOW` when the real limit is below one million tokens.

## Main and small models

Set both model variables to routable IDs:

```sh
ANTHROPIC_MODEL=gpt-6-sol[1m]
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m]
```

Claude Code sends background work to its small/fast model. Its built-in Haiku ID can route through `aliasProvider`, but a concrete provider ID keeps the behavior explicit.

## Claude Code `/model`

When `ANTHROPIC_BASE_URL` already points to the proxy, these can change the request model:

- `claude --model <id>` at launch
- Claude Code's `/model` command in a session
- `ANTHROPIC_MODEL` for a new process

A model change can switch upstream providers because routing is per request. It does not change `ANTHROPIC_BASE_URL` or move the process back to direct Anthropic.

## Gateway model discovery

Enable Claude Code discovery at process start:

```sh
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gpt-6-sol[1m] \
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m] \
  claude
```

`/v1/models` includes raw provider IDs and Anthropic-style aliases. Claude Code's gateway picker filters for IDs beginning with `claude` or `anthropic`, so configured aliases are the entries most likely to appear. Raw IDs remain usable through `--model`, `/model`, and environment variables.

## Alias routing

`CCP_ALIAS_PROVIDER=kimi` or `"aliasProvider": "kimi"` routes recognized Anthropic-style aliases to Kimi. Accepted values are `codex` and `kimi`. Explicit provider IDs always use their provider.
