---
title: Codex
description: Configure ChatGPT Codex authentication, models, reasoning, tools, images, transports, continuation, compaction, and OpenAI-compatible APIs.
---

Codex uses the ChatGPT subscription Responses endpoint at `https://chatgpt.com/backend-api/codex/responses`.

## Account and authentication

Sign in with a **ChatGPT Plus or Pro account**, not OpenAI API credentials.

```sh
claude-code-proxy codex auth login
# Headless device-code flow
claude-code-proxy codex auth device
claude-code-proxy codex auth status
```

The proxy owns its tokens and does not read native Codex CLI credentials. It refreshes expiring access tokens with a single-flight guard. See [Files and storage](../reference/files-and-storage.md) for credential locations.

## Models and fast mode

Use `claude-code-proxy models` as the current catalog. Model access depends on your ChatGPT account. A model rejected by the subscription produces the upstream error verbatim.

The proxy also reads the Codex CLI's optional `$CODEX_HOME/models_cache.json`
(default `~/.codex/models_cache.json`) to recognize additional models. Listed
entries marked `supported_in_api: true` appear in discovery; hidden entries
with that flag can be requested explicitly. Catalog models use their
`use_responses_lite` lane metadata.

The file is checked at most once per second. Missing, malformed, or oversized
(over 8 MiB) files fall back to built-in models. Incomplete entries are skipped.
The catalog is advisory: its age, CLI version, and account identity are not
validated, and it does not guarantee access with the proxy's separate login.
The proxy does not fetch models or read Codex CLI credentials.

`gpt-6.1-terra` is registered with its Responses Lite lane, alongside `gpt-6.1-sol`. The current Codex catalog advertises priority (Fast) service for Terra, exposed locally as `gpt-6.1-terra-fast`; it does not advertise a Terra `-ultrafast` model variant. Explicit `/proxy-speed` tier values still follow the API-pass-through policy below.

Claude-style aliases map to Codex models: `haiku` and `claude-haiku-*` to `gpt-6-luna`, `sonnet` and `claude-sonnet-*` to `gpt-5.6-terra`, and `opus`, `fable`, `claude-opus-*` (including `claude-opus-5-5`), and `claude-fable-*` to `gpt-6-sol`.

Append `-fast` to any registered Codex model to request `service_tier: "priority"`. For example, `gpt-6-sol-fast` selects `gpt-6-sol` with fast service.

Append `-ultrafast` to a model that offers the ultrafast tier to request `service_tier: "ultrafast"`. `gpt-6-astra` and `gpt-6.1-sol` offer it today, so `gpt-6-astra-ultrafast` and `gpt-6.1-sol-ultrafast` are the ultrafast IDs.

Without an explicit session speed command, `/v1/messages` uses `CCP_CODEX_SERVICE_TIER` or `codex.serviceTier` ahead of either suffix. A configured `ultrafast` applies to models that offer it, and the proxy requests `priority` for the others. The OpenAI-compatible routes use only the suffix or, on `/v1/responses`, the request's own `service_tier`.

### Session speed command

The optional [Claude Code `/proxy-speed` command](../using/configure-claude-code.md#install-the-proxy-speed-command) changes the current session without changing its model or reasoning effort:

- `off` omits the tier for ordinary Codex Messages generation, overriding global tier settings and suffixes.
- `fast` forwards the literal `service_tier: "fast"`, not `priority`.
- `ultrafast` forwards the literal `service_tier: "ultrafast"`.

Both full Responses and Responses Lite ordinary generations, including `gpt-6.1-sol`, send the selected tier values to the API without local capability checks or downgrade. API rejections are returned without retrying at another tier. The command's own response is local: it confirms a saved preference, not supported upstream capacity or observed acceleration.

Preferences are in memory, scoped to the session, inherited by its agent requests, and changeable only by its main conversation. In-flight requests keep their captured value. Idle expiration, capacity eviction, or proxy restart clears the preference; history cannot replay it. The command does not control standalone search, compaction, or native Responses/Chat Completions routes. See [models and routing](../using/models-and-routing.md#per-session-proxy-speed) for full precedence and lifecycle rules.

## Reasoning

Claude Code's `/effort` value maps to Codex `reasoning.effort`: `low`, `medium`, `high`, `xhigh`, or `max`. A proxy override can also force `none`.

When reasoning is enabled, the proxy requests an automatic reasoning summary and translates summary deltas into Claude Code thinking blocks. Codex may omit a summary for a simple prompt. `CCP_CODEX_REASONING_SUMMARY=off` suppresses summaries while preserving effort and encrypted continuation content.

Claude Code summary compaction requests are capped at low effort by default because they perform extraction over a large transcript. `CCP_COMPACT_EFFORT=off` disables the cap, `none` removes reasoning, and another valid effort sets a different maximum. The cap never raises effort.

## Usage reporting

Terminal `/v1/messages` usage includes Codex reasoning counts in
`output_tokens_details.thinking_tokens` and `output_tokens_details.reasoning_tokens`
when available. Both fields report the same count, already included in
`output_tokens`; do not add them to the output total.

When a `codex.rate_limits` event supplies a subscription snapshot, usage also
includes `codex_rate_limits`: optional `plan_type`, `limit_reached`, `primary`,
and `secondary` fields. Windows contain available `used_percent`,
`window_minutes`, and Unix-seconds `reset_at` values. This is a proxy extension,
not a standard Anthropic field. Clients must tolerate unknown usage fields.
The snapshot is event-derived, not synthesized from HTTP quota headers, so it
may be absent depending on the upstream transport. Streaming terminal usage
and non-streaming message usage use the same shape.

## Responses lanes and parallel tools

The gpt-5.6 and gpt-6 families use Responses Lite by default. Lite requires
`parallel_tool_calls: false`, so function tools run serially. Set
`codex.fullLane: true` or `CCP_CODEX_FULL_LANE=1` to use the full Responses lane
for `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `gpt-6-sol`, `gpt-6.1-sol`,
and `gpt-6.1-terra`. This permits parallel calls; the model still decides whether to batch tools, and explicit
client restrictions on parallel tools remain effective.

This is not a universal lane override: catalog-only models keep the lane
specified in the local Codex model cache. Luna models stay on Lite.
Automatic hosted web search uses the full lane
regardless of this setting and remaps Luna to its Sol sibling. Forced standalone
search uses `/alpha/search` instead. `/v1/responses` honors the setting;
text-only `/v1/chat/completions` keeps its default lane. Full-lane availability
and quota accounting depend on the upstream subscription; this setting is opt-in.

## Tools and multimodal input

- Claude function tools and tool results map to Responses API function calls and outputs.
- Claude Code's forced `web_search_20250305` uses Codex's standalone search endpoint with the selected model and supported domain filters. It returns a buffered result with answer text, source links and search usage. Automatic hosted search uses Responses and can stream answer text and source links.
- The built-in client `ToolSearch` is excluded. Supplied MCP schemas are registered eagerly as functions; discovery and deferred schema loading are unsupported. For future Claude Code launches, set `ENABLE_TOOL_SEARCH=false` in that client process. Missing schemas are not recovered from tool names. Sending all available schemas uses more context than deferred loading.
- Legacy `tool_reference` result children and unsupported typed children are skipped with bounded diagnostics; supported neighboring text/images and call/result IDs are preserved. Explicit unsupported Anthropic server-tool declarations return a local `400 invalid_request_error`.
- Completed hosted searches produce use/result cards. Search output does not support replay to Anthropic; available query, title, URL and text become translated context for the next OpenAI turn.
- Top-level base64 user images map to `input_image`.
- Supported base64 images nested in tool results also map to `input_image`.
- Remote image URLs, malformed images, and unsupported tool-result image forms remain textual placeholders.
- Strict JSON schema output maps to Responses `text.format`.

## Transport and continuation

`auto` is the default transport: it uses WebSocket with HTTP SSE fallback only when setup fails before a request is sent. Set `CCP_CODEX_TRANSPORT=websocket` for WebSocket-only behavior, or `http` for HTTP SSE. A failure after a request is sent does not trigger HTTP replay.

WebSocket reuse and append-only continuation are enabled by default for a valid session or Agent identity. An explicit configuration is:

```json
{
  "codex": {
    "transport": "websocket",
    "websocketReuse": true,
    "previousResponseId": true
  }
}
```

Set `CCP_CODEX_PREVIOUS_RESPONSE_ID=0` (or `previousResponseId:false`) to opt out of delta continuation while retaining socket reuse. Compatible WebSocket reuse is enabled by default, independently of `previousResponseId`. Explicit `websocketReuse=false` disables both pooling and delta continuation. Reuse needs the same valid session/Agent identity and compatible endpoint, model and authentication. Prompt changes can reuse a healthy socket while carrying full input. Automatic Responses hosted-search output without canonical native replay also starts the next turn with full translated context and no `previous_response_id`, while retaining healthy compatible socket reuse.

The default pool retains at most 64 idle sockets, expiring them after five minutes idle or thirty minutes absolute age. It processes connection and quota events between turns. Connection starts are spaced one second apart; active generations use independent leases. Reuse does not change remote model inference speed.

On the HTTP transport, the proxy waits five minutes for the response headers before failing the request. Codex withholds the response head until the model produces its first output, so a large request to a high-effort model can hold it for minutes. `CCP_CODEX_HEADER_TIMEOUT_MS` or `codex.headerTimeoutMs` changes that bound.

WebSocket setup honors `HTTP_PROXY` for `ws://`, `HTTPS_PROXY` for the default `wss://` endpoint, `ALL_PROXY` as a fallback, and `NO_PROXY` exclusions. A normal HTTP proxy can therefore carry the default WebSocket connection with CONNECT; TUN mode is not required. Set proxy variables before starting the process and restart after changing them. For example, setting `HTTPS_PROXY` to `http://127.0.0.1:7890` sends HTTPS/WSS destinations through the HTTP proxy at port 7890; it does not require an `https://` proxy URL.

With default continuation, the first request carries full input; compatible following requests carry `previous_response_id` and only the new input items. Prompt caching reuses provider computation; continuation additionally reduces uploaded input. A valid identity containing only a Claude Code session ID owns the Main continuation for that session. Each valid direct Agent ID owns an independent continuation and reusable WebSocket within the same session. Nested Agents are keyed by their direct child ID; the parent ID is validated but does not become part of the owner key. The proxy sends `previous_response_id` only when the translated request shape and transcript extension are safe, and only on the exact live WebSocket that produced that response.

An absent, malformed, or ambiguous identity does not reject the HTTP request; that request proceeds without continuation or WebSocket reuse. If the originating socket is missing, dead, or has been replaced, the proxy selects full translated input before writing to a replacement connection. A server-side missing-parent error permits one full-context recovery before semantic output; interrupted output is an error, never an invented successful completion. Continuation and connection state is held only in memory and is lost when the proxy restarts.

Detected auto-review classifier subrequests are intentionally stateless even when valid session and Agent headers are present. They neither consume nor publish continuation or WebSocket ownership.

## Stream completion and usage

HTTP and WebSocket preserve upstream event order. Read tool arguments are buffered up to 5,000,000 bytes and validated at item completion. An incomplete or malformed final Read returns an error; other active tools remain independent.

For streaming Messages over HTTP, answer text streams immediately, but successful completion waits for a validated response-body EOF. A malformed, truncated, or post-terminal JSON tail produces an error instead of success.

After semantic output is committed, failures or EOF produce a typed SSE error without a successful `message_stop`, a reusable response parent, or transparent regeneration. Ambiguous WebSocket writes return `x-should-retry:false`. Cancelled consumers and continuously full downstream queues release their exact attempt; the queue stall budget remains 300 seconds.

A truly empty reply after all tool calls finish successfully gets one full-context retry. A second empty reply may end the turn only with an explicit `completed` status and no new user instructions. This exception applies before visible output; thinking without an answer and a tool call interrupted before the terminal event remain errors.

Live input counts are local estimates. Provider totals replace estimates when reported, and only provider observations enter authoritative session totals and rates. Output totals can include reasoning and tools; the monitor labels them `provider-total`. Missing reasoning breakdown remains unknown. Response-ready, first body yield, EOF and cancellation are distinct local milestones, not network acknowledgement or UI rendering.

## Conversation compaction

Claude Code's manual `/compact` and automatic compaction use one Codex text generation. The proxy preserves the client's summary instructions and focus, sends conversation history without executable tool registrations, and streams the resulting portable text summary through the normal Anthropic API. Model, service tier and the compact effort cap still apply.

Claude Code owns the summary and compact boundary. Its later turns contain the summary and new messages instead of the original transcript. A new CLI process using `--resume` can carry that summary through the proxy; no proxy compaction registry or opaque checkpoint is required. A healthy Codex WebSocket remains reusable after compaction. The first turn with the shortened history starts a fresh Responses input without the old response parent; later append-only turns can use normal continuation.

For streaming compact requests, the proxy sends an early SSE ping after validation and routing, then heartbeats every 15 seconds while waiting. Text deltas pass through as they arrive. Claude Code may continue to display `Compacting…` until the summary finishes, so these transport events do not promise character-by-character UI updates. A successful `message_stop` requires a complete upstream terminal event. An upstream error, incomplete response or EOF after partial text ends with an SSE error after headers, without a successful compact or transparent replay of partial output.

`codex.serverCompaction` and `CCP_CODEX_SERVER_COMPACTION` are deprecated and ignored for Claude Messages; enabling either produces a startup warning.

## OpenAI-compatible APIs

`CCP_CODEX_RESPONSES_API=1` enables both `POST /v1/responses` and `POST /v1/chat/completions`. The setting is under Codex configuration, but the routes also accept Kimi, Grok, OpenCode Go, and Cursor models.

The Responses route preserves native JSON or SSE response bodies for registered Codex models. The Chat Completions route translates standard text messages, reasoning effort, JSON object or JSON Schema output, and buffered or streaming responses. Its omitted reasoning effort defaults to `medium`; the proxy-wide Codex effort override still takes precedence.

The proxy replaces incoming credentials with stored Codex auth for both routes. Response retrieval or deletion, function calling through Chat Completions, and WebSocket ingress are outside their scope. See [HTTP API](../reference/http-api.md) for supported Chat Completions fields and error behavior.

## Images API

`CCP_CODEX_IMAGES_API=1` separately enables `POST /v1/images/generations` and `POST /v1/images/edits`. The routes reuse the proxy's stored ChatGPT OAuth session and target the ChatGPT Codex image backend; no OpenAI Platform API key is required.

```sh
CCP_CODEX_IMAGES_API=1 claude-code-proxy serve
```

The model defaults to and is restricted to `gpt-image-2`. Generation accepts JSON. Editing accepts either Codex JSON data URLs or OpenAI-style multipart uploads, which the proxy validates and converts into the Codex JSON contract. Results are returned as `data[].b64_json`. Masks, remote URLs, URL-formatted output, and image variations are not supported.

This is an internal ChatGPT Codex interface rather than the public Platform Images API. It consumes the signed-in account's image quota and can change without public API compatibility guarantees. Image prompts, uploads, generated base64, and upstream error bodies are excluded from traffic captures and persistent error diagnostics.

Because callers are not authenticated, binding to a LAN address lets every firewall-admitted host consume the signed-in account's quota. Restrict the listener to a trusted interface/subnet and never expose it through router forwarding, UPnP, a public tunnel, or permissive IPv6 rules.

See [Configuration](../reference/configuration.md) for every Codex setting and [Troubleshooting](../using/troubleshooting.md) for auth, model, and transport failures.

The retained continuation budget is 8 MiB per owner and 128 MiB total, including the serialized transcript, prompt signature and response ID. Eviction selects full input without truncating history. These limits do not bound process RSS. HTTP requests always carry full input.

`websocket_request_sent` records request mode, input counts, wire bytes, continuation reason and socket reuse. `Cached` and input usage are provider counters and can include ancestor context even for a small delta. First-response and idle-failure logs identify event/frame kinds without logging each output delta. Diagnostics use a bounded, nonblocking writer with a 1 GB retention budget.

`codex_event_skipped` records unsupported output events/items and input tool-result children without their payloads. Each input-conversion collector or upstream attempt logs at most 16 distinct event types and one count summary; a retry under the same `reqId` has its own bound. Supported neighboring content remains available. A skip does not turn an unfinished supported call or an upstream failure into success.

### Early reasoning status

For streaming Messages requests, a real Codex reasoning item immediately opens
Claude Code's thinking block, before the first summary. Sending a request or a
heartbeat does not count as reasoning. Once the thinking start is published,
the proxy stops transparent retries for that response and forwards errors or
interruptions through SSE. Automatic retries may occur before that point.
This retry boundary applies to streaming Messages, not native Responses or Chat Completions.

Claude Code can independently issue another Messages request after an SSE error.
That request has a new proxy `reqId`; the proxy's retry boundary does not disable
client-side retry policies. A thinking indicator signals reasoning activity,
without guaranteeing that the response will succeed.
