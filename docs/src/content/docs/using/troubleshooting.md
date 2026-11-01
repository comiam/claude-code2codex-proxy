---
title: Troubleshooting
description: Diagnose listener, authentication, model routing, streaming, context, Cursor bundle, rate-limit, logging, and traffic-capture problems.
---

## Proxy is unreachable

Check the process and liveness route:

```sh
curl http://127.0.0.1:18765/healthz
```

A healthy listener returns `{"ok":true}`. Confirm `ANTHROPIC_BASE_URL` uses the same address and port. `claude-code-proxy serve --port 11435` and `PORT=11435` change the listener port. `--port` wins for that command.

## Authentication error

Check the provider selected by the model, then inspect its credential:

```sh
claude-code-proxy codex auth status
claude-code-proxy kimi auth status
claude-code-proxy grok auth status
claude-code-proxy cursor auth status
```

Use that provider's login command when credentials are missing or expired. Codex requires ChatGPT subscription auth, not an OpenAI API key. Each provider uses proxy-owned credentials.
OpenCode Go has no `auth status` command; configure its API key through
`CCP_OPENCODE_API_KEY`, `OPENCODE_API_KEY`, or `opencode.apiKey` in
`config.json`.

## Model returns HTTP 400

Run:

```sh
claude-code-proxy models
claude-code-proxy models --full
```

An unknown local ID returns a catalog in the error. A known ID can still be rejected upstream when the account, subscription, or region lacks access. Cursor's prefixed form, such as `cursor:gpt-5.5`, forces Cursor routing and avoids cross-provider collisions.

## Background requests fail

Set `ANTHROPIC_SMALL_FAST_MODEL` to a concrete routable ID. Claude Code sends title and small background tasks through that model independently of the main model.

## A tool runs twice

Set `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1` for Claude Code. Retrying a partially completed stream as non-streaming can duplicate tool calls.

## Codex tool discovery or search content is missing

For future Claude Code launches, set `ENABLE_TOOL_SEARCH=false` in the client process. The Codex adapter excludes the built-in `ToolSearch` and registers supplied full MCP schemas as ordinary functions. It cannot recover missing schemas from a tool name; eager schemas consume more context than deferred loading.

Automatic hosted search streams real answer text and available source links. Each completed search also supplies a matching use/result card so Claude Code reports the actual search count. Cards are finalized together at successful completion; ready answer text is not buffered for them. Forced subrequests use `/alpha/search` and the same completed-card projection, but wait for a complete DTO without upstream text deltas. Unavailable Anthropic encrypted fields are omitted; replay to Anthropic remains unsupported. A following OpenAI turn uses full translated context without an old response parent; the healthy compatible WebSocket can still be reused.

Correlate `codex_event_skipped` by `reqId` for unsupported output events, old `tool_reference` children and other unsupported typed results. Supported neighboring text/images and call/result IDs are preserved. Diagnostics omit payloads and cap distinct types at 16 with one summary per input-conversion collector or upstream attempt, so a retry can produce another bounded set. Unsupported server-tool declarations fail locally with `400 invalid_request_error`.

After partial text or tool arguments, an upstream failure or EOF is a typed SSE error without a successful `message_stop` or new response parent. The proxy does not regenerate committed output. Response-ready or queued-byte timing does not establish delivery to Claude Code's UI.

## Session reaches context limits

`[1m]` changes Claude Code's local compaction threshold and does not enlarge upstream context. Use a safe `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, remove `[1m]`, or compact earlier. For ChatGPT GPT-5.6 subscription models, `272000` is the documented setup value.

If `/compact` stays at `Compacting…`, distinguish client display from transport progress: Codex compact sends an early SSE ping and then heartbeats every 15 seconds; text deltas stream when available, but Claude Code may wait for the complete summary before updating its UI. Correlate `reqId` and `requestKind: compact` in `codex_upstream_request_started`. Compaction uses one text generation; the deprecated `codex.serverCompaction` / `CCP_CODEX_SERVER_COMPACTION` flag adds no extra pass.

An upstream failure after SSE headers appears as a stream error even though the HTTP status is already 200. Partial summary text does not establish a successful compact boundary and is not transparently generated again. After successful compaction Claude Code saves the summary itself, including for a new process using `--resume`; restarting the proxy does not require restoring a compaction registry. Healthy WebSockets remain reusable, while the shortened history begins without an old response parent.

## Codex WebSocket fails

For a non-TUN local HTTP proxy, set both destination-scheme variables before starting the proxy:

| Variable | Value |
| --- | --- |
| `HTTP_PROXY` | `http://127.0.0.1:7890` |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` |

Set them through the operating system, service manager, or shell, then start `claude-code-proxy serve` in the same environment.

The default `wss://chatgpt.com` connection uses `HTTPS_PROXY`; a working proxy should show `CONNECT chatgpt.com:443`. The `http://` value is normal: it describes how to reach the proxy, while `HTTPS_PROXY` describes which destinations use it. Restart claude-code-proxy after changing these variables because the client and pooled WebSocket route are created at startup.

Check `NO_PROXY` when the proxy sees no request. Proxy connection, authentication, or CONNECT failure is returned as an error and never retried directly. Environment variables are supported; OS proxy settings and PAC/WPAD discovery are not automatic.

Use HTTP SSE to isolate transport behavior:

```sh
CCP_CODEX_TRANSPORT=http claude-code-proxy serve
```

`auto` falls back only when WebSocket setup fails before sending the request. It does not replay an in-flight request.

## Stream waits or cancels after several minutes

Streaming message requests that spend more than 15 seconds preparing or waiting for a provider receive SSE headers and `ping` events while that same provider request continues. A ping confirms that the HTTP connection is alive; it does not indicate model output or token usage. Provider failures after those headers appear as SSE `error` events, while failures before them retain their HTTP status and JSON body.

Correlate `proxy.log` events by `reqId`: `provider_pending_heartbeat_started` identifies the wait, `pending_provider_response_ready` records the eventual provider status and headers, and `downstream_request_cancelled` records the elapsed time and whether cancellation happened in the request handler or response body. The monitor separates response readiness and body delivery from generation start. A provider can still stall despite a healthy downstream connection; inspect its transport events and `request_failed` capture for the cause.

## Measure Codex time to first content

For WebSocket requests, `websocket_phase` logs the first `response_created`,
`reasoning_started`, nonempty `first_summary_delta`, and nonempty
`first_text_delta` observed by the upstream reader. It logs each phase at most
once per transport attempt, without prompt, summary, text, or item IDs.
`phaseTimesUs` on terminal/error records retains the observed timestamps; `null`
means that phase was not observed, including responses without a summary.

Join records by `reqId`, `processId`, and `attemptId`. Socket IDs alone can span
many requests. `attemptId` is a process-wide transport-attempt identifier, not a
request's retry count. `monotonicUs` is microseconds on one process-local monotonic clock,
shared by the reader and body polling; it cannot be compared across processes.
`attemptStartedUs` anchors the transport attempt. The first response event is
usually `response.created`, not a text token or proof that reasoning has started.

`codex_thinking_start_yielded` marks the first actual thinking-start frame yielded
by the Messages provider's HTTP body. For early Codex reasoning, subtract the
matching `reasoning_started.monotonicUs` from the body-yield timestamp to measure
local queues, staging and body polling. Measure the upstream wait for a summary
separately as `first_summary_delta - reasoning_started`. Body yield can precede
the summary: `body_yield - first_summary_delta` can be negative and is
not a local delivery delay. Body yield is not network acknowledgement or proof
of UI rendering. HTTP transport currently supplies no reader phase/attempt ID; its
body-yield record has `attemptId:null` and must not be joined to a WebSocket
attempt. Diagnostic queues remain best effort, so missing records alone do not
prove an event never occurred. The instrumentation itself remains observational;
the early reasoning policy described below controls block opening and retries.

### Distinguish readiness, upstream outcome, and request outcome

Read the `phase` and `outcome` fields before treating a name containing
`completed` as success:

| Record | `phase` | `outcome` and meaning |
| --- | --- | --- |
| `request_completed`, `codex_upstream_response_ready` | `response_ready` | `pending` for HTTP 2xx: the response is ready, but its body has not been fully delivered or validated. `failed` for a non-2xx response. |
| `codex_thinking_start_yielded` | `thinking_started` | `pending`: the provider yielded a thinking-start frame; this is neither a text token nor a successful terminal. |
| `codex_http_stream_completed`, `websocket_terminal` | `upstream_terminal` | `completed`, `failed`, `incomplete`, or `cancelled`: the upstream terminal's outcome, before translation and downstream delivery finish. |
| `codex_http_stream_failed`, `codex_http_stream_dropped` | `upstream_stream` | `failed` for a read/protocol failure, `cancelled` when the consumer drops the stream. |
| `codex_live_stream_failed` | Failure location, such as `translation` or `upstream` | `failed`: the Messages stream failed, including after HTTP 200 or an upstream `completed`. |

Upstream outcomes belong to a transport attempt. A retry before output commits
can succeed under the same `reqId`; the monitor records the final request
outcome. Terminal outcome classification includes nested error envelopes.

For example, one `reqId` can have `request_completed` with HTTP 200 and
`outcome:pending`, followed by `codex_http_stream_completed` with
`outcome:failed`. An empty upstream `completed` can instead log
`outcome:completed` at `upstream_terminal`, then fail translation. The monitor's
final **Failed** status records the logical failure even when the HTTP status
remains 200.

`request_completed.ms` has `timingPhase:response-ready`: it
measures handler time until the response is ready, not stream duration or time
to first text. `codex_upstream_response_ready.ms` measures the provider's wait
until its live response is ready. HTTP stream-end `ms` measures elapsed time
from that HTTP attempt's start, including the wait for headers;
`websocket_terminal.elapsedMs` measures its transport attempt. These durations
have different start and end boundaries. Use the matching monotonic phase
timestamps above to diagnose local thinking delay, and the monitor's
`response_ready_at` and `body_finished_at` timestamps to measure the downstream
body's lifetime. Request-start to body-finished also includes preparation.
Keep correlation by
`reqId` / `processId` / `attemptId`; no new IDs are assigned by these labels.

## Cursor bundle cannot be found

Cursor needs the installed Cursor Agent JavaScript bundle for protobuf schemas. Confirm Cursor Agent is installed, then set:

```sh
CCP_CURSOR_AGENT_BUNDLE=/path/to/cursor-agent/index.js \
  claude-code-proxy serve
```

## Rate limited

Upstream limits are shared with other clients on the same account. Codex limit responses and Kimi HTTP 429 responses surface as HTTP 429 with `retry-after`. Wait for the indicated interval or reduce concurrent traffic.

## Find the complete error

The monitor shows request detail. `proxy.log` contains structured JSONL events, and `errors/` stores complete redacted failed-response payloads. A `request_failed` event includes the error file path.

The monitor's **In** is the provider's gross input count, including **Cached**. Cached comes from Codex's reported cached-input breakdown; `0` is a reported zero and `—` means the breakdown was not supplied. Session In, Cached, and Out totals cover the same session lifetime while the proxy retains its accounting; Cached sums only supplied counts, so an unknown breakdown contributes no invented zero or cache percentage. Requests counts active requests plus the retained recent window (up to 200), so token totals can outlive rows in that window.

Connection and error metadata is logged without enabling traffic capture: correlate `reqId`, socket ID, reuse, first response, timeout phase, close code/reason and provider attempts. The background writer limits managed logs, rotations, failed-response files and generated traffic captures to **1,000,000,000 bytes total (1 GB)**. It removes the oldest diagnostic files to make room; oversized records or unavailable space are dropped and counted in `/monitor`. Capture records removed by retention make that capture incomplete. The limit counts file contents, not filesystem metadata, empty directories or unrelated files.

For a source checkout, run the isolated helper:

```sh
./scripts/debug-proxy
```

It prints a random local URL, a sourceable `client.env`, and artifact paths. Keep the terminal open, reproduce once, then press Ctrl-C. Captures contain sensitive prompt and tool content.

## Focused traffic capture

```sh
CCP_LOG_VERBOSE=1 \
CCP_TRAFFIC_LOG=1 \
  claude-code-proxy serve --no-monitor
```

Traffic capture attempts to record the inbound request, translated upstream request, upstream headers and events, and downstream events in emission order. Ordinary logs and capture share a nonblocking writer with an 8 MiB / 2,048-record queue and a 2 MiB individual-record allowance. One in-flight record is outside the queue allowance. A slow disk or stderr sink does not make API output wait; saturated or failing diagnostics can lose records.

`capture-completion.json` reports diagnostic completeness, not successful generation or delivery. Missing marker, `complete:false`, dropped records or truncated content mean an incomplete capture. The marker itself is best effort. Live Codex captures omit the complete raw SSE mirror. `/monitor` exposes accepted, dropped, failed and written records, queue bytes/records and in-flight bytes even when the sink fails. Shutdown waits at most two seconds for accepted diagnostics; an OS write can remain blocked until the process exits.

Known credentials and encrypted reasoning are redacted, but prompts and tool content are preserved. Disable capture and delete artifacts after diagnosis.

## Local counting overload

A Codex `/v1/messages/count_tokens` HTTP 503 with `overloaded_error` identifies local token-counting CPU capacity: two jobs may execute and four may wait. Reduce concurrent count requests and retry later. This is separate from upstream HTTP 429 quota exhaustion. Live streams use a cheap estimate while authoritative provider usage remains available at completion.

## Reuse and rollback

Independent WebSocket reuse is enabled by default for a stable session/Agent identity. Explicit `websocketReuse:false` disables it. A healthy reused socket still receives full input when continuation is disabled or evicted. Model/authentication changes require a new connection and full input. Malformed or interrupted responses never establish a new parent.

To disable pooling and continuation, set `codex.websocketReuse` to `false` (or `CCP_CODEX_WEBSOCKET_REUSE=0`) and restart the process. Pool and byte budgets are startup settings. Preserve the previous binary and configuration for rollback; restoring them requires the normal operator-controlled restart. The default continuation budgets are 8 MiB per owner and 128 MiB total; configured cache bytes do not bound process RSS.

Continuation is enabled by default for valid WebSocket owners. In `websocket_request_sent`, inspect `requestMode` and `continuationReason` before blaming upload size: prompt changes, a changed transcript, eviction or lost origin require a full request. `wireBytes` measures that request, while `In`/`Cached` measure provider usage. Idle Ping/quota events are processed between turns; an idle Close invalidates the connection. A tool item without a response terminal does not authorize success or another parent. The request-completed log marks response readiness; body-finished timing measures the streaming lifetime. No downstream yield implies acknowledged UI delivery.

### Codex shows thinking, then an error

Thinking means Codex emitted a real reasoning item. For streaming Messages,
the proxy stops retrying once it publishes that block and forwards an upstream
error or interruption through SSE. Automatic retries may occur before reasoning
starts.

Claude Code may retry by sending new Messages requests. Check `reqId`: different
IDs identify separate incoming requests; repeated upstream attempts under one
ID identify proxy retries. The thinking indicator does not promise a successful
answer. Claude Code's partial JSON output is processed client output and can
include client-generated `message_stop` events during error handling; use captured
downstream SSE to inspect the proxy's actual terminal events.
