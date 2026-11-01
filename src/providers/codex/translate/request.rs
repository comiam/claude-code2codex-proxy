use std::collections::HashSet;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::anthropic::schema::MessagesRequest;
use crate::config;
use crate::providers::translate_shared::{
    ContentBlock, flatten_system_text, image_source_to_url, normalize_content, parallel_tool_calls,
    read_effort,
};

use super::event::{SkipDiagnostics, SkippedEvent};
use super::model_allowlist::service_tier_for_model;
use super::read_rewrite::{ReadOffsetRewrite, read_offset_rewrite};
use super::reasoning_signature::decode_reasoning_signature;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    None,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Effort::None => write!(f, "none"),
            Effort::Low => write!(f, "low"),
            Effort::Medium => write!(f, "medium"),
            Effort::High => write!(f, "high"),
            Effort::Xhigh => write!(f, "xhigh"),
            Effort::Max => write!(f, "max"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Fast,
    Priority,
    Ultrafast,
    Flex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(ResponsesToolChoiceMode),
    Function {
        r#type: String,
        name: String,
    },
    WebSearch {
        r#type: String,
    },
    AllowedTools {
        r#type: String,
        mode: String,
        tools: Vec<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<ResponsesInputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    pub store: bool,
    pub stream: bool,
    pub parallel_tool_calls: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<std::collections::HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    pub text: ResponsesText,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
}

/// Borrowed projection shared by continuation signatures and wire shaping.
/// The exhaustive destructure below forces new request fields to be considered.
#[derive(Serialize)]
pub(crate) struct NonInputRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ResponsesTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ResponsesToolChoice>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_metadata: Option<&'a std::collections::HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    text: &'a ResponsesText,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a ResponsesReasoning>,
}

#[derive(Serialize)]
pub(crate) struct WireRequest<'a> {
    #[serde(flatten)]
    parameters: NonInputRequest<'a>,
    input: &'a [ResponsesInputItem],
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<&'a str>,
}

impl ResponsesRequest {
    pub(crate) fn non_input(&self) -> NonInputRequest<'_> {
        let Self {
            model,
            instructions,
            input: _,
            tools,
            tool_choice,
            store,
            stream,
            parallel_tool_calls,
            include,
            client_metadata,
            service_tier,
            prompt_cache_key,
            text,
            reasoning,
        } = self;
        NonInputRequest {
            model,
            instructions: instructions.as_deref(),
            tools: tools.as_deref(),
            tool_choice: tool_choice.as_ref(),
            store: *store,
            stream: Some(*stream),
            parallel_tool_calls: *parallel_tool_calls,
            include: include.as_deref(),
            client_metadata: client_metadata.as_ref(),
            service_tier: service_tier.as_ref(),
            prompt_cache_key: prompt_cache_key.as_deref(),
            text,
            reasoning: reasoning.as_ref(),
        }
    }

    pub(crate) fn wire<'a>(
        &'a self,
        input: &'a [ResponsesInputItem],
        http: bool,
        previous_response_id: Option<&'a str>,
    ) -> WireRequest<'a> {
        let mut parameters = self.non_input();
        if !http {
            parameters.stream = None;
        }
        WireRequest {
            parameters,
            input,
            kind: (!http).then_some("response.create"),
            previous_response_id,
        }
    }
}

pub(crate) use crate::bounded_json_size;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesText {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponsesTextFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesTextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponsesInputItem {
    #[serde(rename = "additional_tools")]
    AdditionalTools {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: String,
        tools: Vec<Value>,
    },
    #[serde(rename = "message")]
    Message {
        role: String,
        content: Vec<ResponsesContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)]
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[serde(default)]
        call_id: String,
        output: ResponsesFunctionCallOutput,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<Value>,
        encrypted_content: String,
    },
    #[serde(rename = "compaction")]
    Compaction { encrypted_content: String },
    #[serde(rename = "compaction_trigger")]
    CompactionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ResponsesFunctionCallOutput {
    Text(String),
    ContentItems(Vec<ResponsesFunctionCallOutputContentPart>),
}

impl ResponsesFunctionCallOutput {
    #[cfg(test)]
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::ContentItems(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesFunctionCallOutputContentPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesTool {
    Function(ResponsesFunctionTool),
    WebSearch(ResponsesWebSearchTool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesFunctionTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSearchTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub external_web_access: bool,
    pub search_content_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<ResponsesWebSearchFilters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSearchFilters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

pub struct TranslateOptions {
    pub session_id: Option<String>,
    pub service_tier: Option<ServiceTier>,
    pub model: String,
    pub use_responses_lite: bool,
}

// ---------------------------------------------------------------------------
// Translation entry point
// ---------------------------------------------------------------------------

pub(crate) fn to_codex_effort(effort: Option<&str>) -> Option<Effort> {
    match effort {
        Some("max") => Some(Effort::Max),
        Some("xhigh") => Some(Effort::Xhigh),
        Some("low") => Some(Effort::Low),
        Some("medium") => Some(Effort::Medium),
        Some("high") => Some(Effort::High),
        _ => None,
    }
}

fn resolve_effort(effort: Option<Effort>) -> Result<Option<Effort>, anyhow::Error> {
    resolve_effort_override(effort, config::codex_effort().as_deref())
}

pub(crate) fn resolve_effort_override(
    effort: Option<Effort>,
    override_effort: Option<&str>,
) -> Result<Option<Effort>, anyhow::Error> {
    if let Some(val) = override_effort {
        let valid = ["none", "low", "medium", "high", "xhigh", "max"];
        if !valid.contains(&val) {
            anyhow::bail!(
                "Invalid effort override: \"{val}\". Must be one of: none, low, medium, high, xhigh, max"
            );
        }
        return Ok(Some(match val {
            "max" => Effort::Max,
            "xhigh" => Effort::Xhigh,
            "high" => Effort::High,
            "medium" => Effort::Medium,
            "low" => Effort::Low,
            _ => Effort::None,
        }));
    }
    Ok(effort)
}

fn reasoning_summary_requested(summary: Option<&str>) -> bool {
    !matches!(summary, Some("off" | "none"))
}

/// Whether the resolved effort asks the upstream for reasoning output.
/// `Effort::None` still names the effort so the wire request overrides the
/// upstream default, but it must not request a summary or encrypted
/// continuation content, which are reasoning artifacts.
fn reasoning_requested(effort: Option<&Effort>) -> bool {
    effort.is_some_and(|effort| *effort != Effort::None)
}

// ---------------------------------------------------------------------------
// Compaction fast path
// ---------------------------------------------------------------------------

const COMPACT_SYSTEM_MARKER: &str =
    "You are a helpful AI assistant tasked with summarizing conversations";
const COMPACT_MESSAGE_PREFIX: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.";
const COMPACT_MESSAGE_TASK: &str =
    "Your task is to create a detailed summary of the conversation so far";

pub(crate) fn is_compact_request(instructions: Option<&str>) -> bool {
    instructions.is_some_and(|text| text.trim_start().starts_with(COMPACT_SYSTEM_MARKER))
}

pub(crate) fn is_compact_message_text(text: &str) -> bool {
    text.trim_start().starts_with(COMPACT_MESSAGE_PREFIX) && text.contains(COMPACT_MESSAGE_TASK)
}

fn is_compact_message_content(content: &Value) -> bool {
    match content {
        Value::String(text) => is_compact_message_text(text),
        // The first text instruction sets the mode; a later block may quote a prompt.
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .find(|text| !text.trim().is_empty())
            .is_some_and(is_compact_message_text),
        _ => false,
    }
}

pub(crate) fn is_compact_messages_request(request: &MessagesRequest) -> bool {
    is_compact_request(flatten_system_text(request.extra.get("system")).as_deref())
        || request.messages.last().is_some_and(|message| {
            message.role == "user" && is_compact_message_content(&message.content)
        })
}

/// Compaction effort cap, or None when disabled. Missing effort takes the cap
/// instead of the upstream default; explicit effort below the cap is preserved.
fn compact_effort_cap() -> Option<Effort> {
    compact_effort_cap_from(config::codex_compact_effort().as_deref())
}

fn apply_compact_effort_cap(resolved: Option<Effort>, cap: Option<Effort>) -> Option<Effort> {
    match (resolved, cap) {
        (resolved, None) => resolved,
        (Some(effort), Some(cap)) if effort <= cap => Some(effort),
        (_, Some(cap)) => Some(cap),
    }
}

fn compact_effort_cap_from(raw: Option<&str>) -> Option<Effort> {
    match raw {
        None | Some("") => Some(Effort::Low),
        Some("off") => None,
        Some("none") => Some(Effort::None),
        Some(other) => to_codex_effort(Some(other)).or(Some(Effort::Low)),
    }
}

const VALID_SERVICE_TIERS: &[&str] = &["fast", "priority", "ultrafast", "flex"];

fn normalize_service_tier(tier: &str) -> Result<ServiceTier, anyhow::Error> {
    match tier {
        "fast" | "priority" => Ok(ServiceTier::Priority),
        "ultrafast" => Ok(ServiceTier::Ultrafast),
        "flex" => Ok(ServiceTier::Flex),
        _ => anyhow::bail!(
            "Invalid service tier override: \"{tier}\". Must be one of: {}",
            VALID_SERVICE_TIERS.join(", ")
        ),
    }
}

fn resolve_service_tier(
    model_tier: Option<ServiceTier>,
    model: &str,
) -> Result<Option<ServiceTier>, anyhow::Error> {
    pick_service_tier(config::codex_service_tier().as_deref(), model_tier, model)
}

/// The configured override wins over the tier implied by the model name. Either
/// one is narrowed to a tier the upstream `model` offers.
fn pick_service_tier(
    override_tier: Option<&str>,
    model_tier: Option<ServiceTier>,
    model: &str,
) -> Result<Option<ServiceTier>, anyhow::Error> {
    let tier = match override_tier {
        Some(value) => Some(normalize_service_tier(value)?),
        None => model_tier,
    };
    Ok(tier.map(|tier| service_tier_for_model(model, tier)))
}

pub fn normalize_strict_json_schema(schema: &Value) -> Value {
    match schema {
        Value::Array(arr) => Value::Array(arr.iter().map(normalize_strict_json_schema).collect()),
        Value::Object(map) => {
            let mut out = map.clone();
            if let Some(properties) = out.get("properties").and_then(|v| v.as_object()) {
                let keys: Vec<String> = properties.keys().cloned().collect();
                out.insert(
                    "required".into(),
                    Value::Array(keys.into_iter().map(Value::String).collect()),
                );
            }
            for (key, val) in out.clone().iter() {
                out.insert(key.clone(), normalize_strict_json_schema(val));
            }
            Value::Object(out)
        }
        _ => schema.clone(),
    }
}

/// Hosted tools (web_search) are rejected by the Responses Lite lane, which
/// only supports function and custom tools. Requests carrying them must use
/// the full Responses API.
pub fn has_hosted_web_search(req: &MessagesRequest) -> bool {
    req.extra
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type").and_then(|v| v.as_str()) == Some("web_search_20250305")
            })
        })
}

pub fn translate_request(
    req: &MessagesRequest,
    opts: TranslateOptions,
) -> Result<ResponsesRequest, anyhow::Error> {
    translate_request_with_request_id(req, opts, None)
}

/// Production input skips use the same request identity as output diagnostics.
pub(crate) fn translate_request_with_request_id(
    req: &MessagesRequest,
    opts: TranslateOptions,
    req_id: Option<&str>,
) -> Result<ResponsesRequest, anyhow::Error> {
    translate_request_inner(req, opts, true, req_id)
}

pub fn translate_openai_compatible_request(
    req: &MessagesRequest,
    model: String,
    session_id: Option<String>,
) -> Result<ResponsesRequest, anyhow::Error> {
    translate_request_inner(
        req,
        TranslateOptions {
            session_id,
            service_tier: None,
            model,
            use_responses_lite: false,
        },
        false,
        None,
    )
}

fn translate_request_inner(
    req: &MessagesRequest,
    opts: TranslateOptions,
    apply_codex_config: bool,
    req_id: Option<&str>,
) -> Result<ResponsesRequest, anyhow::Error> {
    let instructions = flatten_system_text(req.extra.get("system"));
    let is_compact = is_compact_messages_request(req);
    let mut diagnostics = SkipDiagnostics::default();
    if let Some(req_id) = req_id {
        diagnostics.set_request_id(req_id);
    }
    let input = build_input(req, &mut diagnostics);
    // Claude Code compact is a text generation; historical calls stay in input.
    let text_only_compact = apply_codex_config && is_compact;
    let (tools, tool_choice) = if text_only_compact {
        (None, None)
    } else {
        (read_tools(req)?, map_tool_choice(req)?)
    };
    let parallel_tool_calls = parallel_tool_calls(req).unwrap_or(true);

    let mut text = ResponsesText {
        verbosity: Some("low".to_string()),
        format: None,
    };

    if let Some(fmt) = read_output_format(req) {
        text.format = Some(fmt);
    }

    let mut out = ResponsesRequest {
        model: opts.model,
        instructions,
        input,
        store: false,
        stream: true,
        parallel_tool_calls,
        tool_choice,
        text,
        tools: None,
        include: None,
        client_metadata: None,
        service_tier: None,
        prompt_cache_key: None,
        reasoning: None,
    };

    if opts.use_responses_lite {
        out.client_metadata = Some(std::collections::HashMap::from([(
            "ws_request_header_x_openai_internal_codex_responses_lite".to_string(),
            "true".to_string(),
        )]));
        // Responses Lite rejects parallel_tool_calls: true.
        out.parallel_tool_calls = false;

        let mut prefix = Vec::new();
        if let Some(ref tools) = tools
            && !tools.is_empty()
        {
            let tools = tools
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()?;
            prefix.push(ResponsesInputItem::AdditionalTools {
                id: None,
                role: "developer".to_string(),
                tools,
            });
        }
        if let Some(instructions) = out.instructions.take()
            && !instructions.is_empty()
        {
            prefix.push(ResponsesInputItem::Message {
                role: "developer".to_string(),
                content: vec![ResponsesContentPart::InputText { text: instructions }],
            });
        }
        if !prefix.is_empty() {
            prefix.extend(out.input);
            out.input = prefix;
        }
    } else if let Some(tools) = tools
        && !tools.is_empty()
    {
        out.tools = Some(tools);
    }

    // Never force a web_search tool_choice the request didn't register —
    // upstream 502s instead of ignoring it.
    if matches!(
        out.tool_choice,
        Some(ResponsesToolChoice::WebSearch { .. } | ResponsesToolChoice::AllowedTools { .. })
    ) {
        let has_web_search = out.tools.as_ref().is_some_and(|t| {
            t.iter()
                .any(|tool| matches!(tool, ResponsesTool::WebSearch(_)))
        });
        if !has_web_search {
            out.tool_choice = Some(ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Auto));
        }
    }

    if let Some(sid) = opts.session_id {
        out.prompt_cache_key = Some(sid);
    }

    if apply_codex_config {
        // Full and Lite requests use the session tier; compaction uses its own tier policy.
        out.service_tier = match req.proxy_speed_snapshot {
            Some(crate::proxy_speed::ProxySpeedMode::Off) => None,
            Some(_) if is_compact => resolve_service_tier(opts.service_tier, &out.model)?,
            Some(crate::proxy_speed::ProxySpeedMode::Fast) => Some(ServiceTier::Fast),
            Some(crate::proxy_speed::ProxySpeedMode::Ultrafast) => Some(ServiceTier::Ultrafast),
            None => resolve_service_tier(opts.service_tier, &out.model)?,
        };
    }

    let effort = read_effort(req)?;
    let codex_effort = to_codex_effort(effort);
    let mut resolved_effort = if apply_codex_config {
        resolve_effort(codex_effort)?
    } else {
        codex_effort
    };
    if apply_codex_config && is_compact {
        resolved_effort = apply_compact_effort_cap(resolved_effort, compact_effort_cap());
    }
    let wants_reasoning = reasoning_requested(resolved_effort.as_ref());
    if resolved_effort.is_some() || opts.use_responses_lite {
        let summary = if wants_reasoning
            && (!apply_codex_config
                || reasoning_summary_requested(config::codex_reasoning_summary().as_deref()))
        {
            Some("auto".to_string())
        } else {
            None
        };
        out.reasoning = Some(ResponsesReasoning {
            effort: resolved_effort.clone(),
            summary,
            context: opts.use_responses_lite.then_some("all_turns".to_string()),
        });
    }
    if wants_reasoning {
        out.include = Some(vec!["reasoning.encrypted_content".to_string()]);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_output_format(req: &MessagesRequest) -> Option<ResponsesTextFormat> {
    let output_config = req.extra.get("output_config")?.as_object()?;
    let format = output_config.get("format")?.as_object()?;
    let kind = format.get("type")?.as_str()?;
    match kind {
        "json_schema" => {
            let name = format
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("response")
                .to_string();
            let schema = format.get("schema")?;
            let normalized = normalize_strict_json_schema(schema);
            Some(ResponsesTextFormat::JsonSchema {
                name,
                schema: normalized,
                strict: Some(true),
            })
        }
        "json_object" => Some(ResponsesTextFormat::JsonObject),
        _ => Some(ResponsesTextFormat::Text),
    }
}

/// The standalone search route skips Responses translation, but declaration
/// errors must still be local before either upstream route is called.
pub(crate) fn validate_tool_registration(req: &MessagesRequest) -> Result<(), anyhow::Error> {
    read_tools(req)?;
    map_tool_choice(req)?;
    Ok(())
}

fn read_tools(req: &MessagesRequest) -> Result<Option<Vec<ResponsesTool>>, anyhow::Error> {
    let Some(tools) = req.extra.get("tools") else {
        return Ok(None);
    };
    let tools_arr = match tools {
        Value::Array(a) => a,
        _ => return Ok(None),
    };
    let mut out = Vec::new();
    for tool in tools_arr {
        let tool_type = tool
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("function");
        if tool_type == "web_search_20250305" {
            let mut filters = ResponsesWebSearchFilters {
                allowed_domains: None,
                blocked_domains: None,
            };
            let allowed = tool.get("allowed_domains").and_then(|v| v.as_array());
            if allowed.is_some_and(|a| !a.is_empty()) {
                filters.allowed_domains = allowed.map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                });
            }
            let blocked = tool.get("blocked_domains").and_then(|v| v.as_array());
            if blocked.is_some_and(|a| !a.is_empty()) {
                filters.blocked_domains = blocked.map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                });
            }
            let has_filters =
                filters.allowed_domains.is_some() || filters.blocked_domains.is_some();
            out.push(ResponsesTool::WebSearch(ResponsesWebSearchTool {
                kind: "web_search".to_string(),
                external_web_access: true,
                search_content_types: vec!["text".to_string(), "image".to_string()],
                filters: if has_filters { Some(filters) } else { None },
            }));
        } else {
            // Versioned Anthropic built-ins are not client functions. Unknown
            // custom types remain valid when the client supplies a JSON schema.
            if is_unsupported_server_tool_type(tool_type)
                || (tool_type != "function"
                    && tool_type != "custom"
                    && !tool.get("input_schema").is_some_and(Value::is_object))
            {
                anyhow::bail!("Unsupported server tool type: {tool_type}");
            }
            let name = tool
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if name == "ToolSearch" {
                continue;
            }
            let description = tool
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let parameters = tool
                .get("input_schema")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let description = codex_tool_description(&name, description);
            let parameters = codex_tool_parameters(&name, parameters);
            out.push(ResponsesTool::Function(ResponsesFunctionTool {
                kind: "function".to_string(),
                name,
                description,
                parameters,
                strict: false,
            }));
        }
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

fn is_unsupported_server_tool_type(kind: &str) -> bool {
    kind == "server_tool"
        || [
            "web_search_",
            "web_fetch_",
            "code_execution_",
            "tool_search_",
            "computer_",
            "bash_",
            "text_editor_",
        ]
        .iter()
        .any(|prefix| kind.starts_with(prefix))
            && kind.rsplit('_').next().is_some_and(|version| {
                version.len() == 8 && version.bytes().all(|byte| byte.is_ascii_digit())
            })
}

fn codex_tool_description(name: &str, description: Option<String>) -> Option<String> {
    if name != "Read" {
        return description;
    }

    let base = description.unwrap_or_else(|| "Reads a file from the local filesystem.".to_string());
    Some(format!("{base}\n\n{}", read_offset_guidance()))
}

// OpenAI's regex dialect differs from schemas emitted by clients. Drop pattern
// constraints rather than guessing backend compatibility with a local regex engine.
// This relaxes tool argument validation, including otherwise compatible patterns.
// Only visit schema-bearing keywords: defaults, examples and other literal data
// must remain intact, as must names in property and definition maps.
fn strip_tool_schema_patterns(schema: &mut Value) {
    let Some(schema) = schema.as_object_mut() else {
        return;
    };
    schema.remove("pattern");

    for (keyword, value) in schema {
        match keyword.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas"
            | "dependencies" => {
                if let Some(schemas) = value.as_object_mut() {
                    for schema in schemas.values_mut() {
                        strip_tool_schema_patterns(schema);
                    }
                }
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" | "items" => {
                if let Some(schemas) = value.as_array_mut() {
                    for schema in schemas {
                        strip_tool_schema_patterns(schema);
                    }
                } else if keyword == "items" {
                    strip_tool_schema_patterns(value);
                }
            }
            "additionalProperties"
            | "additionalItems"
            | "unevaluatedProperties"
            | "unevaluatedItems"
            | "contains"
            | "propertyNames"
            | "not"
            | "if"
            | "then"
            | "else"
            | "contentSchema" => strip_tool_schema_patterns(value),
            _ => {}
        }
    }
}

fn codex_tool_parameters(name: &str, mut parameters: Value) -> Value {
    strip_tool_schema_patterns(&mut parameters);
    if name != "Read" {
        return parameters;
    }

    let Some(props) = parameters
        .get_mut("properties")
        .and_then(Value::as_object_mut)
    else {
        return parameters;
    };

    if let Some(offset) = props.get_mut("offset").and_then(Value::as_object_mut) {
        offset.insert(
            "description".to_string(),
            Value::String(
                "Optional continuation index. Use only after a prior Read of the same file returned content and more lines are needed. Compute as prior offset plus returned line count. Displayed line numbers, grep line numbers, byte counts, token counts, file sizes, and guessed positions are invalid offsets. Omit when unsure.".to_string(),
            ),
        );
    }

    if let Some(limit) = props.get_mut("limit").and_then(Value::as_object_mut) {
        limit.insert(
            "description".to_string(),
            Value::String(
                "Optional number of lines to read. Omit when opening a file. Use with offset only when continuing a large file."
                    .to_string(),
            ),
        );
    }

    parameters
}

fn map_tool_choice(req: &MessagesRequest) -> Result<Option<ResponsesToolChoice>, anyhow::Error> {
    let choice = match req.extra.get("tool_choice") {
        Some(Value::Object(m)) => m,
        Some(Value::String(s)) => {
            return Ok(Some(match s.as_str() {
                "auto" => ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Auto),
                "none" => ResponsesToolChoice::Mode(ResponsesToolChoiceMode::None),
                "any" | "required" => ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Required),
                _ => ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Auto),
            }));
        }
        _ => return Ok(None),
    };

    let choice_type = choice
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");
    match choice_type {
        "auto" => Ok(Some(ResponsesToolChoice::Mode(
            ResponsesToolChoiceMode::Auto,
        ))),
        "none" => Ok(Some(ResponsesToolChoice::Mode(
            ResponsesToolChoiceMode::None,
        ))),
        "any" | "required" => Ok(Some(ResponsesToolChoice::Mode(
            ResponsesToolChoiceMode::Required,
        ))),
        "tool" => {
            let name = choice.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let tools = req.extra.get("tools").and_then(|v| v.as_array());
            let is_web_search = tools.is_some_and(|t| {
                t.iter().any(|tool| {
                    (tool.get("type").and_then(|v| v.as_str()) == Some("web_search_20250305"))
                        && tool.get("name").and_then(|v| v.as_str()) == Some(name)
                })
            });
            if is_web_search {
                Ok(Some(ResponsesToolChoice::AllowedTools {
                    r#type: "allowed_tools".to_string(),
                    mode: "required".to_string(),
                    tools: vec![serde_json::json!({"type": "web_search"})],
                }))
            } else {
                if name == "ToolSearch" {
                    anyhow::bail!(
                        "ToolSearch is excluded; choose an available client function instead"
                    );
                }
                Ok(Some(ResponsesToolChoice::Function {
                    r#type: "function".to_string(),
                    name: name.to_string(),
                }))
            }
        }
        _ => Ok(None),
    }
}

fn build_input(
    req: &MessagesRequest,
    diagnostics: &mut SkipDiagnostics,
) -> Vec<ResponsesInputItem> {
    let mut out: Vec<ResponsesInputItem> = Vec::new();
    let mut read_tool_uses_with_offset = HashSet::new();

    for msg in &req.messages {
        let blocks = normalize_codex_content(&msg.content, diagnostics);
        match msg.role.as_str() {
            "user" => {
                let mut parts: Vec<ResponsesContentPart> = Vec::new();
                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            parts.push(ResponsesContentPart::InputText { text: text.clone() });
                        }
                        ContentBlock::Image { source } => {
                            parts.push(ResponsesContentPart::InputImage {
                                image_url: image_source_to_url(source),
                                detail: None,
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            if !parts.is_empty() {
                                out.push(ResponsesInputItem::Message {
                                    role: "user".to_string(),
                                    content: std::mem::take(&mut parts),
                                });
                            }
                            let mut rendered = render_tool_result(content, diagnostics);
                            if is_error.unwrap_or(false) {
                                rendered.prepend_text("[tool execution error]".to_string());
                            }
                            if let Some(note) =
                                rewritten_read_offset_note(&rendered.joined_text(), tool_use_id)
                            {
                                rendered.push_text(format!("\n{note}"));
                            }
                            if should_append_read_offset_guidance(
                                &rendered.joined_text(),
                                read_tool_uses_with_offset.contains(tool_use_id),
                                is_error.unwrap_or(false),
                            ) {
                                rendered.push_text(format!("\n{}", read_offset_guidance()));
                            }
                            out.push(ResponsesInputItem::FunctionCallOutput {
                                call_id: tool_use_id.clone(),
                                output: function_call_output(rendered),
                            });
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    out.push(ResponsesInputItem::Message {
                        role: "user".to_string(),
                        content: parts,
                    });
                }
            }
            "system" => {
                let parts: Vec<ResponsesContentPart> = blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => {
                            Some(ResponsesContentPart::InputText { text: text.clone() })
                        }
                        _ => None,
                    })
                    .collect();
                if !parts.is_empty() {
                    out.push(ResponsesInputItem::Message {
                        role: "developer".to_string(),
                        content: parts,
                    });
                }
            }
            _ => {
                let mut text_parts: Vec<ResponsesContentPart> = Vec::new();
                let flush_text =
                    |out: &mut Vec<ResponsesInputItem>,
                     text_parts: &mut Vec<ResponsesContentPart>| {
                        if !text_parts.is_empty() {
                            out.push(ResponsesInputItem::Message {
                                role: "assistant".to_string(),
                                content: std::mem::take(text_parts),
                            });
                        }
                    };
                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts
                                .push(ResponsesContentPart::OutputText { text: text.clone() });
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            flush_text(&mut out, &mut text_parts);
                            if is_read_tool_use_with_offset(name, input) {
                                read_tool_uses_with_offset.insert(id.clone());
                            }
                            let args =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                            out.push(ResponsesInputItem::FunctionCall {
                                call_id: id.clone(),
                                name: name.clone(),
                                arguments: args,
                            });
                        }
                        ContentBlock::Thinking { signature, .. } => {
                            let Some(replay) =
                                signature.as_deref().and_then(decode_reasoning_signature)
                            else {
                                continue;
                            };
                            flush_text(&mut out, &mut text_parts);
                            out.push(ResponsesInputItem::Reasoning {
                                id: replay.id,
                                summary: Vec::new(),
                                encrypted_content: replay.encrypted_content,
                            });
                        }
                        _ => {}
                    }
                }
                flush_text(&mut out, &mut text_parts);
            }
        }
    }

    out
}

/// Preserve only the plain fields actually present in old search history.
/// Projection happens before shared normalization, in the original block order.
/// No native search item, encrypted field, or synthetic function result is replayed.
fn normalize_codex_content(
    content: &Value,
    diagnostics: &mut SkipDiagnostics,
) -> Vec<ContentBlock> {
    let Some(blocks) = content.as_array() else {
        return normalize_content(content, Value::Null);
    };
    let mut has_search = false;
    for block in blocks {
        match block["type"].as_str().unwrap_or("") {
            "web_search_tool_result" => has_search = true,
            "server_tool_use" if block["name"] == "web_search" => has_search = true,
            "text" | "image" | "tool_use" | "tool_result" | "thinking" => {}
            _ => diagnostics.record(&SkippedEvent::new(block, "unsupported_input_block")),
        }
    }
    if !has_search {
        return normalize_content(content, Value::Null);
    }
    let mut visible_text = blocks
        .iter()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = Vec::new();
    for block in blocks {
        let kind = block["type"].as_str().unwrap_or("");
        let search = kind == "web_search_tool_result"
            || (kind == "server_tool_use" && block["name"] == "web_search");
        if search {
            diagnostics.record(&SkippedEvent::new(block, "legacy_search_context_only"));
            let mut text = String::new();
            if kind == "server_tool_use" {
                if let Some(query) = block.pointer("/input/query").and_then(Value::as_str) {
                    let query_line = format!("Web search query: {query}");
                    if !query.is_empty()
                        && !visible_text
                            .lines()
                            .any(|line| line == query || line == query_line)
                    {
                        text.push_str(&query_line);
                    }
                }
            } else if let Some(results) = block["content"].as_array() {
                for result in results {
                    if result["type"] != "web_search_result" {
                        diagnostics.record(&SkippedEvent::new(
                            result,
                            "unsupported_search_history_child",
                        ));
                        continue;
                    }
                    let title = result["title"].as_str().unwrap_or("");
                    let url = result["url"].as_str().unwrap_or("");
                    if !url.is_empty()
                        && !super::web_search_compat::source_url_is_represented(&visible_text, url)
                    {
                        let link = super::web_search_compat::markdown_link(
                            if title.is_empty() { url } else { title },
                            url,
                        );
                        text.push_str(&link);
                        visible_text.push_str(&link);
                    } else if !title.is_empty() && !visible_text.contains(title) {
                        text.push_str(&format!("\nWeb search source: {title}\n"));
                    }
                    for field in ["text", "content"] {
                        if let Some(value) = result[field].as_str()
                            && !value.is_empty()
                            && !visible_text.contains(value)
                        {
                            text.push('\n');
                            text.push_str(value);
                            visible_text.push('\n');
                            visible_text.push_str(value);
                        }
                    }
                }
            }
            if !text.is_empty() {
                visible_text.push('\n');
                visible_text.push_str(&text);
                out.push(ContentBlock::Text { text });
            }
        } else {
            out.extend(normalize_content(
                &Value::Array(vec![block.clone()]),
                Value::Null,
            ));
        }
    }
    out
}

fn is_read_tool_use_with_offset(name: &str, input: &Value) -> bool {
    name == "Read" && input.get("offset").is_some()
}

fn rewritten_read_offset_note(output: &str, tool_use_id: &str) -> Option<String> {
    if output.contains("Proxy Read offset note:") {
        return None;
    }
    read_offset_rewrite(tool_use_id)
        .as_ref()
        .map(read_offset_rewrite_note)
}

fn read_offset_rewrite_note(rewrite: &ReadOffsetRewrite) -> String {
    let file = rewrite
        .file_path
        .as_deref()
        .map(|path| format!(" for {path}"))
        .unwrap_or_default();
    format!(
        "Proxy Read offset note:\n\
         - Requested Read offset {}{} exceeds the proxy rewrite threshold of 1000000.\n\
         - This Read starts at the beginning of the file.\n\
         - For continuation reads, use offset after a prior Read of the same file returned content and more lines are needed.\n\
         - Compute offset as prior offset plus the number of lines returned by that prior Read.",
        rewrite.offset, file
    )
}

fn should_append_read_offset_guidance(
    output: &str,
    read_call_had_offset: bool,
    is_error: bool,
) -> bool {
    read_call_had_offset
        && !output.contains("Codex Read guidance:")
        && looks_like_read_offset_result(output)
        && (is_error || looks_like_read_offset_warning(output))
}

fn looks_like_read_offset_result(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("offset")
        && (lower.contains("file has")
            || lower.contains("out of range")
            || (lower.contains("line") && lower.contains("requested")))
}

fn looks_like_read_offset_warning(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("warning") || lower.contains("system-reminder")
}

fn read_offset_guidance() -> &'static str {
    "Codex Read guidance:\n\
     - offset is an optional zero based continuation index, not a line number lookup.\n\
     - Use offset only after a prior Read of the same file returned content and more lines are needed.\n\
     - Compute offset as prior offset plus the number of lines returned by that prior Read.\n\
     - Displayed line numbers, grep line numbers, byte counts, token counts, file sizes, and guessed positions are invalid offsets.\n\
     - Omit offset and limit when opening a file or when unsure."
}

// ---------------------------------------------------------------------------
// Tool result rendering
// ---------------------------------------------------------------------------

enum RenderedToolResultPart {
    Text(String),
    Image(String),
}

struct RenderedToolResult {
    parts: Vec<RenderedToolResultPart>,
}

impl RenderedToolResult {
    fn has_images(&self) -> bool {
        self.parts
            .iter()
            .any(|part| matches!(part, RenderedToolResultPart::Image(_)))
    }

    fn joined_text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                RenderedToolResultPart::Text(text) => Some(text.as_str()),
                RenderedToolResultPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn prepend_text(&mut self, text: String) {
        self.parts.insert(0, RenderedToolResultPart::Text(text));
    }

    fn push_text(&mut self, text: String) {
        self.parts.push(RenderedToolResultPart::Text(text));
    }
}

fn render_tool_result(content: &Value, diagnostics: &mut SkipDiagnostics) -> RenderedToolResult {
    let parts = match content {
        Value::String(text) => vec![RenderedToolResultPart::Text(text.clone())],
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| render_tool_result_block(block, diagnostics))
            .collect(),
        _ => Vec::new(),
    };
    RenderedToolResult { parts }
}

fn render_tool_result_block(
    block: &Value,
    diagnostics: &mut SkipDiagnostics,
) -> Option<RenderedToolResultPart> {
    let part = match block.get("type").and_then(Value::as_str) {
        Some("text") => block
            .get("text")
            .and_then(Value::as_str)
            .map(|text| RenderedToolResultPart::Text(text.to_string()))
            .unwrap_or_else(|| {
                diagnostics.record(&SkippedEvent::new(block, "malformed_tool_result_child"));
                RenderedToolResultPart::Text(unsupported_tool_result_block_to_string(block))
            }),
        Some("image") => {
            let part = render_tool_result_image(block);
            if matches!(part, RenderedToolResultPart::Text(_)) {
                diagnostics.record(&SkippedEvent::new(block, "unsupported_tool_result_image"));
            }
            part
        }
        Some(kind) => {
            let reason = if kind == "tool_reference" {
                "legacy_tool_reference"
            } else {
                "unsupported_tool_result_child"
            };
            diagnostics.record(&SkippedEvent::new(block, reason));
            return None;
        }
        None => {
            diagnostics.record(&SkippedEvent::new(block, "malformed_tool_result_child"));
            RenderedToolResultPart::Text(unsupported_tool_result_block_to_string(block))
        }
    };
    Some(part)
}

fn render_tool_result_image(block: &Value) -> RenderedToolResultPart {
    let Some(source) = block.get("source").and_then(Value::as_object) else {
        return RenderedToolResultPart::Text(unsupported_tool_result_block_to_string(block));
    };
    match source.get("type").and_then(Value::as_str) {
        Some("url") if source.get("url").and_then(Value::as_str).is_some() => {
            RenderedToolResultPart::Text("[image omitted: url]".to_string())
        }
        Some("base64") => {
            let media_type = source.get("media_type").and_then(Value::as_str);
            let data = source.get("data").and_then(Value::as_str);
            match media_type
                .zip(data)
                .and_then(|(media_type, data)| validated_image_data_url(media_type, data))
            {
                Some(image_url) => RenderedToolResultPart::Image(image_url),
                None => {
                    RenderedToolResultPart::Text(unsupported_tool_result_block_to_string(block))
                }
            }
        }
        _ => RenderedToolResultPart::Text(unsupported_tool_result_block_to_string(block)),
    }
}

fn validated_image_data_url(media_type: &str, data: &str) -> Option<String> {
    if !matches!(
        media_type,
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    ) {
        return None;
    }

    let compact: String = data
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    if compact.is_empty() {
        return None;
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&compact)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&compact))
        .ok()?;
    let canonical = base64::engine::general_purpose::STANDARD.encode(decoded);
    Some(format!("data:{media_type};base64,{canonical}"))
}

fn function_call_output(rendered: RenderedToolResult) -> ResponsesFunctionCallOutput {
    if !rendered.has_images() {
        return ResponsesFunctionCallOutput::Text(rendered.joined_text());
    }

    ResponsesFunctionCallOutput::ContentItems(
        rendered
            .parts
            .into_iter()
            .map(|part| match part {
                RenderedToolResultPart::Text(text) => {
                    ResponsesFunctionCallOutputContentPart::InputText { text }
                }
                RenderedToolResultPart::Image(image_url) => {
                    ResponsesFunctionCallOutputContentPart::InputImage {
                        image_url,
                        detail: None,
                    }
                }
            })
            .collect(),
    )
}

fn unsupported_tool_result_block_to_string(block: &Value) -> String {
    let kind = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    format!("[unsupported content block omitted: {kind}]")
}

#[cfg(test)]
mod tests {
    #[test]
    fn bounded_json_counter_stops_and_preserves_serialization_errors() {
        struct Many<'a>(&'a std::cell::Cell<usize>);
        impl serde::Serialize for Many<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeSeq;
                let mut seq = serializer.serialize_seq(Some(10_000))?;
                for _ in 0..10_000 {
                    self.0.set(self.0.get() + 1);
                    seq.serialize_element("ten-bytes!")?;
                }
                seq.end()
            }
        }
        struct Broken;
        impl serde::Serialize for Broken {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("fixture serialization failure"))
            }
        }
        let visits = std::cell::Cell::new(0);
        assert_eq!(super::bounded_json_size(&Many(&visits), 64).unwrap(), None);
        assert!(visits.get() < 10);
        assert_eq!(super::bounded_json_size(&vec!["é"], 6).unwrap(), Some(6));
        assert_eq!(super::bounded_json_size(&vec!["é"], 5).unwrap(), None);
        assert!(
            super::bounded_json_size(&Broken, 64)
                .unwrap_err()
                .to_string()
                .contains("fixture serialization failure")
        );
    }

    #[test]
    fn borrowed_request_projection_preserves_every_non_input_field() {
        let request: super::ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-6-sol", "instructions":"instruction", "input":[{"type":"reasoning","id":"r","summary":[],"encrypted_content":"opaque"}],
            "tools":[{"type":"function","name":"f","parameters":{"type":"object"},"strict":true}],
            "tool_choice":"auto", "store":false, "stream":true, "parallel_tool_calls":false,
            "include":["reasoning.encrypted_content"], "client_metadata":{"z":"last","a":"first"},
            "service_tier":"priority", "prompt_cache_key":"cache", "text":{"verbosity":"low","format":{"type":"json_object"}},
            "reasoning":{"effort":"high","summary":"auto","context":"compact"}
        })).unwrap();
        let mut reference = serde_json::to_value(&request).unwrap();
        reference.as_object_mut().unwrap().remove("input");
        assert_eq!(
            serde_json::to_value(request.non_input()).unwrap(),
            reference
        );
        let wire = serde_json::to_value(request.wire(&[], false, Some("resp_before"))).unwrap();
        assert_eq!(wire["input"], serde_json::json!([]));
        assert_eq!(wire["type"], "response.create");
        assert_eq!(wire["previous_response_id"], "resp_before");
        assert!(wire.get("stream").is_none());
        let mut expected = reference;
        expected.as_object_mut().unwrap().remove("stream");
        for (key, value) in expected.as_object().unwrap() {
            assert_eq!(&wire[key], value, "field {key}");
        }
        assert_eq!(
            serde_json::to_value(request.wire(&request.input, true, None)).unwrap(),
            serde_json::to_value(&request).unwrap()
        );
    }
    use super::*;
    use serde_json::json;

    const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

    fn opts() -> TranslateOptions {
        TranslateOptions {
            session_id: None,
            service_tier: None,
            model: "gpt-5.5".to_string(),
            use_responses_lite: false,
        }
    }

    #[tokio::test]
    async fn service_tier_session_snapshot_overrides_legacy_policy_literally() {
        use crate::proxy_speed::ProxySpeedMode::{Fast, Off, Ultrafast};
        for (snapshot, global, suffix, expected) in [
            (None, None, None, None),
            (None, Some("fast"), None, Some("priority")),
            (None, None, Some(ServiceTier::Ultrafast), Some("priority")),
            (
                Some(Fast),
                Some("ultrafast"),
                Some(ServiceTier::Ultrafast),
                Some("fast"),
            ),
            (
                Some(Ultrafast),
                Some("flex"),
                Some(ServiceTier::Priority),
                Some("ultrafast"),
            ),
            (
                Some(Off),
                Some("ultrafast"),
                Some(ServiceTier::Priority),
                None,
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut env = std::collections::HashMap::from([
                ("HOME".into(), dir.path().display().to_string()),
                ("CCP_CONFIG_DIR".into(), dir.path().display().to_string()),
            ]);
            if let Some(global) = global {
                env.insert("CCP_CODEX_SERVICE_TIER".into(), global.into());
            }
            config::CodexRequestSettings::for_env(env)
                .scope(async {
                    let mut req: MessagesRequest = serde_json::from_value(json!({
                        "model":"gpt-5.5-fast", "messages":[{"role":"user","content":"hello"}],
                        "output_config":{"effort":"high"}
                    }))
                    .unwrap();
                    req.proxy_speed_snapshot = snapshot;
                    let translated = translate_request(
                        &req,
                        TranslateOptions {
                            service_tier: suffix,
                            ..opts()
                        },
                    )
                    .unwrap();
                    let wire = serde_json::to_value(&translated).unwrap();
                    assert_eq!(
                        wire.get("service_tier").and_then(Value::as_str),
                        expected,
                        "{snapshot:?} {global:?}"
                    );
                    assert_eq!(wire["model"], "gpt-5.5");
                    assert_eq!(wire["reasoning"]["effort"], "high");
                })
                .await;
        }
    }

    #[test]
    fn service_tier_snapshot_cannot_be_injected_through_json() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model":"gpt-5.5", "messages":[], "proxy_speed_snapshot":"Ultrafast"
        }))
        .unwrap();
        assert_eq!(req.proxy_speed_snapshot, None);
        let mut req = req;
        req.proxy_speed_snapshot = Some(crate::proxy_speed::ProxySpeedMode::Fast);
        assert!(
            serde_json::to_value(&req)
                .unwrap()
                .get("proxy_speed_snapshot")
                .is_none()
        );
    }

    #[tokio::test]
    async fn service_tier_session_lite_preserves_literal_tier_on_http_and_websocket() {
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            ("HOME".into(), dir.path().display().to_string()),
            ("CCP_CONFIG_DIR".into(), dir.path().display().to_string()),
        ]));
        settings
            .scope(async {
                for (mode, expected) in [
                    (crate::proxy_speed::ProxySpeedMode::Fast, "fast"),
                    (crate::proxy_speed::ProxySpeedMode::Ultrafast, "ultrafast"),
                ] {
                    let mut req: MessagesRequest = serde_json::from_value(json!({
                        "model":"gpt-6.1-sol",
                        "messages":[{"role":"user","content":"hello"}],
                        "output_config":{"effort":"xhigh"}
                    })).unwrap();
                    req.proxy_speed_snapshot = Some(mode);
                    let translated = translate_request(
                        &req,
                        TranslateOptions {
                            model: "gpt-6.1-sol".into(),
                            use_responses_lite: true,
                            ..opts()
                        },
                    ).unwrap();
                    for http in [true, false] {
                        let wire = serde_json::to_value(
                            translated.wire(&translated.input, http, None)
                        ).unwrap();
                        assert_eq!(wire["service_tier"], expected);
                        assert_eq!(wire["model"], "gpt-6.1-sol");
                        assert_eq!(wire["reasoning"]["effort"], "xhigh");
                        assert_eq!(wire["reasoning"]["context"], "all_turns");
                        assert_eq!(wire["parallel_tool_calls"], false);
                        assert_eq!(wire["client_metadata"]["ws_request_header_x_openai_internal_codex_responses_lite"], "true");
                    }
                }
            })
            .await;
    }

    #[tokio::test]
    async fn service_tier_session_keeps_compact_and_native_legacy_policy_separate() {
        let dir = tempfile::tempdir().unwrap();
        config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            ("CCP_CONFIG_DIR".into(), dir.path().display().to_string()),
            ("CCP_CODEX_SERVICE_TIER".into(), "priority".into()),
        ])).scope(async {
            for mode in [crate::proxy_speed::ProxySpeedMode::Fast, crate::proxy_speed::ProxySpeedMode::Ultrafast] {
                for (compact, lite) in [(true, false), (false, true), (true, true)] {
                    let mut req: MessagesRequest = serde_json::from_value(json!({
                        "messages":[{"role":"user","content":"Summarize the conversation"}],
                        "system":if compact {"You are a helpful AI assistant tasked with summarizing conversations."} else {"Ordinary request"}
                    })).unwrap();
                    req.proxy_speed_snapshot = Some(mode);
                    let wire = serde_json::to_value(translate_request(&req, TranslateOptions { use_responses_lite: lite, ..opts() }).unwrap()).unwrap();
                    let expected = if compact { "priority" } else { mode.as_str() };
                    assert_eq!(wire["service_tier"], expected);
                }
                let mut req: MessagesRequest = serde_json::from_value(json!({"messages":[]})).unwrap();
                req.proxy_speed_snapshot = Some(mode);
                let wire = serde_json::to_value(translate_openai_compatible_request(&req, "gpt-5.5".into(), None).unwrap()).unwrap();
                assert!(wire.get("service_tier").is_none());
            }
        }).await;
    }

    #[test]
    fn service_tier_override_values_normalize_to_wire_tiers() {
        for (value, tier, wire) in [
            ("fast", ServiceTier::Priority, "priority"),
            ("priority", ServiceTier::Priority, "priority"),
            ("ultrafast", ServiceTier::Ultrafast, "ultrafast"),
            ("flex", ServiceTier::Flex, "flex"),
        ] {
            let normalized = normalize_service_tier(value).unwrap();
            assert_eq!(normalized, tier, "{value}");
            assert_eq!(serde_json::to_value(normalized).unwrap(), json!(wire));
        }
        for value in ["", "default", "Ultrafast", "ultra-fast", "turbo"] {
            let error = normalize_service_tier(value).unwrap_err().to_string();
            assert!(error.contains("ultrafast"), "{value}: {error}");
        }
    }

    #[test]
    fn service_tier_override_wins_and_narrows_to_the_model() {
        let cases = [
            (None, None, "gpt-6-astra", None),
            (
                None,
                Some(ServiceTier::Ultrafast),
                "gpt-6-astra",
                Some(ServiceTier::Ultrafast),
            ),
            (
                None,
                Some(ServiceTier::Ultrafast),
                "gpt-6-sol",
                Some(ServiceTier::Priority),
            ),
            (
                Some("ultrafast"),
                None,
                "gpt-6-astra",
                Some(ServiceTier::Ultrafast),
            ),
            (
                Some("ultrafast"),
                Some(ServiceTier::Priority),
                "gpt-6-astra",
                Some(ServiceTier::Ultrafast),
            ),
            (
                Some("ultrafast"),
                None,
                "gpt-6-luna",
                Some(ServiceTier::Priority),
            ),
            (
                Some("fast"),
                Some(ServiceTier::Ultrafast),
                "gpt-6-astra",
                Some(ServiceTier::Priority),
            ),
            (
                Some("flex"),
                Some(ServiceTier::Ultrafast),
                "gpt-6-astra",
                Some(ServiceTier::Flex),
            ),
        ];
        for (override_tier, model_tier, model, expected) in cases {
            let picked = pick_service_tier(override_tier, model_tier.clone(), model).unwrap();
            assert_eq!(picked, expected, "{override_tier:?} {model_tier:?} {model}");
        }
        assert!(pick_service_tier(Some("turbo"), None, "gpt-6-astra").is_err());
    }

    #[test]
    fn responses_tool_choice_modes_serialize_as_openai_strings() {
        for (mode, expected) in [
            (ResponsesToolChoiceMode::Auto, json!("auto")),
            (ResponsesToolChoiceMode::None, json!("none")),
            (ResponsesToolChoiceMode::Required, json!("required")),
        ] {
            assert_eq!(
                serde_json::to_value(ResponsesToolChoice::Mode(mode)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn translate_tool_choice_preserves_wire_and_parallel_semantics() {
        for (tool_choice, expected_choice, expected_parallel) in [
            (json!({"type":"auto"}), json!("auto"), true),
            (
                json!({"type":"auto","disable_parallel_tool_use":false}),
                json!("auto"),
                true,
            ),
            (json!({"type":"none"}), json!("none"), true),
            (json!({"type":"any"}), json!("required"), true),
            (
                json!({"type":"any","disable_parallel_tool_use":true}),
                json!("required"),
                false,
            ),
            (
                json!({
                    "type":"tool",
                    "name":"test",
                    "disable_parallel_tool_use":true
                }),
                json!({"type":"function","name":"test"}),
                false,
            ),
        ] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "model": "gpt-5.5",
                "messages": [{"role":"user", "content":"use the tool"}],
                "tools": [{
                    "name":"test",
                    "input_schema":{"type":"object","properties":{}}
                }],
                "tool_choice": tool_choice
            }))
            .unwrap();
            let wire = serde_json::to_value(translate_request(&req, opts()).unwrap()).unwrap();

            assert_eq!(wire["tool_choice"], expected_choice);
            assert_eq!(wire["parallel_tool_calls"], expected_parallel);
        }
    }

    #[test]
    fn responses_lite_keeps_parallel_tool_calls_disabled() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"use the tool"}],
            "tools": [{
                "name":"test",
                "input_schema":{"type":"object","properties":{}}
            }],
            "tool_choice": {
                "type":"any",
                "disable_parallel_tool_use":false
            }
        }))
        .unwrap();
        let wire = serde_json::to_value(
            translate_request(
                &req,
                TranslateOptions {
                    model: "gpt-5.6-luna".to_string(),
                    use_responses_lite: true,
                    ..opts()
                },
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(wire["tool_choice"], json!("required"));
        assert_eq!(wire["parallel_tool_calls"], false);
    }

    #[test]
    fn translate_web_search_tool_to_codex_tool() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": [{
                "type":"web_search_20250305",
                "name":"web_search",
                "allowed_domains":["example.com"]
            }],
            "tool_choice": {"type":"tool", "name":"web_search"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                session_id: Some("s".into()),
                service_tier: None,
                model: "gpt-5.5".to_string(),
                use_responses_lite: false,
            },
        )
        .unwrap();
        assert_eq!(out.prompt_cache_key.as_deref(), Some("s"));
        assert!(matches!(
            out.tool_choice,
            Some(ResponsesToolChoice::AllowedTools { .. })
        ));
        let tool_choice = serde_json::to_value(out.tool_choice.as_ref().unwrap()).unwrap();
        assert_eq!(tool_choice["type"], "allowed_tools");
        assert_eq!(tool_choice["mode"], "required");
        assert_eq!(tool_choice["tools"], json!([{"type":"web_search"}]));
        let ResponsesTool::WebSearch(tool) = &out.tools.as_ref().unwrap()[0] else {
            panic!("expected web_search tool");
        };
        assert!(tool.external_web_access);
        assert_eq!(
            tool.filters.as_ref().unwrap().allowed_domains.as_deref(),
            Some(&["example.com".to_string()][..])
        );
        assert!(out.instructions.is_none());
    }

    #[test]
    fn automatic_filtered_web_search_keeps_native_filters() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": [{
                "type":"web_search_20250305",
                "name":"web_search",
                "allowed_domains":["example.com"],
                "blocked_domains":["spam.example"]
            }],
            "tool_choice": {"type":"auto"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let ResponsesTool::WebSearch(tool) = &out.tools.as_ref().unwrap()[0] else {
            panic!("expected web_search tool");
        };
        assert!(tool.external_web_access);
        let filters = tool.filters.as_ref().unwrap();
        assert_eq!(
            filters.allowed_domains.as_deref(),
            Some(&["example.com".to_string()][..])
        );
        assert_eq!(
            filters.blocked_domains.as_deref(),
            Some(&["spam.example".to_string()][..])
        );
        assert!(out.instructions.is_none());
    }

    #[test]
    fn forced_filtered_web_search_keeps_native_filters() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"find it"}],
            "system": "Be brief.",
            "tools": [{
                "type":"web_search_20250305",
                "name":"web_search",
                "allowed_domains":["a.example", "b.example"],
                "blocked_domains":["spam.example"]
            }],
            "tool_choice": {"type":"tool", "name":"web_search"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let ResponsesTool::WebSearch(tool) = &out.tools.as_ref().unwrap()[0] else {
            panic!("expected web_search tool");
        };
        let filters = tool.filters.as_ref().unwrap();
        assert_eq!(
            filters.allowed_domains.as_deref(),
            Some(&["a.example".to_string(), "b.example".to_string()][..])
        );
        assert_eq!(
            filters.blocked_domains.as_deref(),
            Some(&["spam.example".to_string()][..])
        );
        assert_eq!(out.instructions.as_deref(), Some("Be brief."));
        assert!(matches!(
            out.tool_choice,
            Some(ResponsesToolChoice::AllowedTools { .. })
        ));
    }

    #[test]
    fn unfiltered_web_search_adds_no_domain_instructions() {
        for tool_choice in [None, Some(json!({"type":"tool", "name":"web_search"}))] {
            let mut body = json!({
                "model": "gpt-5.5",
                "messages": [{"role":"user", "content":"find it"}],
                "tools": [{"type":"web_search_20250305", "name":"web_search"}]
            });
            if let Some(tool_choice) = tool_choice {
                body["tool_choice"] = tool_choice;
            }
            let req: MessagesRequest = serde_json::from_value(body).unwrap();
            let out = translate_request(&req, opts()).unwrap();
            let ResponsesTool::WebSearch(tool) = &out.tools.as_ref().unwrap()[0] else {
                panic!("expected web_search tool");
            };
            assert!(tool.external_web_access);
            assert!(tool.filters.is_none());
            assert!(out.instructions.is_none());
        }
    }

    #[test]
    fn has_hosted_web_search_detects_web_search_tool() {
        let with: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": [
                {"name":"Bash", "input_schema":{}},
                {"type":"web_search_20250305", "name":"web_search"}
            ]
        }))
        .unwrap();
        assert!(has_hosted_web_search(&with));

        let without: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{"role":"user", "content":"run it"}],
            "tools": [{"name":"Bash", "input_schema":{}}]
        }))
        .unwrap();
        assert!(!has_hosted_web_search(&without));
    }

    #[test]
    fn responses_lite_downgrades_unregistered_web_search_tool_choice() {
        // On the lite lane tools travel in the AdditionalTools developer
        // prefix, so a top-level web_search tool_choice would reference a
        // tool upstream doesn't know about and 502.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": [{
                "type":"web_search_20250305",
                "name":"web_search"
            }],
            "tool_choice": {"type":"tool", "name":"web_search"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                session_id: None,
                service_tier: None,
                model: "gpt-5.6-sol".to_string(),
                use_responses_lite: true,
            },
        )
        .unwrap();
        assert!(out.tools.is_none());
        assert!(matches!(
            out.tool_choice,
            Some(ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Auto))
        ));
        assert_eq!(
            serde_json::to_value(&out).unwrap()["tool_choice"],
            json!("auto")
        );
    }

    #[test]
    fn full_lane_keeps_web_search_tool_choice_registered() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": [{
                "type":"web_search_20250305",
                "name":"web_search"
            }],
            "tool_choice": {"type":"tool", "name":"web_search"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                session_id: None,
                service_tier: None,
                model: "gpt-5.6-sol".to_string(),
                use_responses_lite: false,
            },
        )
        .unwrap();
        assert!(out.tools.as_ref().is_some_and(|t| {
            t.iter()
                .any(|tool| matches!(tool, ResponsesTool::WebSearch(_)))
        }));
        assert!(matches!(
            out.tool_choice,
            Some(ResponsesToolChoice::AllowedTools { .. })
        ));
    }

    #[test]
    fn translate_artifact_tool_strips_exact_incompatible_pattern() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "name": "Artifact",
                    "description": "Manage artifacts.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "pattern": "^(?!__.*__$)[^\\p{Cc}\\p{Cf}\\p{Zl}\\p{Zp}\"\\\\./[\\]]{1,200}$",
                                "minLength": 1,
                                "maxLength": 200
                            }
                        },
                        "required": ["field"],
                        "additionalProperties": false
                    }
                }]
            }"#,
        )
        .unwrap();
        let original_tools = req.extra["tools"].clone();
        assert_eq!(
            original_tools[0]["input_schema"]["properties"]["field"]["pattern"],
            r#"^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$"#
        );
        let mut expected = original_tools[0]["input_schema"].clone();
        expected["properties"]["field"]
            .as_object_mut()
            .unwrap()
            .remove("pattern");

        let normal = translate_request(&req, opts()).unwrap();
        let compatible =
            translate_openai_compatible_request(&req, "gpt-5.5".to_string(), None).unwrap();
        let lite = translate_request(
            &req,
            TranslateOptions {
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        for out in [normal, compatible, lite] {
            let wire = serde_json::to_value(out).unwrap();
            let tool = if wire["tools"].is_array() {
                &wire["tools"][0]
            } else {
                &wire["input"][0]["tools"][0]
            };
            assert_eq!(
                tool,
                &json!({
                    "type": "function",
                    "name": "Artifact",
                    "description": "Manage artifacts.",
                    "parameters": expected,
                    "strict": false
                })
            );
        }
        assert_eq!(req.extra["tools"], original_tools);
    }

    #[test]
    fn tool_schema_patterns_are_removed_only_from_schema_locations() {
        let literal = json!({
            "pattern": "literal pattern",
            "properties": {"pattern": {"pattern": "also literal"}},
            "items": [{"pattern": "literal item"}]
        });
        let leaf = json!({
            "type": "string",
            "pattern": "^[a-z]+$",
            "description": "Keep metadata and literal data.",
            "default": literal,
            "examples": [literal],
            "const": literal,
            "enum": [literal],
            "x-custom": literal
        });
        let mut clean_leaf = leaf.clone();
        clean_leaf.as_object_mut().unwrap().remove("pattern");

        let mut schema = json!({"pattern": "^root$", "required": ["pattern"]});
        let mut expected = json!({"required": ["pattern"]});
        for keyword in [
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "dependencies",
        ] {
            schema[keyword] = json!({
                "pattern": leaf,
                "properties": {"properties": {"pattern": leaf}},
                "booleanSchema": false
            });
            expected[keyword] = json!({
                "pattern": clean_leaf,
                "properties": {"properties": {"pattern": clean_leaf}},
                "booleanSchema": false
            });
        }
        for keyword in ["allOf", "anyOf", "oneOf", "prefixItems", "items"] {
            schema[keyword] = json!([leaf, true, false]);
            expected[keyword] = json!([clean_leaf, true, false]);
        }
        for keyword in [
            "additionalProperties",
            "additionalItems",
            "unevaluatedProperties",
            "unevaluatedItems",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "contentSchema",
        ] {
            schema[keyword] = leaf.clone();
            expected[keyword] = clean_leaf.clone();
        }
        schema["dependencies"]["literalDependency"] = json!(["pattern"]);
        expected["dependencies"]["literalDependency"] = json!(["pattern"]);
        schema["$ref"] = json!("#/$defs/pattern");
        expected["$ref"] = json!("#/$defs/pattern");
        assert_eq!(codex_tool_parameters("Custom", schema), expected);

        let nested = json!({"items": {"properties": {"pattern": leaf}}});
        let expected = json!({"items": {"properties": {"pattern": clean_leaf}}});
        assert_eq!(codex_tool_parameters("Custom", nested), expected);
        for unchanged in [json!({}), json!(true), json!(false)] {
            assert_eq!(
                codex_tool_parameters("Custom", unchanged.clone()),
                unchanged
            );
        }
    }

    #[test]
    fn translate_read_tool_adds_codex_offset_guidance() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"read it"}],
            "tools": [{
                "name": "Read",
                "description": "Reads a file from the local filesystem.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "pattern": "^/"},
                        "offset": {"type": "integer", "description": "old offset"},
                        "limit": {"type": "integer", "description": "old limit"}
                    },
                    "required": ["file_path"]
                }
            }]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let tools = out.tools.as_ref().unwrap();
        let ResponsesTool::Function(tool) = &tools[0] else {
            panic!("expected function tool");
        };
        let description = tool.description.as_deref().unwrap();
        assert!(description.contains("Codex Read guidance"));
        assert!(description.contains("zero based continuation index"));
        assert!(description.contains("guessed positions are invalid offsets"));

        let props = tool
            .parameters
            .get("properties")
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(props["file_path"], json!({"type": "string"}));
        assert_eq!(
            props
                .get("offset")
                .and_then(|v| v.get("description"))
                .and_then(Value::as_str),
            Some(
                "Optional continuation index. Use only after a prior Read of the same file returned content and more lines are needed. Compute as prior offset plus returned line count. Displayed line numbers, grep line numbers, byte counts, token counts, file sizes, and guessed positions are invalid offsets. Omit when unsure."
            )
        );
        assert_eq!(
            props
                .get("limit")
                .and_then(|v| v.get("description"))
                .and_then(Value::as_str),
            Some(
                "Optional number of lines to read. Omit when opening a file. Use with offset only when continuing a large file."
            )
        );
    }

    #[test]
    fn translate_non_read_tool_preserves_tool_metadata() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"search"}],
            "tools": [{
                "name": "Search",
                "description": "Find matching records.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer", "description": "record offset"}
                    }
                }
            }]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let tools = out.tools.as_ref().unwrap();
        let ResponsesTool::Function(tool) = &tools[0] else {
            panic!("expected function tool");
        };
        assert_eq!(tool.description.as_deref(), Some("Find matching records."));
        assert!(!tool.strict);
        assert_eq!(
            serde_json::to_value(tool).unwrap()["strict"],
            Value::Bool(false)
        );
        assert_eq!(
            tool.parameters
                .get("properties")
                .and_then(|v| v.get("offset"))
                .and_then(|v| v.get("description"))
                .and_then(Value::as_str),
            Some("record offset")
        );
    }

    #[test]
    fn translate_omits_reasoning_when_not_enabled() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hello"}]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(out.reasoning.is_none());
        assert!(out.include.is_none());
    }

    #[test]
    fn translate_includes_reasoning_when_enabled() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "medium"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let reasoning = out.reasoning.unwrap();
        assert!(matches!(reasoning.effort, Some(Effort::Medium)));
        assert_eq!(reasoning.summary.as_deref(), Some("auto"));
        assert_eq!(
            out.include,
            Some(vec!["reasoning.encrypted_content".to_string()])
        );
    }

    #[test]
    fn translate_effort_max_maps_to_max() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "max"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::Max)));
    }

    #[test]
    fn translate_effort_override_max_maps_to_max() {
        let effort = resolve_effort_override(Some(Effort::Low), Some("max")).unwrap();
        assert!(matches!(effort, Some(Effort::Max)));
    }

    #[test]
    fn compact_request_detected_from_system_marker() {
        assert!(is_compact_request(Some(
            "You are a helpful AI assistant tasked with summarizing conversations."
        )));
        assert!(!is_compact_request(Some("You are Claude Code.")));
        assert!(!is_compact_request(None));
    }

    #[test]
    fn compact_request_detected_from_final_user_message() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "user", "content": "prior turn"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "result"
                        },
                        {
                            "type": "text",
                            "text": concat!(
                                "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n",
                                "Your task is to create a detailed summary of the conversation so far, ",
                                "paying close attention to the user's explicit requests."
                            )
                        }
                    ]
                }
            ],
            "system": "You are Claude Code."
        }))
        .unwrap();

        assert!(is_compact_messages_request(&req));
    }

    #[test]
    fn compact_message_markers_must_be_in_final_user_message() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [
                {
                    "role": "user",
                    "content": concat!(
                        "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n",
                        "Your task is to create a detailed summary of the conversation so far."
                    )
                },
                {"role": "user", "content": "continue normally"}
            ],
            "system": "You are Claude Code."
        }))
        .unwrap();

        assert!(!is_compact_messages_request(&req));
    }

    #[test]
    fn compact_message_requires_both_markers() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{
                "role": "user",
                "content": "Your task is to create a detailed summary of the conversation so far."
            }],
            "system": "You are Claude Code."
        }))
        .unwrap();

        assert!(!is_compact_messages_request(&req));
    }

    #[test]
    fn quoted_compact_user_instructions_remain_ordinary() {
        let prompt = concat!(
            "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n",
            "Your task is to create a detailed summary of the conversation so far."
        );
        for text in [
            format!(
                "Explain this compact prompt: \"{prompt}\" Then inspect src/main.rs with Read."
            ),
            format!("\"{prompt}\""),
            format!("```text\n{prompt}\n```"),
            format!("> {prompt}"),
        ] {
            assert!(
                !is_compact_message_text(&text),
                "quoted user prompt: {text}"
            );
        }
    }

    #[test]
    fn quoted_compact_system_instructions_remain_ordinary() {
        for text in [
            "You are Claude Code. Example prompt to discuss, not execute: \"You are a helpful AI assistant tasked with summarizing conversations.\"",
            "```text\nYou are a helpful AI assistant tasked with summarizing conversations.\n```",
        ] {
            assert!(
                !is_compact_request(Some(text)),
                "quoted system prompt: {text}"
            );
        }
    }

    #[test]
    fn quoted_compact_multiblock_instructions_remain_ordinary() {
        let prompt = concat!(
            "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n",
            "Your task is to create a detailed summary of the conversation so far."
        );
        for (system, content) in [
            (
                json!("You are Claude Code."),
                json!([
                    {"type":"text", "text":"Explain this quoted compact prompt, then inspect src/main.rs with Read:"},
                    {"type":"text", "text":prompt}
                ]),
            ),
            (
                json!([
                    {"type":"text", "text":"You are Claude Code. Discuss the following example prompt:"},
                    {"type":"text", "text":"You are a helpful AI assistant tasked with summarizing conversations."}
                ]),
                json!("Inspect src/main.rs with Read."),
            ),
        ] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "model":"gpt-5.6-sol", "system":system,
                "messages":[{"role":"user", "content":content}]
            }))
            .unwrap();
            assert!(!is_compact_messages_request(&req));
        }
    }

    #[test]
    fn compact_instruction_prefixes_allow_whitespace_and_cli_bullets() {
        assert!(is_compact_request(Some(
            " \nYou are a helpful AI assistant tasked with summarizing conversations. Keep the focus."
        )));
        assert!(is_compact_message_text(concat!(
            " \nCRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n",
            "- Do NOT use Read, Bash, or ANY other tool.\n",
            "- Your entire response must be plain text.\n\n",
            "Your task is to create a detailed summary of the conversation so far. Focus on migration."
        )));
    }

    #[test]
    fn compact_effort_cap_parses_env_values() {
        assert!(matches!(compact_effort_cap_from(None), Some(Effort::Low)));
        assert!(matches!(
            compact_effort_cap_from(Some("")),
            Some(Effort::Low)
        ));
        assert!(compact_effort_cap_from(Some("off")).is_none());
        assert!(matches!(
            compact_effort_cap_from(Some("none")),
            Some(Effort::None)
        ));
        assert!(matches!(
            compact_effort_cap_from(Some("medium")),
            Some(Effort::Medium)
        ));
        // Unrecognized values fall back to the safe default.
        assert!(matches!(
            compact_effort_cap_from(Some("bogus")),
            Some(Effort::Low)
        ));
    }

    #[test]
    fn compact_effort_cap_defaults_a_missing_effort() {
        // No effort named: take the cap instead of the upstream default.
        assert!(matches!(
            apply_compact_effort_cap(None, Some(Effort::Low)),
            Some(Effort::Low)
        ));
        assert!(matches!(
            apply_compact_effort_cap(None, Some(Effort::None)),
            Some(Effort::None)
        ));
        // An explicit effort at or below the cap survives.
        assert!(matches!(
            apply_compact_effort_cap(Some(Effort::None), Some(Effort::Low)),
            Some(Effort::None)
        ));
        assert!(matches!(
            apply_compact_effort_cap(Some(Effort::Low), Some(Effort::Low)),
            Some(Effort::Low)
        ));
        // Above the cap is lowered.
        assert!(matches!(
            apply_compact_effort_cap(Some(Effort::High), Some(Effort::Low)),
            Some(Effort::Low)
        ));
        // Cap disabled: the request is left exactly as it asked.
        assert!(apply_compact_effort_cap(None, None).is_none());
        assert!(matches!(
            apply_compact_effort_cap(Some(Effort::High), None),
            Some(Effort::High)
        ));
    }

    #[test]
    fn only_a_non_none_effort_requests_reasoning_artifacts() {
        assert!(!reasoning_requested(None));
        assert!(!reasoning_requested(Some(&Effort::None)));
        for effort in [
            Effort::Low,
            Effort::Medium,
            Effort::High,
            Effort::Xhigh,
            Effort::Max,
        ] {
            assert!(reasoning_requested(Some(&effort)));
        }
    }

    #[test]
    fn compact_request_downgrades_effort_to_cap() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"summarize"}],
            "system": "You are a helpful AI assistant tasked with summarizing conversations.",
            "output_config": {"effort": "medium"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::Low)));
    }

    #[test]
    fn compact_cap_never_raises_effort() {
        // A compact request already at or below the cap is left alone.
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"summarize"}],
            "system": "You are a helpful AI assistant tasked with summarizing conversations.",
            "output_config": {"effort": "low"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::Low)));
    }

    #[test]
    fn non_compact_request_keeps_requested_effort() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hello"}],
            "system": "You are Claude Code.",
            "output_config": {"effort": "high"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::High)));
    }

    #[test]
    fn effort_ordering_matches_variant_order() {
        assert!(Effort::None < Effort::Low);
        assert!(Effort::Low < Effort::Medium);
        assert!(Effort::Medium < Effort::High);
        assert!(Effort::High < Effort::Xhigh);
        assert!(Effort::Xhigh < Effort::Max);
    }

    #[test]
    fn max_tokens_is_not_serialized_for_codex() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "max_tokens": 4096,
            "messages": [{"role":"user", "content":"hello"}]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let value = serde_json::to_value(out).unwrap();
        assert!(value.get("max_output_tokens").is_none());
    }

    #[test]
    fn translate_effort_xhigh_maps_to_xhigh() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "xhigh"}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::Xhigh)));
        assert_eq!(
            out.include,
            Some(vec!["reasoning.encrypted_content".to_string()])
        );
    }

    #[test]
    fn reasoning_summary_override_values() {
        assert!(reasoning_summary_requested(None));
        assert!(reasoning_summary_requested(Some("auto")));
        assert!(reasoning_summary_requested(Some("detailed")));
        assert!(!reasoning_summary_requested(Some("off")));
        assert!(!reasoning_summary_requested(Some("none")));
    }

    #[test]
    fn translate_user_text_and_image() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content": [
                {"type":"text", "text":"describe"},
                {"type":"image", "source": {"type":"base64", "media_type":"image/jpeg", "data":"xyz"}}
            ]}]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 1);
        if let ResponsesInputItem::Message { role, content } = &out.input[0] {
            assert_eq!(role, "user");
            assert_eq!(content.len(), 2);
        } else {
            panic!("expected Message");
        }
    }

    #[test]
    fn translate_assistant_with_text_and_tool_use() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"assistant", "content": [
                {"type":"text", "text":"answer"},
                {"type":"tool_use", "id":"tu_1", "name":"search", "input": {"q":"rust"}}
            ]}]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 2);
    }

    #[test]
    fn translate_strict_json_schema_normalization() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content":"hi"}],
            "output_config": {"format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}},
                    "required": ["ok"]
                }
            }}
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        if let Some(ResponsesTextFormat::JsonSchema { schema, .. }) = &out.text.format {
            let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
            assert!(required.iter().any(|v| v == "ok"));
            assert!(required.iter().any(|v| v == "reason"));
        } else {
            panic!("expected JsonSchema format");
        }
    }

    // These fixtures catch loss of history IDs/data, unsafe generic server-tool
    // fallback, and deferred tooling leaking into the Responses registration.
    #[test]
    fn task4_toolsearch_excluded_full_mcp_schema_and_paired_history_survive() {
        let schema = json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false});
        let req: MessagesRequest = serde_json::from_value(json!({
            "tools":[
                {"name":"ToolSearch","input_schema":{"type":"object"}},
                {"name":"mcp__fixture__ToolSearch","defer_loading":true,"input_schema":schema},
                {"name":"SearchNotes","type":"custom","input_schema":{"type":"object"}}
            ],
            "tool_choice":{"type":"tool","name":"mcp__fixture__ToolSearch"},
            "messages":[
                {"role":"assistant","content":[
                    {"type":"text","text":"Before"},
                    {"type":"tool_use","id":"old_discovery","name":"ToolSearch","input":{}},
                    {"type":"tool_use","id":"mcp_call","name":"mcp__fixture__ToolSearch","input":{"query":"fixture"}},
                    {"type":"text","text":"Between"},
                    {"type":"tool_use","id":"notes_call","name":"SearchNotes","input":{}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"old_discovery","content":[{"type":"tool_reference","tool_name":"mcp__fixture__ToolSearch"}]},
                    {"type":"tool_result","tool_use_id":"mcp_call","content":[
                        {"type":"text","text":"before"},
                        {"type":"tool_reference","tool_name":"not_registered"},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":PNG_BASE64}},
                        {"type":"future_typed_child","payload":"DO_NOT_COPY"},
                        {"type":"text","text":"after"}
                    ]},
                    {"type":"tool_result","tool_use_id":"notes_call","content":"notes result"},
                    {"type":"text","text":"Continue"}
                ]}
            ]
        })).unwrap();
        for lite in [false, true] {
            let mut options = opts();
            options.use_responses_lite = lite;
            let out = translate_request(&req, options).unwrap();
            let wire = serde_json::to_value(&out).unwrap();
            let tools = if lite {
                &wire["input"][0]["tools"]
            } else {
                &wire["tools"]
            };
            assert_eq!(tools.as_array().unwrap().len(), 2);
            assert_eq!(tools[0]["name"], "mcp__fixture__ToolSearch");
            assert_eq!(tools[0]["parameters"], schema);
            assert_eq!(tools[1]["name"], "SearchNotes");
            assert_eq!(
                wire["tool_choice"],
                json!({"type":"function","name":"mcp__fixture__ToolSearch"})
            );
            let input = &wire["input"].as_array().unwrap()[usize::from(lite)..];
            assert_eq!(
                input[1],
                json!({"type":"function_call","call_id":"old_discovery","name":"ToolSearch","arguments":"{}"})
            );
            assert_eq!(
                input[2],
                json!({"type":"function_call","call_id":"mcp_call","name":"mcp__fixture__ToolSearch","arguments":"{\"query\":\"fixture\"}"})
            );
            assert_eq!(input[4]["call_id"], "notes_call");
            assert_eq!(
                input[5],
                json!({"type":"function_call_output","call_id":"old_discovery","output":""})
            );
            assert_eq!(
                input[6],
                json!({"type":"function_call_output","call_id":"mcp_call","output":[
                    {"type":"input_text","text":"before"},
                    {"type":"input_image","image_url":format!("data:image/png;base64,{PNG_BASE64}")},
                    {"type":"input_text","text":"after"}
                ]})
            );
            assert_eq!(
                input[7],
                json!({"type":"function_call_output","call_id":"notes_call","output":"notes result"})
            );
            assert_eq!(input[8]["content"][0]["text"], "Continue");
            let serialized = wire.to_string();
            for forbidden in [
                "tool_reference",
                "future_typed_child",
                "DO_NOT_COPY",
                "defer_loading",
                "reqId",
                "skip_diagnostics",
                "sequence_number",
            ] {
                assert!(!serialized.contains(forbidden), "{forbidden}: {wire}");
            }
        }
        let mut forced = req;
        forced.extra.insert(
            "tool_choice".into(),
            json!({"type":"tool","name":"ToolSearch"}),
        );
        assert!(
            translate_request(&forced, opts())
                .unwrap_err()
                .to_string()
                .contains("ToolSearch")
        );
    }

    #[test]
    fn task4_rejects_explicit_server_tools_but_preserves_custom_schema_tools() {
        for kind in [
            "web_search_20260209",
            "web_fetch_20250910",
            "code_execution_20250825",
            "tool_search_tool_regex_20251119",
        ] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "tools":[{"type":kind,"name":"native"}],"messages":[]
            }))
            .unwrap();
            assert!(translate_request(&req, opts()).is_err(), "{kind}");
        }
        // Compact removes registrations/choice before validating executable tools.
        let compact: MessagesRequest = serde_json::from_value(json!({
            "system":"You are a helpful AI assistant tasked with summarizing conversations.",
            "tools":[{"type":"web_fetch_20250910","name":"native"},{"name":"ToolSearch"}],
            "tool_choice":{"type":"tool","name":"ToolSearch"}, "messages":[]
        }))
        .unwrap();
        let compact = translate_request(&compact, opts()).unwrap();
        assert!(compact.tools.is_none() && compact.tool_choice.is_none());
        let req: MessagesRequest = serde_json::from_value(json!({
            "tools":[{"type":"fixture_schema_tool","name":"NativeSearch","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],"messages":[]
        })).unwrap();
        let out = serde_json::to_value(translate_request(&req, opts()).unwrap()).unwrap();
        assert_eq!(out["tools"][0]["name"], "NativeSearch");
        assert_eq!(
            out["tools"][0]["parameters"]["properties"]["q"]["type"],
            "string"
        );
    }

    #[test]
    fn task4_input_skip_diagnostics_are_bounded_and_keep_request_identity() {
        let temp = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
            ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
        ]));
        let log_path = settings.clone().sync_scope(crate::logging::log_file);
        settings.sync_scope(|| {
            let mut children = vec![json!({"type":"text","text":"supported neighbor"})];
            for _ in 0..100 {
                children.push(json!({"type":"tool_reference","tool_name":"DO_NOT_LOG"}));
            }
            for i in 0..24 {
                children.push(json!({"type":format!("future_child_{i}"),"payload":"DO_NOT_LOG"}));
            }
            let req: MessagesRequest = serde_json::from_value(json!({
                "messages":[{"role":"assistant","content":[{"type":"tool_use","id":"paired","name":"mcp__fixture__read","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"paired","content":children}]}]
            })).unwrap();
            let wire = serde_json::to_value(translate_request_with_request_id(&req, opts(), Some("task4-input-fixture")).unwrap()).unwrap();
            assert_eq!(wire["input"][1], json!({"type":"function_call_output","call_id":"paired","output":"supported neighbor"}));
        });
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let log = std::fs::read_to_string(log_path).expect("input skip diagnostics");
        assert!(!log.contains("DO_NOT_LOG"));
        let records = log
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|record| record["fields"]["reqId"] == "task4-input-fixture")
            .collect::<Vec<_>>();
        let initial = records
            .iter()
            .filter(|r| r["msg"] == "codex_event_skipped")
            .collect::<Vec<_>>();
        assert_eq!(initial.len(), 16, "{records:?}");
        assert!(
            initial
                .iter()
                .any(|r| r["fields"]["event_type"] == "tool_reference"
                    && r["fields"]["reason"] == "legacy_tool_reference")
        );
        assert!(
            initial
                .iter()
                .any(|r| r["fields"]["event_type"] == "future_child_0"
                    && r["fields"]["reason"] == "unsupported_tool_result_child")
        );
        let summaries = records
            .iter()
            .filter(|r| r["msg"] == "codex_event_skipped_summary")
            .collect::<Vec<_>>();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0]["fields"]["counts"]["tool_reference"], 100);
        assert_eq!(summaries[0]["fields"]["suppressed_count"], 9);
    }

    #[test]
    fn task4_legacy_search_history_projects_only_available_context_in_order() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "messages":[{"role":"assistant","content":[
                {"type":"text","text":"Before search https://example.test/source-long"},
                {"type":"server_tool_use","id":"native_search","name":"web_search","input":{"query":"fixture query"}},
                {"type":"web_search_tool_result","tool_use_id":"native_search","content":[
                    {"type":"web_search_result","title":"Fixture title","url":"https://example.test/source","text":"Available source text","encrypted_content":"DO_NOT_COPY"},
                    {"type":"web_search_result","title":"Already linked","url":"https://example.test/already","encrypted_index":"DO_NOT_COPY"},
                    {"type":"web_search_result","title":"Article","url":"https://en.wikipedia.org/wiki/Article_(topic)","encrypted_index":"DO_NOT_COPY"}
                ]},
                {"type":"text","text":"Answer [Already linked](https://example.test/already) [Article](https://en.wikipedia.org/wiki/Article_%28topic%29)"}
            ]},{"role":"user","content":"Continue"}]
        })).unwrap();
        let wire = serde_json::to_value(translate_request(&req, opts()).unwrap()).unwrap();
        let parts = wire["input"][0]["content"].as_array().unwrap();
        let text = parts
            .iter()
            .map(|p| p["text"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.find("Before search").unwrap()
                < text.find("fixture query").expect("available query")
        );
        assert!(
            text.find("fixture query").unwrap()
                < text.find("Fixture title").expect("available title")
        );
        assert!(
            text.find("Available source text").expect("available text")
                < text.find("Answer").unwrap()
        );
        assert!(
            text.contains("[Fixture title](https://example.test/source)"),
            "{text}"
        );
        assert_eq!(text.matches("https://example.test/already").count(), 1);
        assert_eq!(
            text.matches("[Article](https://en.wikipedia.org/wiki/Article_%28topic%29)")
                .count(),
            1,
            "{text}"
        );
        for item in wire["input"].as_array().unwrap() {
            assert_eq!(item["type"], "message");
            for part in item["content"].as_array().unwrap() {
                assert!(matches!(
                    part["type"].as_str(),
                    Some("input_text" | "output_text")
                ));
                assert_eq!(part.as_object().unwrap().len(), 2);
            }
        }
        for forbidden in [
            "server_tool_use",
            "web_search_tool_result",
            "encrypted",
            "DO_NOT_COPY",
            "function_call_output",
        ] {
            assert!(!wire.to_string().contains(forbidden), "{wire}");
        }
    }

    #[test]
    fn translate_tool_result_content() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content": [{
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [{"type":"text", "text":"result"}]
            }]}]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 1);
        if let ResponsesInputItem::FunctionCallOutput { call_id, output } = &out.input[0] {
            assert_eq!(call_id, "tu_1");
            assert_eq!(output.as_text(), Some("result"));
        } else {
            panic!("expected FunctionCallOutput");
        }
    }

    #[test]
    fn translate_read_offset_error_adds_guidance() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role":"assistant", "content": [{
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a", "offset": 2952, "limit": 200}
                }]},
                {"role":"user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "is_error": true,
                    "content": [{"type":"text", "text":"File has 331 lines, but offset 2952 was requested."}]
                }]}
            ]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 2);
        if let ResponsesInputItem::FunctionCallOutput { output, .. } = &out.input[1] {
            let output = output.as_text().expect("text tool output");
            assert!(output.contains("[tool execution error]"));
            assert!(output.contains("File has 331 lines"));
            assert!(output.contains("Codex Read guidance:"));
            assert!(output.contains("zero based continuation index"));
        } else {
            panic!("expected FunctionCallOutput");
        }
    }

    #[test]
    fn translate_read_unrelated_error_keeps_original_output() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role":"assistant", "content": [{
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a", "offset": 10, "limit": 20}
                }]},
                {"role":"user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "is_error": true,
                    "content": [{"type":"text", "text":"File does not exist."}]
                }]}
            ]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 2);
        if let ResponsesInputItem::FunctionCallOutput { output, .. } = &out.input[1] {
            assert_eq!(
                output.as_text(),
                Some("[tool execution error]\nFile does not exist.")
            );
        } else {
            panic!("expected FunctionCallOutput");
        }
    }

    #[test]
    fn translate_rewritten_read_result_adds_proxy_note() {
        crate::providers::codex::translate::read_rewrite::sanitize_read_args(
            "Read",
            r#"{"file_path":"/tmp/a","offset":1300000,"limit":20}"#,
            Some("tu_rewritten_read"),
        );
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role":"assistant", "content": [{
                    "type": "tool_use",
                    "id": "tu_rewritten_read",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a", "limit": 20}
                }]},
                {"role":"user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_rewritten_read",
                    "content": [{"type":"text", "text":"1\tcontent"}]
                }]}
            ]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 2);
        if let ResponsesInputItem::FunctionCallOutput { output, .. } = &out.input[1] {
            let output = output.as_text().expect("text tool output");
            assert!(output.contains("1\tcontent"));
            assert!(output.contains("Proxy Read offset note:"));
            assert!(output.contains("1300000"));
            assert!(output.contains("/tmp/a"));
        } else {
            panic!("expected FunctionCallOutput");
        }
    }

    #[test]
    fn translate_read_success_with_offset_words_keeps_original_output() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role":"assistant", "content": [{
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a", "offset": 10, "limit": 20}
                }]},
                {"role":"user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [{"type":"text", "text":"File has 331 lines, and the requested offset is shown in this fixture."}]
                }]}
            ]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(out.input.len(), 2);
        if let ResponsesInputItem::FunctionCallOutput { output, .. } = &out.input[1] {
            assert_eq!(
                output.as_text(),
                Some("File has 331 lines, and the requested offset is shown in this fixture.")
            );
        } else {
            panic!("expected FunctionCallOutput");
        }
    }

    #[test]
    fn translate_tool_result_preserves_mixed_content_order() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content": [{
                "type": "tool_result",
                "tool_use_id": "tu_image",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": PNG_BASE64
                    }},
                    {"type": "text", "text": "after"}
                ]
            }]}]
        }))
        .unwrap();

        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(
            serde_json::to_value(&out.input[0]).unwrap(),
            json!({
                "type": "function_call_output",
                "call_id": "tu_image",
                "output": [
                    {"type": "input_text", "text": "before"},
                    {"type": "input_image", "image_url": format!("data:image/png;base64,{PNG_BASE64}")},
                    {"type": "input_text", "text": "after"}
                ]
            })
        );
    }

    #[test]
    fn translate_tool_result_preserves_image_then_text_order() {
        let rendered = render_tool_result(
            &json!([
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": PNG_BASE64
                }},
                {"type": "text", "text": "caption"}
            ]),
            &mut SkipDiagnostics::default(),
        );

        assert_eq!(
            serde_json::to_value(function_call_output(rendered)).unwrap(),
            json!([
                {"type": "input_image", "image_url": format!("data:image/png;base64,{PNG_BASE64}")},
                {"type": "input_text", "text": "caption"}
            ])
        );
    }

    #[test]
    fn unsupported_tool_result_images_become_in_place_text_placeholders() {
        let rendered = render_tool_result(
            &json!([
                {"type": "text", "text": "before"},
                {"type": "image", "source": {
                    "type": "url",
                    "url": "https://example.invalid/a.png"
                }},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "text/plain",
                    "data": "aGVsbG8="
                }},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "not base64"
                }},
                {"type": "text", "text": "after"}
            ]),
            &mut SkipDiagnostics::default(),
        );

        assert_eq!(
            serde_json::to_value(function_call_output(rendered)).unwrap(),
            json!(
                "before\n[image omitted: url]\n[unsupported content block omitted: image]\n[unsupported content block omitted: image]\nafter"
            )
        );
    }

    #[test]
    fn supported_tool_result_image_media_types_pass_validation() {
        for media_type in ["image/jpeg", "image/png", "image/gif", "image/webp"] {
            assert_eq!(
                validated_image_data_url(media_type, "YQ"),
                Some(format!("data:{media_type};base64,YQ=="))
            );
        }
        assert!(validated_image_data_url("image/svg+xml", "YQ==").is_none());
        assert!(validated_image_data_url("image/png", "").is_none());
    }

    #[test]
    fn text_only_tool_result_keeps_string_wire_format() {
        let rendered = render_tool_result(
            &json!([
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]),
            &mut SkipDiagnostics::default(),
        );

        assert_eq!(
            serde_json::to_value(function_call_output(rendered)).unwrap(),
            json!("first\nsecond")
        );
    }

    #[test]
    fn tool_result_error_prefix_precedes_image_content() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user", "content": [{
                "type": "tool_result",
                "tool_use_id": "tu_error_image",
                "is_error": true,
                "content": [{"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": PNG_BASE64
                }}]
            }]}]
        }))
        .unwrap();

        let out = translate_request(&req, opts()).unwrap();
        assert_eq!(
            serde_json::to_value(&out.input[0]).unwrap()["output"],
            json!([
                {"type": "input_text", "text": "[tool execution error]"},
                {"type": "input_image", "image_url": format!("data:image/png;base64,{PNG_BASE64}")}
            ])
        );
    }

    #[test]
    fn malformed_tool_result_blocks_still_become_text_placeholders() {
        let rendered = render_tool_result(
            &json!([
                {"type": "text"},
                {"type": "image"},
                {}
            ]),
            &mut SkipDiagnostics::default(),
        );

        assert_eq!(
            rendered.joined_text(),
            "[unsupported content block omitted: text]\n[unsupported content block omitted: image]\n[unsupported content block omitted: unknown]"
        );
        assert!(!rendered.has_images());
    }

    #[test]
    fn luna_preserves_high_effort() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "high"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.6-luna".to_string(),
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::High)));
    }

    #[test]
    fn sol_preserves_high_effort() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "high"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.6-sol".to_string(),
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        assert!(matches!(out.reasoning.unwrap().effort, Some(Effort::High)));
    }

    #[test]
    fn responses_lite_moves_instructions_and_tools_into_input() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"hello"}],
            "system": "be helpful",
            "tools": [{"name":"test","input_schema":{"type":"object"}}]
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.6-luna".to_string(),
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        assert!(out.instructions.is_none());
        assert!(out.tools.is_none());
        assert!(!out.parallel_tool_calls);
        assert!(out.client_metadata.is_some());
        assert_eq!(out.input.len(), 3);
        assert!(matches!(
            out.input[0],
            ResponsesInputItem::AdditionalTools { .. }
        ));
        if let ResponsesInputItem::Message { role, content } = &out.input[1] {
            assert_eq!(role, "developer");
            assert!(matches!(content[0], ResponsesContentPart::InputText { .. }));
        } else {
            panic!("expected developer message");
        }
    }

    #[test]
    fn responses_lite_without_effort_uses_all_turns_context() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role":"user", "content":"hello"}]
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.6-luna".to_string(),
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        let reasoning = out.reasoning.unwrap();
        assert!(reasoning.effort.is_none());
        assert!(reasoning.summary.is_none());
        assert_eq!(reasoning.context.as_deref(), Some("all_turns"));
        assert!(out.include.is_none());
    }

    #[test]
    fn responses_lite_reasoning_uses_all_turns_context() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"hello"}],
            "output_config": {"effort": "medium"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.6-luna".to_string(),
                use_responses_lite: true,
                ..opts()
            },
        )
        .unwrap();
        assert_eq!(out.reasoning.unwrap().context.as_deref(), Some("all_turns"));
    }

    #[test]
    fn translate_returns_only_expected_top_level_fields() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role":"user", "content":"hello"}],
            "system": "be helpful",
            "tools": [{"name":"test","input_schema":{"type":"object"}}],
            "tool_choice": {"type":"tool", "name":"test"}
        }))
        .unwrap();
        let out = translate_request(
            &req,
            TranslateOptions {
                model: "gpt-5.4".to_string(),
                ..opts()
            },
        )
        .unwrap();
        assert_eq!(out.model, "gpt-5.4");
        let out_value = serde_json::to_value(&out).unwrap();
        let keys: std::collections::BTreeSet<String> =
            out_value.as_object().unwrap().keys().cloned().collect();
        for key in &[
            "model",
            "input",
            "store",
            "stream",
            "parallel_tool_calls",
            "text",
        ] {
            assert!(keys.contains(*key), "missing key: {key}");
        }
    }

    #[test]
    fn assistant_thinking_signature_replays_codex_reasoning_item() {
        let replay = super::super::reasoning_signature::ReasoningReplay {
            id: "rs_1".to_string(),
            encrypted_content: "opaque".to_string(),
        };
        let signature =
            super::super::reasoning_signature::encode_reasoning_signature(&replay).unwrap();
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role":"user","content":"start"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"visible summary","signature":signature},
                    {"type":"text","text":"done"}
                ]},
                {"role":"user","content":"continue"}
            ]
        }))
        .unwrap();
        let out = translate_request(&req, opts()).unwrap();
        let reasoning_index = out
            .input
            .iter()
            .position(|item| matches!(item, ResponsesInputItem::Reasoning { .. }))
            .unwrap();
        let ResponsesInputItem::Reasoning {
            id,
            summary,
            encrypted_content,
        } = &out.input[reasoning_index]
        else {
            unreachable!();
        };
        assert_eq!(id, "rs_1");
        assert!(summary.is_empty());
        assert_eq!(encrypted_content, "opaque");
        assert!(matches!(
            out.input.get(reasoning_index + 1),
            Some(ResponsesInputItem::Message { role, .. }) if role == "assistant"
        ));
    }
}
