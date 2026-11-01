use super::translate::request::{
    ResponsesContentPart, ResponsesFunctionCallOutput, ResponsesFunctionCallOutputContentPart,
    ResponsesInputItem, ResponsesRequest, ResponsesTool,
};
use std::sync::{Arc, OnceLock};
use tiktoken_rs::o200k_base_singleton;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Exact endpoint preparation has its own CPU budget, independent of inference.
pub(crate) struct CountingGate {
    admission: Arc<Semaphore>,
    execution: Arc<Semaphore>,
}

pub(crate) struct CountingPermits {
    _admission: OwnedSemaphorePermit,
    _execution: OwnedSemaphorePermit,
}

impl CountingGate {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            admission: Arc::new(Semaphore::new(6)),
            execution: Arc::new(Semaphore::new(2)),
        })
    }

    pub(crate) fn shared() -> Arc<Self> {
        static GATE: OnceLock<Arc<CountingGate>> = OnceLock::new();
        Arc::clone(GATE.get_or_init(Self::new))
    }

    /// Reject excess work before spawning. Cancellation while waiting drops
    /// admission; both permits must then move into the actual blocking job.
    pub(crate) async fn acquire(&self) -> Result<CountingPermits, ()> {
        let admission = Arc::clone(&self.admission)
            .try_acquire_owned()
            .map_err(|_| ())?;
        let execution = Arc::clone(&self.execution)
            .acquire_owned()
            .await
            .map_err(|_| ())?;
        Ok(CountingPermits {
            _admission: admission,
            _execution: execution,
        })
    }
}

/// Approximate token counter for Codex translated requests.
/// Text uses the OpenAI `o200k_base` tokenizer. Images, encrypted reasoning,
/// and protocol framing still use local estimates, so the result is not a
/// provider billing count.
pub fn count_translated_tokens(translated: &ResponsesRequest) -> u64 {
    count_with_text_counter(translated, approx_token_count)
}

/// Live estimates deliberately avoid BPE work and are replaced by final usage.
pub(crate) fn estimate_translated_tokens(translated: &ResponsesRequest) -> u64 {
    #[cfg(test)]
    observations::ESTIMATES.set(observations::ESTIMATES.get() + 1);
    count_with_text_counter(translated, cheap_text_count)
}

#[cfg(test)]
pub(crate) mod observations {
    std::thread_local! {
        pub(crate) static BPE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(crate) static ESTIMATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
}

fn cheap_text_count(text: &str) -> u64 {
    // Non-ASCII scalars have a larger typical token cost than English bytes.
    // This also gives long unbroken strings a meaningful nonzero estimate.
    let mut ascii = 0u64;
    let mut unicode = 0u64;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii += 1;
        } else {
            unicode += 1;
        }
    }
    ascii.div_ceil(4) + unicode
}

fn count_with_text_counter(translated: &ResponsesRequest, text_count: fn(&str) -> u64) -> u64 {
    let mut total = 0u64;

    // Instructions
    if let Some(ref instructions) = translated.instructions {
        total += text_count(instructions);
    }

    // Input items
    for item in &translated.input {
        total += count_input_item_tokens(item, text_count);
    }

    // Tools
    if let Some(ref tools) = translated.tools {
        total += count_tool_tokens(tools, text_count);
    }

    // Overhead
    total += translated.input.len() as u64 * 4;
    total += translated.tools.as_ref().map_or(0, |t| t.len() as u64 * 4);

    // Model name
    total += text_count(&translated.model);

    total.max(1)
}

fn count_input_item_tokens(item: &ResponsesInputItem, text_count: fn(&str) -> u64) -> u64 {
    match item {
        ResponsesInputItem::AdditionalTools { tools, .. } => tools
            .iter()
            .map(|tool| text_count(&serde_json::to_string(tool).unwrap_or_default()))
            .sum(),
        ResponsesInputItem::Message { content, .. } => {
            let mut total = 0u64;
            for part in content {
                total += count_content_part_tokens(part, text_count);
            }
            total
        }
        ResponsesInputItem::FunctionCall {
            name, arguments, ..
        } => text_count(name) + text_count(arguments),
        ResponsesInputItem::FunctionCallOutput { output, .. } => {
            count_function_call_output_tokens(output, text_count)
        }
        ResponsesInputItem::Reasoning {
            encrypted_content, ..
        }
        | ResponsesInputItem::Compaction { encrypted_content } => {
            approx_reasoning_token_count(encrypted_content)
        }
        ResponsesInputItem::CompactionTrigger => 0,
    }
}

fn count_function_call_output_tokens(
    output: &ResponsesFunctionCallOutput,
    text_count: fn(&str) -> u64,
) -> u64 {
    match output {
        ResponsesFunctionCallOutput::Text(text) => text_count(text),
        ResponsesFunctionCallOutput::ContentItems(content) => content
            .iter()
            .map(|part| match part {
                ResponsesFunctionCallOutputContentPart::InputText { text } => text_count(text),
                ResponsesFunctionCallOutputContentPart::InputImage { .. } => 2000,
            })
            .sum(),
    }
}

fn approx_reasoning_token_count(encoded_content: &str) -> u64 {
    let model_visible_bytes = encoded_content
        .len()
        .saturating_mul(3)
        .checked_div(4)
        .unwrap_or(0)
        .saturating_sub(650);
    u64::try_from(model_visible_bytes.saturating_add(3) / 4).unwrap_or(u64::MAX)
}

fn count_content_part_tokens(part: &ResponsesContentPart, text_count: fn(&str) -> u64) -> u64 {
    match part {
        ResponsesContentPart::InputText { text } => text_count(text),
        ResponsesContentPart::OutputText { text } => text_count(text),
        ResponsesContentPart::InputImage { .. } => 2000, // Image token estimate
    }
}

fn count_tool_tokens(tools: &[ResponsesTool], text_count: fn(&str) -> u64) -> u64 {
    let mut total = 0u64;
    for tool in tools {
        match tool {
            ResponsesTool::Function(f) => {
                total += text_count(&f.name);
                if let Some(ref desc) = f.description {
                    total += text_count(desc);
                }
                total += text_count(&serde_json::to_string(&f.parameters).unwrap_or_default());
            }
            ResponsesTool::WebSearch(_) => {
                total += 10; // fixed overhead for web search tool
            }
        }
    }
    total
}

pub(crate) fn approx_token_count(text: &str) -> u64 {
    #[cfg(test)]
    observations::BPE_CALLS.set(observations::BPE_CALLS.get() + 1);
    u64::try_from(o200k_base_singleton().count_ordinary(text)).unwrap_or(u64::MAX)
}

pub(crate) fn truncate_to_token_budget(text: &str, max_tokens: u64) -> String {
    let max_tokens = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    if max_tokens == 0 {
        return String::new();
    }

    let tokenizer = o200k_base_singleton();
    let tokens = tokenizer.encode_ordinary(text);
    if tokens.len() <= max_tokens {
        return text.to_string();
    }

    // A token boundary may split a multi-byte UTF-8 scalar. Back up until the
    // decoded prefix is valid rather than replacing bytes or splitting text.
    for end in (1..=max_tokens).rev() {
        if let Ok(prefix) = tokenizer.decode(&tokens[..end]) {
            return prefix;
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exact_fixture_counts_preserve_unicode_images_encryption_and_schema() {
        // Fixed original o200k fixture counts: 512 ASCII, 2688 Unicode, one
        // token for {}, 2000 per image, 588 for the encrypted payload.
        for (text, want) in [
            ("a".repeat(4096), 3113),
            ("汉字测试上下文压缩".repeat(384), 5289),
        ] {
            let request: ResponsesRequest = serde_json::from_value(json!({
                "model":"", "input":[
                    {"type":"message","role":"user","content":[{"type":"input_text","text":text},{"type":"input_image","image_url":"data:image/png;base64,YQ=="}]},
                    {"type":"reasoning","id":"r","summary":[],"encrypted_content":"A".repeat(4000)}
                ],
                "tools":[{"type":"function","name":"","parameters":{}}],
                "store":false,"stream":true,"parallel_tool_calls":false,"text":{}
            })).unwrap();
            assert_eq!(count_translated_tokens(&request), want);
        }
    }

    #[test]
    fn cheap_estimate_counts_unbroken_unicode_and_shared_nontext_costs() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"", "input":[
                {"type":"message","role":"user","content":[{"type":"input_text","text":"a".repeat(4096)},{"type":"input_image","image_url":"unused"}]},
                {"type":"reasoning","id":"r","summary":[],"encrypted_content":"A".repeat(4000)}
            ],"store":false,"stream":true,"parallel_tool_calls":false,"text":{}
        })).unwrap();
        assert_eq!(estimate_translated_tokens(&request), 3620);
        assert_eq!(cheap_text_count("汉字测试上下文压缩"), 9);
        assert_eq!(cheap_text_count(""), 0);
    }

    #[test]
    fn count_simple_request() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let count = count_translated_tokens(&req);
        assert!(count > 0);
    }

    #[test]
    fn count_request_with_tools() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "use tool"}]}],
            "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let count = count_translated_tokens(&req);
        assert!(count > 0);
    }

    #[test]
    fn count_is_monotonic() {
        let short: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let long: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "this is a much longer message with many words in it"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        assert!(count_translated_tokens(&long) >= count_translated_tokens(&short));
    }

    #[test]
    fn o200k_counts_unbroken_text_instead_of_collapsing_it() {
        assert_eq!(approx_token_count(&"a".repeat(4096)), 512);
        assert_eq!(approx_token_count(&"汉字测试上下文压缩".repeat(384)), 2688);
        assert_eq!(
            approx_token_count(&"QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo".repeat(112)),
            2240
        );
        assert_eq!(
            approx_token_count(&"function f(a){return a===0?1:a*f(a-1)};".repeat(96)),
            1632
        );
    }

    #[test]
    fn truncation_respects_o200k_budget_and_utf8_boundaries() {
        for text in [
            "a".repeat(4096),
            "汉字测试上下文压缩".repeat(384),
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo".repeat(112),
        ] {
            let truncated = truncate_to_token_budget(&text, 100);
            assert!(text.starts_with(&truncated));
            assert!(approx_token_count(&truncated) <= 100);
            assert!(!truncated.is_empty());
        }
    }

    #[test]
    fn structured_tool_output_counts_text_and_images() {
        let text = ResponsesFunctionCallOutput::Text("caption".to_string());
        let mixed = ResponsesFunctionCallOutput::ContentItems(vec![
            ResponsesFunctionCallOutputContentPart::InputText {
                text: "caption".to_string(),
            },
            ResponsesFunctionCallOutputContentPart::InputImage {
                image_url: "data:image/png;base64,YQ==".to_string(),
                detail: None,
            },
        ]);

        assert_eq!(
            count_function_call_output_tokens(&mixed, approx_token_count),
            count_function_call_output_tokens(&text, approx_token_count) + 2000
        );
    }

    #[test]
    fn encrypted_reasoning_uses_codex_model_visible_size_estimate() {
        let encoded_content = "A".repeat(4000);
        assert_eq!(approx_reasoning_token_count(&encoded_content), 588);
        assert_eq!(approx_reasoning_token_count("short"), 0);
    }
}
