//! ClaudeCode search UI compatibility using actual calls and source metadata.
//! These cards carry no Anthropic opaque replay credentials.
/// Escape Markdown delimiters without harvesting or inventing any source.
pub(crate) fn markdown_link(title: &str, url: &str) -> String {
    let title = title
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace(['\n', '\r'], " ");
    let url = escaped_markdown_url(url);
    format!("\n[{title}]({url})\n")
}

fn escaped_markdown_url(url: &str) -> String {
    url.replace(' ', "%20")
        .replace('(', "%28")
        .replace(')', "%29")
        .replace(['\n', '\r'], "")
}

/// Match an entire URL token in plain text or Markdown. A different source
/// whose URL merely starts with this one must not suppress the real citation.
pub(crate) fn source_url_is_represented(text: &str, url: &str) -> bool {
    if url.is_empty() {
        return false;
    }
    let boundary = |c: char| {
        c.is_whitespace() || matches!(c, '(' | ')' | '[' | ']' | '<' | '>' | '\'' | '"' | '`')
    };
    let represented = |candidate: &str| {
        !candidate.is_empty()
            && text.match_indices(candidate).any(|(start, _)| {
                text[..start].chars().next_back().is_none_or(boundary)
                    && text[start + candidate.len()..]
                        .chars()
                        .next()
                        .is_none_or(boundary)
            })
    };
    if represented(url) {
        return true;
    }
    // Compare the same representation emitted by markdown_link as well as
    // the raw source, without decoding or otherwise normalizing source URLs.
    let emitted = escaped_markdown_url(url);
    emitted != url && represented(&emitted)
}

/// A completed OpenAI search projected for ClaudeCode's search counter/cards.
/// Native Anthropic replay is unavailable; encrypted fields are never invented.
#[derive(Debug, Clone)]
pub(crate) struct SearchCard {
    pub id: String,
    pub input: serde_json::Value,
    pub results: Vec<WebSearchResult>,
}

impl SearchCard {
    pub(crate) fn blocks(&self) -> [serde_json::Value; 2] {
        use serde_json::json;
        [
            json!({"type":"server_tool_use", "id":self.id, "name":"web_search", "input":self.input}),
            json!({"type":"web_search_tool_result", "tool_use_id":self.id,
                "content": self.results.iter().map(|result| json!({"type":"web_search_result", "title":result.title, "url":result.url})).collect::<Vec<_>>() }),
        ]
    }
}

impl WebSearchResult {
    pub(crate) fn from_value(value: &serde_json::Value) -> Option<Self> {
        let url = value["url"]
            .as_str()
            .filter(|url| url.starts_with("https://") || url.starts_with("http://"))?;
        Some(Self {
            title: value["title"]
                .as_str()
                .filter(|title| !title.is_empty())
                .unwrap_or(url)
                .to_owned(),
            url: url.to_owned(),
        })
    }
}

// Legacy public facade: OpenAI search lacks Anthropic's required opaque replay.
// Keep source compatibility while declining to fabricate native result cards.
use serde_json::Value;

pub struct WebSearchCompatBlock {
    pub index: usize,
    pub content: WebSearchCompatContent,
}

pub enum WebSearchCompatContent {
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: Vec<WebSearchResult>,
    },
}

#[derive(Debug, Clone)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
}

pub fn server_tool_use_id_from_codex_web_search_id(id: &str) -> String {
    let suffix: String = id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("srvtoolu_{suffix}")
}

pub fn build_web_search_compat_blocks(
    _searches: &[super::reducer::ReducerEvent],
    _text: &str,
) -> Vec<WebSearchCompatBlock> {
    Vec::new()
}
