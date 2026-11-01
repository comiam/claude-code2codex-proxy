use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip)]
    pub bypass_provider_model_override: bool,
    /// Trusted ingress snapshot, never accepted from or exposed through JSON.
    #[serde(skip)]
    pub proxy_speed_snapshot: Option<crate::proxy_speed::ProxySpeedMode>,
    #[serde(flatten, deserialize_with = "deserialize_extra")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn deserialize_extra<'de, D>(
    deserializer: D,
) -> Result<serde_json::Map<String, serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut extra = serde_json::Map::deserialize(deserializer)?;
    // `flatten` otherwise retains skipped field names as ordinary extra data.
    extra.remove("proxy_speed_snapshot");
    Ok(extra)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}
