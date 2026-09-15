use std::sync::Arc;
use anyhow::{Context, Result};
use lucy_adk::{LucyAdk, MemoryCandidate};
use lucy_agent::OpenAIProvider;
use lucy_core::InterruptSignal;
use serde::{Deserialize, Serialize};

const MAX_CANDIDATES: usize = 4;
const MIN_CONFIDENCE: f32 = 0.75;
const MIN_IMPORTANCE: f32 = 0.50;
const MAX_RELATED: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Extraction { #[serde(default)] memories: Vec<ExtractedMemory> }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExtractedMemory {
    kind: String,
    text: String,
    #[serde(default)] importance: f32,
    #[serde(default)] confidence: f32,
    #[serde(default)] action: String,
    #[serde(default)] supersedes: Vec<String>,
}

const SYSTEM: &str = r#"You are Lucy's persistent-memory extractor.
Extract only durable information explicitly supplied or clearly established by the USER.
Never invent, infer, or store facts solely from the assistant response.
Do not store transient requests, one-off questions, tool results, temporary state, credentials, secrets, API keys, passwords, tokens, private keys, or sensitive personal data.
Prefer concise canonical statements useful across future sessions.
Kinds: fact, preference, decision, project, instruction.
importance and confidence are numbers from 0 to 1.
Use action=add for new durable knowledge and action=update when the user explicitly corrects or replaces existing knowledge.
For updates, put the old memory text(s) actually superseded in supersedes.
Return ONLY JSON: {"memories":[{"kind":"fact|preference|decision|project|instruction","text":"...","importance":0.0,"confidence":0.0,"action":"add|update","supersedes":["..."]}]}"#;

pub async fn extract_and_store(provider: Arc<OpenAIProvider>, adk: Arc<LucyAdk>, prompt: &str, response: &str, interrupt: InterruptSignal) -> Result<()> {
    if should_skip(prompt) || response.trim().is_empty() || interrupt.is_set() { return Ok(()); }
    let related = adk.memory_context(prompt, MAX_RELATED).await.unwrap_or_default();
    let input = format!("USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}\n\nEXISTING RELATED LONG-TERM MEMORIES (advisory; newest explicit user correction wins):\n{}", if related.trim().is_empty() { "(none)" } else { &related });
    let model = provider.model();
    let value = provider.complete_json(&model, SYSTEM, &input, interrupt).await?;
    let extraction: Extraction = serde_json::from_value(value).context("invalid persistent-memory extraction schema")?;
    let mut candidates = Vec::new();
    for item in extraction.memories.into_iter().take(MAX_CANDIDATES) {
        if !valid_candidate(&item) { continue; }
        let mut text = item.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if item.action.trim().eq_ignore_ascii_case("update") && !item.supersedes.is_empty() {
            let superseded = item.supersedes.iter().take(3).map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| s.chars().take(600).collect::<String>()).collect::<Vec<_>>().join(" | ");
            if !superseded.is_empty() { text.push_str(" [supersedes: "); text.push_str(&superseded); text.push(']'); }
        }
        candidates.push(MemoryCandidate { kind: item.kind, text });
    }
    if candidates.is_empty() { return Ok(()); }
    adk.remember_candidates(&candidates).await
}

fn valid_candidate(item: &ExtractedMemory) -> bool {
    let kind = item.kind.trim().to_ascii_lowercase();
    matches!(kind.as_str(), "fact" | "preference" | "decision" | "project" | "instruction")
        && !item.text.trim().is_empty() && item.text.chars().count() <= 1000
        && item.confidence.is_finite() && item.confidence >= MIN_CONFIDENCE && item.confidence <= 1.0
        && item.importance.is_finite() && item.importance >= MIN_IMPORTANCE && item.importance <= 1.0
        && matches!(item.action.trim().to_ascii_lowercase().as_str(), "add" | "update")
        && !looks_secret(&item.text)
}

fn looks_secret(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    ["api key", "apikey", "password", "secret", "private key", "access token", "bearer token", "refresh token", "seed phrase"].iter().any(|needle| lower.contains(needle))
}

fn should_skip(prompt: &str) -> bool {
    let lower = prompt.trim().to_ascii_lowercase();
    if lower.len() < 8 { return true; }
    ["hello", "hi", "hey", "thanks", "thank you", "what is ", "what's ", "who is ", "who's ", "how do i ", "how can i ", "please open ", "open ", "close ", "run ", "search for ", "look up ", "show me ", "tell me ", "what time ", "weather"].iter().any(|needle| lower == *needle || lower.starts_with(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn item(kind: &str, text: &str, confidence: f32, importance: f32, action: &str) -> ExtractedMemory { ExtractedMemory { kind: kind.into(), text: text.into(), importance, confidence, action: action.into(), supersedes: vec![] } }
    #[test] fn accepts_durable_memory() { assert!(valid_candidate(&item("preference", "I prefer concise terminal answers", 0.95, 0.8, "add"))); }
    #[test] fn rejects_low_quality_memory() { assert!(!valid_candidate(&item("fact", "I use Rust", 0.5, 0.9, "add"))); assert!(!valid_candidate(&item("unknown", "I use Rust", 0.9, 0.9, "add"))); assert!(!valid_candidate(&item("fact", "My API key is abc", 0.99, 0.99, "add"))); }
    #[test] fn skips_transient_requests() { assert!(should_skip("open the browser and search for docs")); assert!(should_skip("what is Rust?")); assert!(!should_skip("I prefer Rust for new projects")); }
}
