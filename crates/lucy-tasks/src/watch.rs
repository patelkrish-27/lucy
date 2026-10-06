//! Web page watching: poll pages for content changes.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::WatchId;

/// The type of change detected on a watched page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// New content was added.
    ContentAdded(String),
    /// Content was removed.
    ContentRemoved(String),
    /// Existing content was modified.
    ContentModified(String),
    /// The page structure changed (e.g., different element count).
    PageStructureChanged,
}

/// A detected change on a watched page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageChange {
    pub watch_id: WatchId,
    pub url: String,
    pub change_type: ChangeType,
    pub diff: String,
    pub timestamp: u64,
}

/// A snapshot of a page's content at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSnapshot {
    pub hash: String,
    pub content: String,
    pub timestamp: u64,
}

/// A diff between two page snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<(String, String)>,
}

/// Configuration for watching a web page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchConfig {
    pub id: WatchId,
    pub url: String,
    pub interval: Duration,
    pub selector: Option<String>,
    pub last_hash: Option<String>,
    pub last_content: Option<String>,
}

impl WatchConfig {
    /// Create a new watch configuration.
    pub fn new(url: impl Into<String>, interval: Duration) -> Self {
        Self {
            id: WatchId::new(),
            url: url.into(),
            interval,
            selector: None,
            last_hash: None,
            last_content: None,
        }
    }

    /// Set a CSS selector to watch a specific element.
    pub fn with_selector(mut self, selector: impl Into<String>) -> Self {
        self.selector = Some(selector.into());
        self
    }
}

/// Watches multiple web pages for content changes.
pub struct WebPageWatcher {
    watchers: Vec<WatchConfig>,
    client: Client,
}

impl WebPageWatcher {
    /// Create a new web page watcher.
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("LucyBot/1.0 (web page watcher)")
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            watchers: Vec::new(),
            client,
        }
    }

    /// Start watching a page. Returns the watch ID.
    pub async fn watch(&mut self, config: WatchConfig) -> Result<WatchId> {
        let id = config.id;
        self.watchers.push(config);
        Ok(id)
    }

    /// Stop watching a page.
    pub async fn unwatch(&mut self, id: WatchId) -> Result<()> {
        let pos = self
            .watchers
            .iter()
            .position(|w| w.id == id)
            .with_context(|| format!("watch id {} not found", id.0))?;
        self.watchers.remove(pos);
        Ok(())
    }

    /// Check all watched pages for changes.
    pub async fn check_all(&self) -> Result<Vec<PageChange>> {
        let mut changes = Vec::new();
        for config in &self.watchers {
            if let Some(change) = self.check_page(config).await? {
                changes.push(change);
            }
        }
        Ok(changes)
    }

    /// Check a single page for changes. Returns `None` if no change detected.
    pub async fn check_page(&self, config: &WatchConfig) -> Result<Option<PageChange>> {
        let response = self
            .client
            .get(&config.url)
            .send()
            .await
            .with_context(|| format!("fetching {}", config.url))?;

        let body = response
            .text()
            .await
            .with_context(|| format!("reading body from {}", config.url))?;

        // Extract content based on selector if provided
        let content = if let Some(selector) = &config.selector {
            extract_by_selector(&body, selector)
        } else {
            body.clone()
        };

        let hash = compute_hash(&content);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // First check — just store the baseline
        if config.last_hash.is_none() {
            return Ok(Some(PageChange {
                watch_id: config.id,
                url: config.url.clone(),
                change_type: ChangeType::PageStructureChanged,
                diff: String::new(),
                timestamp,
            }));
        }

        let last_hash = config.last_hash.as_ref().unwrap();
        if &hash == last_hash {
            return Ok(None);
        }

        // Compute diff
        let last_content = config.last_content.as_deref().unwrap_or("");
        let diff = compute_diff(last_content, &content);

        let change_type = if diff.added.is_empty() && diff.removed.is_empty() && diff.modified.is_empty() {
            ChangeType::PageStructureChanged
        } else if !diff.added.is_empty() && diff.removed.is_empty() && diff.modified.is_empty() {
            ChangeType::ContentAdded(diff.added.join("\n"))
        } else if diff.added.is_empty() && !diff.removed.is_empty() && diff.modified.is_empty() {
            ChangeType::ContentRemoved(diff.removed.join("\n"))
        } else {
            ChangeType::ContentModified(format_diff(&diff))
        };

        Ok(Some(PageChange {
            watch_id: config.id,
            url: config.url.clone(),
            change_type,
            diff: format_diff(&diff),
            timestamp,
        }))
    }

    /// Get the current watch configurations.
    pub fn watchers(&self) -> &[WatchConfig] {
        &self.watchers
    }

    /// Update the last known hash and content for a watch.
    pub async fn update_baseline(&mut self, id: WatchId, hash: String, content: String) -> Result<()> {
        let config = self
            .watchers
            .iter_mut()
            .find(|w| w.id == id)
            .with_context(|| format!("watch id {} not found", id.0))?;
        config.last_hash = Some(hash);
        config.last_content = Some(content);
        Ok(())
    }
}

impl Default for WebPageWatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a SHA-256 hash of the given content.
fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract content from HTML by a simple CSS selector.
/// Supports tag names, `.class`, and `#id` selectors.
fn extract_by_selector(html: &str, selector: &str) -> String {
    let selector = selector.trim();

    if let Some(id) = selector.strip_prefix('#') {
        extract_by_id(html, id)
    } else if let Some(class) = selector.strip_prefix('.') {
        extract_by_class(html, class)
    } else {
        extract_by_tag(html, selector)
    }
}

fn extract_by_tag(html: &str, tag: &str) -> String {
    let mut result = Vec::new();
    let mut rest = html;
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    while let Some(start) = rest.find(&open) {
        let after_open = &rest[start..];
        // Find the end of the opening tag
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let content_start = start + tag_end + 1;
        let after_content_start = &rest[content_start..];
        if let Some(content_end) = after_content_start.find(&close) {
            let content = &after_content_start[..content_end];
            result.push(content.trim().to_string());
            rest = &after_content_start[content_end + close.len()..];
        } else {
            break;
        }
    }

    result.join("\n")
}

fn extract_by_id(html: &str, id: &str) -> String {
    let search = format!("id=\"{}\"", id);
    let Some(pos) = html.find(&search) else {
        return String::new();
    };
    // Find the enclosing tag
    let before = &html[..pos];
    let Some(tag_start) = before.rfind('<') else {
        return String::new();
    };
    let after = &html[pos..];
    let Some(tag_end) = after.find('>') else {
        return String::new();
    };
    let tag = &html[tag_start..pos + tag_end + 1];
    let tag_name = tag
        .trim_start_matches('<')
        .split_whitespace()
        .next()
        .unwrap_or("");
    let close = format!("</{}>", tag_name);
    let content_start = pos + tag_end + 1;
    let after_content = &html[content_start..];
    if let Some(content_end) = after_content.find(&close) {
        after_content[..content_end].trim().to_string()
    } else {
        String::new()
    }
}

fn extract_by_class(html: &str, class: &str) -> String {
    let search = format!("class=\"{}\"", class);
    let mut result = Vec::new();
    let mut rest = html;

    while let Some(pos) = rest.find(&search) {
        let before = &rest[..pos];
        let Some(tag_start) = before.rfind('<') else {
            rest = &rest[pos + search.len()..];
            continue;
        };
        let after = &rest[pos..];
        let Some(tag_end) = after.find('>') else {
            break;
        };
        let tag = &rest[tag_start..pos + tag_end + 1];
        let tag_name = tag
            .trim_start_matches('<')
            .split_whitespace()
            .next()
            .unwrap_or("");
        let close = format!("</{}>", tag_name);
        let content_start = pos + tag_end + 1;
        let after_content = &rest[content_start..];
        if let Some(content_end) = after_content.find(&close) {
            result.push(after_content[..content_end].trim().to_string());
            rest = &after_content[content_end + close.len()..];
        } else {
            break;
        }
    }

    result.join("\n")
}

/// Compute a line-based diff between two strings.
fn compute_diff(old: &str, new: &str) -> PageDiff {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut modified = Vec::new();

    // Simple LCS-based diff
    let lcs = lcs_table(&old_lines, &new_lines);

    let mut i = 0;
    let mut j = 0;
    while i < old_lines.len() && j < new_lines.len() {
        if old_lines[i] == new_lines[j] {
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            removed.push(old_lines[i].to_string());
            i += 1;
        } else {
            added.push(new_lines[j].to_string());
            j += 1;
        }
    }
    while i < old_lines.len() {
        removed.push(old_lines[i].to_string());
        i += 1;
    }
    while j < new_lines.len() {
        added.push(new_lines[j].to_string());
        j += 1;
    }

    // Detect modifications: pairs of removed+added at similar positions
    let mut final_added = Vec::new();
    let mut final_removed = Vec::new();
    let mut ai = 0;
    let mut ri = 0;
    while ai < added.len() && ri < removed.len() {
        // If the added and removed lines are similar, it's a modification
        let similarity = line_similarity(&removed[ri], &added[ai]);
        if similarity > 0.5 {
            modified.push((removed[ri].clone(), added[ai].clone()));
            ai += 1;
            ri += 1;
        } else if added[ai].len() > removed[ri].len() {
            final_added.push(added[ai].clone());
            ai += 1;
        } else {
            final_removed.push(removed[ri].clone());
            ri += 1;
        }
    }
    while ai < added.len() {
        final_added.push(added[ai].clone());
        ai += 1;
    }
    while ri < removed.len() {
        final_removed.push(removed[ri].clone());
        ri += 1;
    }

    PageDiff {
        added: final_added,
        removed: final_removed,
        modified,
    }
}

/// Compute LCS length table for two slices of strings.
fn lcs_table(a: &[&str], b: &[&str]) -> Vec<Vec<usize>> {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    dp
}

/// Simple line similarity: ratio of common characters.
fn line_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut common = 0;
    let mut b_used = vec![false; b_chars.len()];
    for &ac in &a_chars {
        for (bi, &bc) in b_chars.iter().enumerate() {
            if !b_used[bi] && ac == bc {
                common += 1;
                b_used[bi] = true;
                break;
            }
        }
    }
    let max_len = a_chars.len().max(b_chars.len());
    if max_len == 0 {
        0.0
    } else {
        common as f64 / max_len as f64
    }
}

/// Format a [`PageDiff`] as a human-readable string.
fn format_diff(diff: &PageDiff) -> String {
    let mut parts = Vec::new();
    for line in &diff.removed {
        parts.push(format!("- {}", line));
    }
    for line in &diff.added {
        parts.push(format!("+ {}", line));
    }
    for (old, new) in &diff.modified {
        parts.push(format!("- {}\n+ {}", old, new));
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(compute_hash("hello"), compute_hash("hello"));
        assert_ne!(compute_hash("hello"), compute_hash("world"));
    }

    #[test]
    fn diff_detects_additions() {
        let diff = compute_diff("line1\nline2", "line1\nline2\nline3");
        assert_eq!(diff.added, vec!["line3"]);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn diff_detects_removals() {
        let diff = compute_diff("line1\nline2\nline3", "line1\nline3");
        assert_eq!(diff.removed, vec!["line2"]);
        assert!(diff.added.is_empty());
    }

    #[test]
    fn diff_detects_modifications() {
        let diff = compute_diff("hello world", "hello there");
        assert!(!diff.modified.is_empty() || !diff.added.is_empty() || !diff.removed.is_empty());
    }

    #[test]
    fn selector_by_tag_extracts_content() {
        let html = r#"<html><body><p>Hello</p><p>World</p></body></html>"#;
        let result = extract_by_selector(html, "p");
        assert_eq!(result, "Hello\nWorld");
    }

    #[test]
    fn selector_by_id_extracts_content() {
        let html = r#"<div id="main"><p>Content here</p></div>"#;
        let result = extract_by_selector(html, "#main");
        assert_eq!(result, "<p>Content here</p>");
    }

    #[test]
    fn selector_by_class_extracts_content() {
        let html = r#"<div class="highlight">First</div><div class="highlight">Second</div>"#;
        let result = extract_by_selector(html, ".highlight");
        assert_eq!(result, "First\nSecond");
    }

    #[test]
    fn watch_config_builder_works() {
        let config = WatchConfig::new("https://example.com", Duration::from_secs(60))
            .with_selector("#content");
        assert_eq!(config.url, "https://example.com");
        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.selector.as_deref(), Some("#content"));
    }

    #[test]
    fn line_similarity_identical() {
        assert_eq!(line_similarity("hello", "hello"), 1.0);
    }

    #[test]
    fn line_similarity_different() {
        assert!(line_similarity("abc", "xyz") < 0.5);
    }
}
