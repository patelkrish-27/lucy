//! What an error looks like to the person reading it.
//!
//! Every failure in Lucy reaches a reader as a string: a provider body, an MCP
//! result envelope, an `anyhow` chain, a status line. Left alone those arrive as
//! what they are on the wire — `{"error":{"message":"...","code":429}}`, a bare
//! `429`, a URL — which is a debugging artifact, not an explanation. This module
//! is the one place that turns any of them into a sentence plus, when we know
//! one, a next step.
//!
//! Two rules, and they are the whole contract:
//!
//! 1. **Never show a payload.** A JSON body is read for its message and then
//!    dropped; a body that is not JSON at all (an HTML error page, say) becomes
//!    "that endpoint did not answer with JSON". [`explain`] cannot return text
//!    that still looks like wire data.
//! 2. **Never lose the detail.** The raw string stays exactly where it was —
//!    `tracing`, the model-call log, the returned `anyhow` error — and only the
//!    reader-facing rendering changes. [`explain`] is a view, not a filter.
//!
//! The classification keys off *protocol* facts: JSON error-envelope shapes, HTTP
//! status codes, transport error text. Nothing here knows what a task is.

use serde_json::Value;

/// Longest reader-facing message. Long enough for a provider sentence, short
/// enough that a chat line still reads as a line.
pub const MAX_MESSAGE_CHARS: usize = 220;

/// A failure, split into what went wrong and what to do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Explained {
    /// What went wrong, in one sentence.
    pub message: String,
    /// The next step, when the failure implies one.
    pub hint: Option<String>,
    /// The HTTP status this failure was read from, when there was one.
    ///
    /// Carried rather than re-derived so a caller can key its own advice off the
    /// *status* instead of searching the text for digits. A caller that matches
    /// `"404"` as a substring misreads any URL containing those digits — a stub
    /// server on port 404 once turned a 401 "check your API key" into "add /v1".
    pub status: Option<u16>,
}

impl Explained {
    /// The single line a status bar, chat message or push notification shows.
    pub fn one_line(&self) -> String {
        match &self.hint {
            Some(h) if !h.is_empty() => format!("{} — {h}", self.message),
            _ => self.message.clone(),
        }
    }

    /// Whether this failure carries `code` as its HTTP status.
    pub fn is_status(&self, code: u16) -> bool {
        self.status == Some(code)
    }
}

/// Turn any error text into a reader-facing sentence.
///
/// Accepts everything a failing call can hand back: a provider JSON body, an
/// MCP result envelope, a status-prefixed string, an `anyhow` chain flattened
/// with `{:#}`, or a plain sentence. The result never contains a JSON payload.
pub fn explain(raw: &str) -> Explained {
    let text = raw.trim();
    if text.is_empty() {
        return Explained {
            message: "something went wrong, with no detail to show".to_owned(),
            hint: None,
            status: None,
        };
    }
    // A body that *is* JSON carries the provider's own words. Prefer them over
    // our rendering of the status line around it: the status says *what class* of
    // failure, the body says *which* failure.
    //
    // Only a value that parsed counts. `payload_message` returns a plain string
    // unchanged, which is right for a tool result and wrong here — an HTML error
    // page would be adopted as its own explanation and printed verbatim, which is
    // the exact leak the `reads_as_payload` arm below exists to prevent.
    let parsed = serde_json::from_str::<Value>(text).ok();
    let from_json = parsed
        .as_ref()
        .and_then(payload_message)
        .or_else(|| embedded_message(text));
    // The status is resolved once, here, from wherever this body carries it: the
    // status line around it, a `"code"` field inside it, or a JSON tail embedded
    // in a longer string. Everything downstream reads this field and never the
    // text, so no caller has to search for digits and misread a URL or a port.
    let embedded = parsed.is_none().then(|| embedded_value(text)).flatten();
    let status = http_status(text)
        .or_else(|| parsed.as_ref().and_then(json_status))
        .or_else(|| embedded.as_ref().and_then(json_status));
    if let Some(message) = from_json {
        return Explained {
            message: clean(&message),
            hint: hint_for(text, status),
            status,
        };
    }
    let message = clean(text);
    // Text that survived cleaning but still reads as wire data is not a
    // sentence. Say what is known instead of printing the bytes.
    if reads_as_payload(&message) {
        return Explained {
            message: "the endpoint answered with data Lucy could not read".to_owned(),
            hint: Some(HINT_NOT_JSON.to_owned()),
            status,
        };
    }
    let hint = hint_for(text, status);
    Explained {
        message: match hint.as_deref() {
            Some(HINT_OFFLINE) => "the local model server is not reachable".to_owned(),
            Some(HINT_DNS) => "the model host could not be resolved".to_owned(),
            _ => message,
        },
        hint,
        status,
    }
}

/// [`explain`] as the one line most surfaces actually want.
pub fn friendly(raw: &str) -> String {
    explain(raw).one_line()
}

/// The human message inside a tool result, when it has one.
///
/// A tool result is a `Value`, not a string, and the same shapes show up in both
/// places: an MCP envelope (`{"content":[{"text":"error: ..."}]}`), a server's
/// `{"error":{"message":...}}`, a stringified result nested as text. One
/// extractor serves both, so a failure read out of a `Value` and the same
/// failure read out of a string are worded identically.
pub fn payload_message(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                return None;
            }
            // MCP servers stringify their result inside `text`, so the message
            // is one parse deeper. Unwrap once per level rather than
            // special-casing the envelope.
            match serde_json::from_str::<Value>(t) {
                Ok(inner) => payload_message(&inner).or_else(|| Some(t.to_owned())),
                Err(_) => Some(t.to_owned()),
            }
        }
        Value::Object(map) => MESSAGE_KEYS
            .iter()
            .find_map(|k| map.get(*k))
            .and_then(payload_message),
        Value::Array(items) => items.iter().find_map(payload_message),
        _ => None,
    }
}

/// Keys that carry a human message in some error envelope, most specific first.
/// An order, not a lookup table of services: `error` may be a string *or* an
/// object holding `message`, and both resolve through the same recursion.
const MESSAGE_KEYS: [&str; 10] = [
    "error",
    "message",
    "detail",
    "error_description",
    "msg",
    "reason",
    "description",
    "content",
    "text",
    "output",
];

const HINT_KEY: &str =
    "open /settings → Connect providers and check the API Key, then press [Test] and [Save]";
const HINT_MODEL: &str =
    "check the model name is one the provider lists — /settings → Connect providers [Test] refreshes it";
const HINT_NOT_JSON: &str = "the URL probably needs its /v1 suffix and must be OpenAI-compatible";
const HINT_RATE: &str = "wait a few seconds, or pick another model in /settings";
const HINT_CREDIT: &str = "this key has no credit left on the provider — top it up or connect another provider";
const HINT_SERVER: &str = "the provider had a problem on its side — try again in a moment";
const HINT_OFFLINE: &str = "start the local server, or point the API URL at a provider that is running";
const HINT_DNS: &str = "check the API URL for a typo in the hostname";
const HINT_TIMEOUT: &str =
    "the endpoint may be slow to answer — try a smaller request, or another model in /settings";
const HINT_CONTEXT: &str = "the request is longer than this model accepts — start a new session with /new";
const HINT_NO_MODEL: &str = "pick a model in /settings → Chat mode → Level 3";
const HINT_EMPTY: &str = "the model answered with nothing — try again, or pick another model in /settings";

/// A numeric error code carried *inside* a JSON body.
///
/// `"code":402` next to `"message":"…"` is the whole diagnosis in some provider
/// responses, and it never reaches the flat text because the body has no spaces
/// to tokenise. `hint_for_code` is what turns it into advice.
fn json_status(value: &Value) -> Option<u16> {
    match value {
        Value::Object(map) => {
            // This level first, then deeper: the code sits next to the message in
            // most bodies (`{"error":{"message":…,"code":402}}`), which is one
            // object below the root.
            for key in ["code", "status", "status_code", "statusCode", "type"] {
                if let Some(found) = map.get(key).and_then(status_of) {
                    return Some(found);
                }
            }
            map.values().find_map(json_status)
        }
        Value::Array(items) => items.iter().find_map(json_status),
        _ => None,
    }
}

/// A status carried by one field: a number, or a name/phrase that spells it out.
fn status_of(value: &Value) -> Option<u16> {
    match value {
        Value::Number(n) => n.as_u64().and_then(|n| u16::try_from(n).ok()),
        // Some bodies carry the status as its reason phrase, e.g.
        // `"type":"rate_limit_error"` or `"status":"Too Many Requests"`.
        Value::String(s) => reason_phrase(s).or_else(|| code_from_name(s)),
        _ => None,
    }
}

/// `"Too Many Requests"` → `429`.
fn reason_phrase(s: &str) -> Option<u16> {
    let head = marker_word(s)?;
    if REASON_PHRASES.contains(&head.as_str()) {
        // The phrase names the class; the exact code for each is spelled out
        // rather than derived, because several phrases share a first word
        // (`Not Found` / `Not Acceptable` / `Not Implemented` all start `not`).
        return match head.as_str() {
            "bad" => Some(400),
            "unauthorized" => Some(401),
            "payment" => Some(402),
            "forbidden" => Some(403),
            "not" => Some(404),
            "method" => Some(405),
            "too" => Some(429),
            "unprocessable" => Some(422),
            "internal" => Some(500),
            "implemented" => Some(501),
            "gateway" => Some(502),
            _ => None,
        };
    }
    None
}

/// `"rate_limit_error"` / `HTTP_429` → `429`.
fn code_from_name(s: &str) -> Option<u16> {
    let digits: String = s
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse::<u16>()
        .ok()
        .filter(|c| (400..600).contains(c))
}

/// The next step for a status we read out of a JSON body rather than the text.
fn hint_for_code(code: u16) -> Option<String> {
    match code {
        400 | 422 => Some("the request was rejected as malformed — /doctor can show the full text".to_owned()),
        401 | 403 => Some(HINT_KEY.to_owned()),
        402 => Some(HINT_CREDIT.to_owned()),
        404 => Some(HINT_MODEL.to_owned()),
        408 => Some(HINT_TIMEOUT.to_owned()),
        429 => Some(HINT_RATE.to_owned()),
        code if (500..600).contains(&code) => Some(HINT_SERVER.to_owned()),
        _ => None,
    }
}

/// The next step implied by the failure, if any.
///
/// `status` is the already-resolved HTTP code ([`explain`] found it in the status
/// line or a `"code"` field); the keyword checks run first because they name
/// *why* a code happened — a rate limit and a credit exhaustion are both 429-adjacent
/// advice, and only the words tell them apart.
fn hint_for(text: &str, status: Option<u16>) -> Option<String> {
    let low = text.to_ascii_lowercase();
    // An error `code` in a provider body is the most specific signal there is,
    // and it often arrives without a status line around it.
    if low.contains("rate_limit") || low.contains("too many requests") {
        return Some(HINT_RATE.to_owned());
    }
    if low.contains("insufficient_quota")
        || low.contains("billing")
        || low.contains("payment required")
        || low.contains("no credit")
    {
        return Some(HINT_CREDIT.to_owned());
    }
    if low.contains("context_length") || low.contains("maximum context") {
        return Some(HINT_CONTEXT.to_owned());
    }
    if low.contains("invalid_api_key")
        || low.contains("invalid api key")
        || low.contains("incorrect api key")
        || low.contains("unauthorized")
    {
        return Some(HINT_KEY.to_owned());
    }
    if low.contains("model_not_found") || low.contains("unknown model") {
        return Some(HINT_MODEL.to_owned());
    }
    if low.contains("connection refused") {
        return Some(HINT_OFFLINE.to_owned());
    }
    if low.contains("dns error")
        || low.contains("failed to lookup")
        || low.contains("name resolution")
        || low.contains("no such host")
    {
        return Some(HINT_DNS.to_owned());
    }
    if low.contains("timed out") || low.contains("timeout") {
        return Some(HINT_TIMEOUT.to_owned());
    }
    if low.contains("cancelled") || low.contains("canceled") {
        return Some("cancelled — nothing was changed".to_owned());
    }
    if low.contains("no model selected") || low.contains("model name must not be empty") {
        return Some(HINT_NO_MODEL.to_owned());
    }
    if low.contains("empty reply") || low.contains("returned no content") {
        return Some(HINT_EMPTY.to_owned());
    }
    if low.contains("non-json") || low.contains("not json") || low.contains("expected value") {
        return Some(HINT_NOT_JSON.to_owned());
    }
    match status {
        Some(413) => Some("the request was larger than the provider accepts".to_owned()),
        Some(code) => {
            hint_for_code(code).or_else(|| Some(format!("the provider answered with HTTP {code}")))
        }
        None => None,
    }
}

/// The first HTTP status in the text, when there is one.
///
/// Deliberately narrow: a bare three-digit number is only a status when it sits
/// next to something that means HTTP, so a step number or a latency in the same
/// string cannot be mistaken for a 500.
fn http_status(text: &str) -> Option<u16> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    for (i, tok) in tokens.iter().enumerate() {
        // A token with no number in it is simply not a candidate. `continue`,
        // never an early return: most words in an error message are not numbers,
        // and bailing on the first one would make this return `None` for nearly
        // every input that *does* carry a status.
        let digits: String = tok
            .trim_matches(|c: char| !c.is_ascii_digit())
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let Ok(code) = digits.parse::<u16>() else {
            continue;
        };
        if !(400..600).contains(&code) {
            continue;
        }
        // `HTTP 429`, `returned 429`, `status: 429`, `"code":429` inside a JSON
        // body, `429 Too Many Requests`.
        let before = i.checked_sub(1).and_then(|j| marker_word(tokens[j]));
        let after = tokens.get(i + 1).and_then(|s| marker_word(s));
        let here = marker_word(tok);
        let is_marker = |w: &Option<String>| {
            w.as_deref().is_some_and(|w| {
                matches!(
                    w,
                    "http" | "returned" | "returns" | "status" | "code" | "error"
                )
            })
        };
        let marked = is_marker(&before)
            || is_marker(&here)
            || after
                .as_deref()
                .is_some_and(|word| REASON_PHRASES.contains(&word));
        if marked {
            return Some(code);
        }
    }
    None
}

/// The marker word a token starts with, lowercased, once the punctuation around
/// it is skipped: `"code":429` and `code:429` both carry `code`.
fn marker_word(token: &str) -> Option<String> {
    let rest = token.trim_start_matches(|c: char| !c.is_ascii_alphabetic());
    let word: String = rest
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    if word.is_empty() {
        None
    } else {
        Some(word.to_ascii_lowercase())
    }
}

/// First word of every standard HTTP reason phrase. A status followed by one of
/// these is unambiguous even without a marker word before it.
const REASON_PHRASES: [&str; 15] = [
    "bad",
    "unauthorized",
    "payment",
    "forbidden",
    "not",
    "method",
    "gone",
    "request",
    "conflict",
    "unsupported",
    "unprocessable",
    "too",
    "internal",
    "implemented",
    "gateway",
];

/// The JSON payload embedded in a longer string, if there is exactly one.
///
/// `"model returned 429: {\"error\":{\"message\":\"quota\"}}"` — the whole string
/// is not JSON, but the tail is, and the tail is the part worth reading.
///
/// Balanced rather than first-to-last `{`: a sentence that quotes two payloads
/// (`'{"code":402}' while sending '{"model":"x"}'`) would otherwise parse as
/// neither, and a payload nested inside another is already covered by
/// [`payload_message`] recursing into its own fields.
fn embedded_value(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    let start = text.find(['{', '['])?;
    let open = bytes[start];
    let close = match open {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            match b {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => in_string = false,
                _ => escaped = false,
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b if *b == open => depth += 1,
            b if *b == close => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str::<Value>(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// [`embedded_value`] reduced to the human message it carries.
fn embedded_message(text: &str) -> Option<String> {
    payload_message(&embedded_value(text)?)
}

/// Normalize reader-facing text: one line, no leading markers, no wire noise.
fn clean(text: &str) -> String {
    let mut s = text.trim().to_owned();
    // The UI already prints a failure marker; a message that repeats it reads
    // like two failures.
    for marker in ["✖", "⚠", "error:", "Error:", "ERROR:"] {
        if let Some(rest) = s.strip_prefix(marker) {
            s = rest.trim_start().to_owned();
            break;
        }
    }
    // A provider sentence can be several paragraphs of prose. One line is a line.
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let s = strip_status_prefix(&s);
    if s.chars().count() > MAX_MESSAGE_CHARS {
        let mut t: String = s.chars().take(MAX_MESSAGE_CHARS - 1).collect();
        t.push('…');
        return t;
    }
    s
}

/// `"model-x returned 429 Too Many Requests: quota"` → `"quota"`.
fn strip_status_prefix(s: &str) -> String {
    let Some((head, tail)) = s.split_once(": ") else {
        return s.to_owned();
    };
    let low = head.to_ascii_lowercase();
    let says_status = http_status(head).is_some()
        || low.contains("http")
        || low.contains("status");
    if says_status && !tail.trim().is_empty() {
        return tail.trim().to_owned();
    }
    s.to_owned()
}

/// Whether text still carries the shape of a payload rather than a sentence.
fn reads_as_payload(s: &str) -> bool {
    let t = s.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        return true;
    }
    let low = t.to_ascii_lowercase();
    low.contains("<html") || low.contains("<!doctype")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_provider_json_body_becomes_its_own_message() {
        let raw = r#"{"error":{"message":"You requested 65535 tokens but can only afford 54814","code":402}}"#;
        let out = explain(raw);
        assert!(out.message.contains("can only afford 54814"), "{:?}", out);
        assert_eq!(out.hint.as_deref(), Some(HINT_CREDIT));
        assert!(!out.message.contains('{'), "payload leaked: {}", out.message);
    }

    #[test]
    fn a_status_prefixed_body_keeps_the_body_and_drops_the_prefix() {
        let raw = r#"gemini-web returned 429 Too Many Requests: {"error":{"message":"rate limited"}}"#;
        let out = explain(raw);
        assert_eq!(out.message, "rate limited");
        assert_eq!(out.hint.as_deref(), Some(HINT_RATE));
    }

    #[test]
    fn an_mcp_envelope_is_unwrapped_to_its_text() {
        let value = json!({"content": [{"type": "text", "text": "error: no such element"}]});
        assert_eq!(
            payload_message(&value).as_deref(),
            Some("error: no such element")
        );
        let out = explain(&value.to_string());
        assert!(!out.message.contains("content"), "{:?}", out);
        assert!(out.message.contains("no such element"), "{:?}", out);
    }

    #[test]
    fn a_stringified_result_is_read_one_level_deeper() {
        let value = json!({"content": [{"text": "{\"success\":false,\"message\":\"No action found\"}"}]});
        assert_eq!(
            payload_message(&value).as_deref(),
            Some("No action found")
        );
    }

    #[test]
    fn every_status_class_maps_to_its_own_next_step() {
        // A table, not an anecdote: the rule under test is "each class of HTTP
        // failure gets its own advice", so a single code proves nothing.
        let cases = [
            (401, HINT_KEY),
            (403, HINT_KEY),
            (402, HINT_CREDIT),
            (404, HINT_MODEL),
            (408, HINT_TIMEOUT),
            (429, HINT_RATE),
            (500, HINT_SERVER),
            (503, HINT_SERVER),
        ];
        for (code, hint) in cases {
            let raw = format!("request returned HTTP {code}: something went wrong");
            let out = explain(&raw);
            assert_eq!(out.hint.as_deref(), Some(hint), "HTTP {code}");
            assert!(!out.message.contains(&code.to_string()), "HTTP {code}: {}", out.message);
        }
    }

    #[test]
    fn a_transport_failure_gets_its_remedy() {
        for (raw, hint) in [
            ("error sending request: tcp connect error: Connection refused", HINT_OFFLINE),
            ("dns error: failed to lookup address info", HINT_DNS),
            ("CDP response timeout for method Page.navigate", HINT_TIMEOUT),
            ("OpenAI API request timed out", HINT_TIMEOUT),
        ] {
            assert_eq!(explain(raw).hint.as_deref(), Some(hint), "{raw}");
        }
    }

    /// The general rule for a transport failure: one sentence and one next step,
    /// whichever stack the error came off. A table, because a single
    /// `Connection refused` proves nothing about the other shapes.
    #[test]
    fn a_transport_failure_is_one_readable_line() {
        for raw in [
            "error sending request: tcp connect error: Connection refused (os error 111)",
            "request to http://127.0.0.1:11435/v1/chat/completions failed: connection refused",
            "dns error: failed to lookup address information: Name or service not known",
            "CDP call Page.navigate timed out after 5000ms",
        ] {
            let line = friendly(raw);
            assert!(!line.is_empty(), "{raw}");
            assert!(line.chars().count() <= 500, "{line}");
        }
    }

    #[test]
    fn a_three_digit_number_that_is_not_a_status_is_left_alone() {
        // The same digits appear in step numbers and latencies. Reading one as
        // HTTP 500 would tell the user to retry a run that never failed.
        let out = explain("recovered after 500 ms at step 3/4");
        assert_eq!(out.message, "recovered after 500 ms at step 3/4");
        assert_eq!(out.hint, None);
    }

    #[test]
    fn a_status_phrase_after_the_code_is_enough_to_recognise_it() {
        assert_eq!(http_status("429 Too Many Requests"), Some(429));
        assert_eq!(http_status("401 Unauthorized"), Some(401));
        assert_eq!(http_status("up to 512 tokens"), None);
    }

    #[test]
    fn an_html_error_page_never_reaches_the_reader() {
        let raw = "<!DOCTYPE html><html><body>502 Bad Gateway</body></html>";
        let out = explain(raw);
        assert!(!out.message.contains("<"), "{}", out.message);
        assert_eq!(out.hint.as_deref(), Some(HINT_NOT_JSON));
    }

    #[test]
    fn a_multi_paragraph_body_collapses_to_one_line() {
        let out = explain("rate limited.\n\nRetry after 30s.\nSee https://example.test/docs");
        assert!(!out.message.contains('\n'), "{:?}", out);
    }

    #[test]
    fn a_failure_marker_is_not_repeated_by_the_message() {
        assert_eq!(explain("✖ Step 2/3 failed").message, "Step 2/3 failed");
        assert_eq!(explain("error: browser is not connected").message, "browser is not connected");
    }

    #[test]
    fn a_long_message_is_capped_instead_of_flooding_the_chat() {
        let out = explain(&"x".repeat(4000));
        assert!(out.message.chars().count() <= MAX_MESSAGE_CHARS);
        assert!(out.message.ends_with('…'));
    }

    #[test]
    fn an_empty_failure_still_says_something() {
        assert!(!friendly("   ").is_empty());
        assert!(!friendly("").is_empty());
    }

    #[test]
    fn a_plain_sentence_is_returned_unchanged() {
        let raw = "Step 2/4 failed (browser_click): the element was not found";
        assert_eq!(explain(raw).message, raw);
    }

    #[test]
    fn one_line_joins_the_message_and_the_hint() {
        let out = explain("returned HTTP 401: {\"error\":\"bad key\"}");
        let line = out.one_line();
        assert!(line.contains("bad key"), "{line}");
        assert!(line.contains("Connect providers"), "{line}");
    }
}