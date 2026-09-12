//! Deterministic shell-command risk classification.
//!
//! This module is adapted from the safety ideas in jcode's `jcode-command-risk`:
//! classify by blast radius, escalate unknown targets, and absolutely deny
//! destructive access to core system/credential paths. Lucy keeps the policy
//! intentionally small and auditable while the tool runtime grows around it.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Safe,
    Low,
    Confirm,
    Catastrophic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskFinding {
    pub level: RiskLevel,
    pub reason: String,
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub findings: Vec<RiskFinding>,
}

#[derive(Debug, Clone, Default)]
pub struct RiskContext {
    pub working_dir: Option<PathBuf>,
    pub home_dir: Option<PathBuf>,
}

impl RiskContext {
    pub fn from_env(working_dir: Option<PathBuf>) -> Self {
        Self {
            working_dir,
            home_dir: std::env::var_os("HOME").map(PathBuf::from),
        }
    }
}

impl RiskAssessment {
    pub fn explanation(&self) -> String {
        self.findings
            .iter()
            .map(|f| match &f.target {
                Some(target) => format!("- {} (target: {target})", f.reason),
                None => format!("- {}", f.reason),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn assess(command: &str, ctx: &RiskContext) -> RiskAssessment {
    let mut findings = Vec::new();

    for segment in split_segments(command) {
        assess_segment(segment, ctx, &mut findings);
    }

    let level = findings.iter().map(|f| f.level).max().unwrap_or(RiskLevel::Safe);
    RiskAssessment { level, findings }
}

fn assess_segment(segment: &str, ctx: &RiskContext, findings: &mut Vec<RiskFinding>) {
    let words = tokenize(segment);
    if words.is_empty() {
        return;
    }

    let mut start = 0;
    while start < words.len() && is_wrapper(&words[start]) {
        start += 1;
        while start < words.len() && words[start].starts_with('-') {
            start += 1;
            if start < words.len() && !words[start].starts_with('-') && is_wrapper_value(&words[start - 1]) {
                start += 1;
            }
        }
    }
    if start >= words.len() {
        findings.push(RiskFinding {
            level: RiskLevel::Confirm,
            reason: "a command wrapper hides the actual command".into(),
            target: None,
        });
        return;
    }

    let program = basename(&words[start]);
    if matches!(program.as_str(), "sh" | "bash" | "zsh" | "dash" | "fish") {
        if let Some(script) = words.iter().skip(start + 1).find(|w| *w != "-c" && !w.starts_with('-')) {
            for nested in split_segments(script) {
                assess_segment(nested, ctx, findings);
            }
        } else {
            findings.push(RiskFinding {
                level: RiskLevel::Confirm,
                reason: "the shell script body could not be inspected safely".into(),
                target: None,
            });
        }
        return;
    }

    let destructive = matches!(
        program.as_str(),
        "rm" | "rmdir" | "unlink" | "shred" | "truncate" | "dd" | "mkfs" | "fdisk" | "parted" | "wipefs"
    ) || (program == "git" && words.iter().any(|w| w == "clean"))
        || (program == "find" && words.iter().any(|w| w == "-delete"));

    let redirects = parse_redirect_targets(&words);
    if !destructive && redirects.is_empty() {
        return;
    }

    let mut targets = if destructive {
        words.iter().skip(start + 1).filter(|w| !w.starts_with('-')).cloned().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    targets.extend(redirects);

    if targets.is_empty() {
        findings.push(RiskFinding {
            level: RiskLevel::Confirm,
            reason: format!("`{program}` is potentially destructive but its target is unknown"),
            target: None,
        });
        return;
    }

    let recursive = words.iter().any(|w| w == "-r" || w == "-R" || w.contains("-rf") || w.contains("--recursive"));
    for raw in targets {
        classify_target(&raw, recursive, ctx, findings);
    }
}

fn classify_target(raw: &str, recursive: bool, ctx: &RiskContext, findings: &mut Vec<RiskFinding>) {
    let expanded = expand(raw, ctx);

    if raw.contains('*') || raw.contains('?') || raw.contains('$') || raw.contains('`') {
        let parent = expanded.parent().unwrap_or_else(|| Path::new("/"));
        if is_protected(parent, ctx) || raw.contains("$HOME") || raw.starts_with("~/") {
            findings.push(RiskFinding {
                level: RiskLevel::Catastrophic,
                reason: "target may expand into a protected system or credential path".into(),
                target: Some(raw.into()),
            });
        } else {
            findings.push(RiskFinding {
                level: RiskLevel::Confirm,
                reason: "target is computed or wildcarded, so its blast radius is uncertain".into(),
                target: Some(raw.into()),
            });
        }
        return;
    }

    if is_protected(&expanded, ctx) {
        findings.push(RiskFinding {
            level: RiskLevel::Catastrophic,
            reason: "targets a protected system, home, or credential path".into(),
            target: Some(expanded.display().to_string()),
        });
        return;
    }

    let low = ctx
        .working_dir
        .as_ref()
        .is_some_and(|cwd| expanded.starts_with(cwd))
        || expanded.starts_with("/tmp")
        || expanded.starts_with("/var/tmp")
        || expanded.starts_with("/private/tmp");

    findings.push(RiskFinding {
        level: if low { RiskLevel::Low } else { RiskLevel::Confirm },
        reason: if recursive {
            "recursive destructive operation".into()
        } else {
            "destructive operation on a concrete path".into()
        },
        target: Some(expanded.display().to_string()),
    });
}

fn is_protected(path: &Path, ctx: &RiskContext) -> bool {
    let p = normalize(path);
    let system_exact = [
        "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/sbin", "/sys", "/usr", "/var", "/System", "/Library", "/home", "/Users",
    ];
    let system_recursive = [
        "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/sbin", "/sys", "/usr", "/var/lib", "/System", "/Library",
    ];
    if system_exact.iter().any(|x| p == Path::new(x)) || system_recursive.iter().any(|x| p.starts_with(x)) {
        return true;
    }
    let Some(home) = &ctx.home_dir else { return false; };
    let home = normalize(home);
    if p == home {
        return true;
    }
    [".ssh", ".gnupg", ".aws", ".kube", ".docker"]
        .iter()
        .any(|x| p.starts_with(home.join(x)))
        || [".config", "Desktop", "Documents"]
            .iter()
            .any(|x| p == home.join(x))
}

fn expand(raw: &str, ctx: &RiskContext) -> PathBuf {
    let mut value = raw.to_string();
    if let Some(home) = &ctx.home_dir {
        let h = home.to_string_lossy();
        value = value.replace("${HOME}", &h).replace("$HOME", &h);
        if value == "~" { value = h.into_owned(); }
        else if let Some(rest) = value.strip_prefix("~/") { value = format!("{h}/{rest}"); }
    }
    let p = PathBuf::from(value);
    if p.is_absolute() { normalize(&p) }
    else { ctx.working_dir.as_ref().map(|cwd| normalize(&cwd.join(p))).unwrap_or(p) }
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => { out.pop(); }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() { PathBuf::from("/") } else { out }
}

fn basename(s: &str) -> String { s.rsplit('/').next().unwrap_or(s).to_string() }

fn is_wrapper(s: &str) -> bool {
    matches!(s, "sudo" | "doas" | "env" | "nice" | "ionice" | "timeout" | "nohup" | "xargs" | "exec" | "command")
}

fn is_wrapper_value(flag: &str) -> bool {
    matches!(flag, "-u" | "--user" | "-g" | "--group" | "-n" | "--adjustment" | "-s" | "--signal")
}

fn parse_redirect_targets(words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, word) in words.iter().enumerate() {
        if word == ">" && i + 1 < words.len() { out.push(words[i + 1].clone()); }
        else if let Some(target) = word.strip_prefix('>') { if !target.is_empty() { out.push(target.to_string()); } }
    }
    out
}

fn split_segments(command: &str) -> Vec<&str> {
    command.split(|c| c == '\n' || c == ';' || c == '|' || c == '&').filter(|s| !s.trim().is_empty()).collect()
}

fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote = None;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '\'' | '"') => quote = Some(ch),
            (None, ' ' | '\t' | '\n') => {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            }
            (None, '>') => {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                out.push(">".into());
            }
            (None, c) => cur.push(c),
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

pub fn gate(assessment: &RiskAssessment, justification: Option<&str>) -> GateDecision {
    match assessment.level {
        RiskLevel::Safe | RiskLevel::Low => GateDecision::Allow,
        RiskLevel::Catastrophic => GateDecision::Deny(assessment.explanation()),
        RiskLevel::Confirm => {
            let substantive = justification.map(str::trim).is_some_and(|s| s.len() >= 25 && !matches!(s.to_ascii_lowercase().as_str(), "yes" | "ok" | "sure" | "do it"));
            if substantive { GateDecision::Allow } else { GateDecision::Reflect(assessment.explanation()) }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision { Allow, Reflect(String), Deny(String) }

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RiskContext { RiskContext::from_env(Some(PathBuf::from("/tmp/lucy-work"))) }

    #[test]
    fn safe_command_is_safe() { assert_eq!(assess("ls -la", &ctx()).level, RiskLevel::Safe); }

    #[test]
    fn home_delete_is_catastrophic() { assert_eq!(assess("rm -rf ~", &ctx()).level, RiskLevel::Catastrophic); }

    #[test]
    fn credential_delete_is_catastrophic() { assert_eq!(assess("rm -f ~/.ssh/id_rsa", &ctx()).level, RiskLevel::Catastrophic); }

    #[test]
    fn workspace_delete_is_low() { assert_eq!(assess("rm -rf build", &ctx()).level, RiskLevel::Low); }

    #[test]
    fn unknown_glob_requires_confirmation() { assert_eq!(assess("rm -rf /data/*", &ctx()).level, RiskLevel::Confirm); }
}
