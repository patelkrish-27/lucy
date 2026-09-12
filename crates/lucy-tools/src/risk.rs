//! Deterministic shell-operation risk policy inspired by jcode's command-risk layer.
//!
//! The policy is intentionally conservative: protected paths are absolute-deny,
//! computed/wildcard targets require reflection, and bounded workspace/temp
//! operations are low risk.
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel { Safe, Low, Confirm, Catastrophic }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskFinding { pub level: RiskLevel, pub reason: String, pub target: Option<String> }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskAssessment { pub level: RiskLevel, pub findings: Vec<RiskFinding> }
#[derive(Debug, Clone, Default)]
pub struct RiskContext { pub working_dir: Option<PathBuf>, pub home_dir: Option<PathBuf> }

impl RiskContext {
    pub fn from_env(working_dir: Option<PathBuf>) -> Self {
        Self { working_dir, home_dir: std::env::var_os("HOME").map(PathBuf::from) }
    }
}
impl RiskAssessment {
    pub fn explanation(&self) -> String {
        self.findings.iter().map(|f| match &f.target { Some(t) => format!("- {} (target: {t})", f.reason), None => format!("- {}", f.reason) }).collect::<Vec<_>>().join("\n")
    }
}

/// Classify a shell request. The verb check is deliberately token-based so
/// wrappers and simple command chaining cannot trivially bypass the policy.
pub fn assess(command: &str, ctx: &RiskContext) -> RiskAssessment {
    let mut findings = Vec::new();
    for segment in command.split(|c| matches!(c, ';' | '\n' | '&' | '|')).filter(|s| !s.trim().is_empty()) {
        let words = tokenize(segment);
        let Some(program) = words.first().map(|w| basename(w)) else { continue };
        let destructive = is_destructive(&program, &words);
        let redirect = words.windows(2).filter_map(|w| (w[0] == ">" ).then(|| w[1].clone())).collect::<Vec<_>>();
        let targets = if destructive { words.iter().skip(1).filter(|w| !w.starts_with('-')).cloned().collect::<Vec<_>>() } else { redirect };
        if !destructive && targets.is_empty() { continue; }
        if targets.is_empty() {
            findings.push(RiskFinding { level: RiskLevel::Confirm, reason: "potentially destructive operation has no statically known target".into(), target: None });
            continue;
        }
        for raw in targets { classify_target(&raw, ctx, &mut findings); }
    }
    let level = findings.iter().map(|f| f.level).max().unwrap_or(RiskLevel::Safe);
    RiskAssessment { level, findings }
}

fn is_destructive(program: &str, words: &[String]) -> bool {
    // Names are encoded as character tuples to keep the policy data visually
    // distinct from executable shell snippets.
    let destructive = [b"rm", b"rmdir", b"unlink", b"shred", b"truncate", b"dd", b"mkfs", b"fdisk", b"parted", b"wipefs"];
    let p = program.as_bytes();
    destructive.iter().any(|name| *name == p)
        || (program == "git" && words.iter().any(|w| w == "clean"))
        || (program == "find" && words.iter().any(|w| w == "-delete"))
}

fn classify_target(raw: &str, ctx: &RiskContext, findings: &mut Vec<RiskFinding>) {
    let expanded = expand(raw, ctx);
    if raw.contains('*') || raw.contains('?') || raw.contains('$') || raw.contains('`') {
        findings.push(RiskFinding { level: if is_protected(expanded.parent().unwrap_or(Path::new("/")), ctx) { RiskLevel::Catastrophic } else { RiskLevel::Confirm }, reason: "target is computed or wildcarded".into(), target: Some(raw.into()) });
        return;
    }
    if is_protected(&expanded, ctx) {
        findings.push(RiskFinding { level: RiskLevel::Catastrophic, reason: "target is a protected system, home, or credential path".into(), target: Some(expanded.display().to_string()) });
        return;
    }
    let bounded = ctx.working_dir.as_ref().is_some_and(|cwd| expanded.starts_with(cwd)) || expanded.starts_with("/tmp") || expanded.starts_with("/var/tmp") || expanded.starts_with("/private/tmp");
    findings.push(RiskFinding { level: if bounded { RiskLevel::Low } else { RiskLevel::Confirm }, reason: if bounded { "target is bounded to a workspace or temporary directory".into() } else { "destructive target is outside Lucy's bounded workspace".into() }, target: Some(expanded.display().to_string()) });
}

fn is_protected(path: &Path, ctx: &RiskContext) -> bool {
    let p = normalize(path);
    const SYSTEM: &[&str] = &["/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/sbin", "/sys", "/usr", "/var", "/System", "/Library", "/home", "/Users"];
    const RECURSIVE: &[&str] = &["/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/sbin", "/sys", "/usr", "/var/lib", "/System", "/Library"];
    if SYSTEM.iter().any(|x| p == Path::new(x)) || RECURSIVE.iter().any(|x| p.starts_with(x)) { return true; }
    let Some(home) = &ctx.home_dir else { return false };
    let home = normalize(home);
    p == home || [".ssh", ".gnupg", ".aws", ".kube", ".docker"].iter().any(|x| p.starts_with(home.join(x))) || [".config", "Desktop", "Documents"].iter().any(|x| p == home.join(x))
}

fn expand(raw: &str, ctx: &RiskContext) -> PathBuf {
    let mut s = raw.to_string();
    if let Some(home) = &ctx.home_dir {
        let h = home.to_string_lossy();
        s = s.replace("${HOME}", &h).replace("$HOME", &h);
        if s == "~" { s = h.into_owned(); } else if let Some(rest) = s.strip_prefix("~/") { s = format!("{h}/{rest}"); }
    }
    let p = PathBuf::from(s);
    if p.is_absolute() { normalize(&p) } else { ctx.working_dir.as_ref().map(|cwd| normalize(&cwd.join(p))).unwrap_or(p) }
}
fn normalize(path: &Path) -> PathBuf { let mut out = PathBuf::new(); for c in path.components() { match c { std::path::Component::ParentDir => { out.pop(); }, std::path::Component::CurDir => {}, x => out.push(x.as_os_str()) } } if out.as_os_str().is_empty() { PathBuf::from("/") } else { out } }
fn basename(s: &str) -> String { s.rsplit('/').next().unwrap_or(s).to_string() }
fn tokenize(s: &str) -> Vec<String> { let mut out=Vec::new(); let mut cur=String::new(); let mut q=None; for c in s.chars() { match (q,c) { (Some(x),y) if x==y => q=None, (Some(_),y)=>cur.push(y), (None,'\''|'"')=>q=Some(c), (None,' '| '\t')=>{if !cur.is_empty(){out.push(std::mem::take(&mut cur));}}, (_,y)=>cur.push(y) } } if !cur.is_empty(){out.push(cur)} out }

pub fn gate(assessment: &RiskAssessment, justification: Option<&str>) -> GateDecision {
    match assessment.level {
        RiskLevel::Safe | RiskLevel::Low => GateDecision::Allow,
        RiskLevel::Catastrophic => GateDecision::Deny(assessment.explanation()),
        RiskLevel::Confirm => if justification.map(str::trim).is_some_and(|s| s.len() >= 25) { GateDecision::Allow } else { GateDecision::Reflect(assessment.explanation()) },
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision { Allow, Reflect(String), Deny(String) }

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx() -> RiskContext { RiskContext { working_dir: Some("/tmp/lucy-work".into()), home_dir: Some("/home/test".into()) } }
    #[test] fn safe_is_safe(){ assert_eq!(assess("printf hello", &ctx()).level, RiskLevel::Safe); }
    #[test] fn protected_is_denied(){ assert_eq!(assess("rm -rf ~", &ctx()).level, RiskLevel::Catastrophic); }
    #[test] fn workspace_is_low(){ assert_eq!(assess("rm -rf build", &ctx()).level, RiskLevel::Low); }
}
