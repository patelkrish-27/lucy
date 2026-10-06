use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Question types supported by JEV and Laya System-1 engines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

/// A typed decision question presented to the System-1 engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub question_type: QuestionType,
    pub instructions: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
}

impl Question {
    /// Create a choice question with a map of label -> description/criteria.
    pub fn choice<I: Into<Value>>(instructions: I, criteria: HashMap<String, Value>) -> Self {
        Self {
            question_type: QuestionType::Choice,
            instructions: instructions.into(),
            criteria: Some(serde_json::to_value(criteria).unwrap_or(Value::Null)),
        }
    }

    /// Create a choice question with simple string criteria.
    pub fn choice_simple<I: Into<Value>>(instructions: I, criteria: &[(&str, &str)]) -> Self {
        let map: HashMap<String, Value> = criteria
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect();
        Self::choice(instructions, map)
    }

    /// Create a noul (boolean calibrated probability 0.0..1.0) question.
    pub fn noul<I: Into<Value>>(instructions: I) -> Self {
        Self {
            question_type: QuestionType::Noul,
            instructions: instructions.into(),
            criteria: None,
        }
    }

    /// Create an ordinal score question with ordered scale levels.
    pub fn score<I: Into<Value>>(instructions: I, levels: &[&str]) -> Self {
        let levs: Vec<Value> = levels
            .iter()
            .map(|l| Value::String(l.to_string()))
            .collect();
        Self {
            question_type: QuestionType::Score,
            instructions: instructions.into(),
            criteria: Some(Value::Array(levs)),
        }
    }
}

/// Calibrated choice answer returned by JEV/Laya.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub probabilities: HashMap<String, f64>,
}

/// Calibrated noul (calibrated boolean probability) answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulAnswer {
    pub noul: f64,
    #[serde(default)]
    pub confidence: f64,
}

/// Calibrated score answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreAnswer {
    pub score: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub probabilities: Option<HashMap<String, f64>>,
}

/// Generic answer matching choice, noul, or score.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Answer {
    Choice(ChoiceAnswer),
    Noul(NoulAnswer),
    Score(ScoreAnswer),
    Generic(Value),
}

impl Answer {
    /// Return the winning choice string if this is a choice answer.
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Self::Choice(c) => Some(&c.choice),
            Self::Generic(v) => v.get("choice").and_then(Value::as_str),
            _ => None,
        }
    }

    /// Return the confidence score (0.0 .. 1.0).
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Choice(c) => c.confidence,
            Self::Noul(n) => n.confidence,
            Self::Score(s) => s.confidence,
            Self::Generic(v) => v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
        }
    }

    /// Return the noul probability (0.0 .. 1.0) if applicable.
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Self::Noul(n) => Some(n.noul),
            Self::Generic(v) => v.get("noul").and_then(Value::as_f64),
            _ => None,
        }
    }
}

/// Request payload for JEV / Laya predict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    pub state: Value,
    pub questions: HashMap<String, Question>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Routing metadata returned by Laya checkpoint router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMeta {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Full response payload from JEV / Laya System-1 inference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredictResponse {
    #[serde(default)]
    pub answers: HashMap<String, Answer>,
    #[serde(default)]
    pub routing: Option<RoutingMeta>,
    #[serde(default)]
    pub latency_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_serialization() {
        let mut criteria = HashMap::new();
        criteria.insert(
            "chat".to_string(),
            Value::String("Conversation".to_string()),
        );
        criteria.insert(
            "act".to_string(),
            Value::String("Computer action".to_string()),
        );
        let q = Question::choice("Choose mode", criteria);
        let val = serde_json::to_value(&q).unwrap();
        assert_eq!(val["type"], "choice");
        assert_eq!(val["instructions"], "Choose mode");
        assert!(val["criteria"]["chat"].is_string());
    }

    #[test]
    fn test_choice_answer_deserialization() {
        let json = r#"{
            "choice": "act",
            "confidence": 0.94,
            "probabilities": {"chat": 0.06, "act": 0.94}
        }"#;
        let ans: Answer = serde_json::from_str(json).unwrap();
        assert_eq!(ans.as_choice(), Some("act"));
        assert!((ans.confidence() - 0.94).abs() < 1e-6);
    }
}
