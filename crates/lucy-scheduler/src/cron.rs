//! Cron expression parser and matcher.
//!
//! Supports standard cron syntax (`*`, `*/5`, `1-5`, `1,3,5`, `0 8 * * 1-5`)
//! and named schedules (`@hourly`, `@daily`, `@weekly`, `@monthly`, `@every 30m`).

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// A single cron field (minute, hour, day-of-month, month, day-of-week).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CronField {
    /// `*` — matches any value.
    Any,
    /// A single specific value, e.g. `5`.
    Specific(u32),
    /// A range, e.g. `1-5`.
    Range(u32, u32),
    /// A list of values, e.g. `1,3,5`.
    List(Vec<u32>),
    /// A step, e.g. `*/5` → Step(5, 59) for minutes.
    Step(u32, u32),
}

impl CronField {
    /// Parse a single cron field string.
    pub fn parse(s: &str, min: u32, max: u32) -> Result<Self, CronError> {
        let s = s.trim();
        if s == "*" {
            return Ok(CronField::Any);
        }

        // Step: */N or a-b/N
        if let Some(step_str) = s.strip_prefix("*/") {
            let step: u32 = step_str
                .parse()
                .map_err(|_| CronError::InvalidStep(step_str.to_string()))?;
            if step == 0 || step > max {
                return Err(CronError::InvalidStep(step_str.to_string()));
            }
            return Ok(CronField::Step(step, max));
        }

        // Range with step: a-b/N
        if let Some((range_part, step_part)) = s.split_once('/') {
            let step: u32 = step_part
                .parse()
                .map_err(|_| CronError::InvalidStep(step_part.to_string()))?;
            if step == 0 {
                return Err(CronError::InvalidStep(step_part.to_string()));
            }
            if let Some((lo, hi)) = range_part.split_once('-') {
                let lo: u32 = lo
                    .parse()
                    .map_err(|_| CronError::InvalidRange(range_part.to_string()))?;
                let hi: u32 = hi
                    .parse()
                    .map_err(|_| CronError::InvalidRange(range_part.to_string()))?;
                if lo < min || hi > max || lo > hi {
                    return Err(CronError::InvalidRange(range_part.to_string()));
                }
                return Ok(CronField::Step(step, hi));
            }
            // Single value with step: N/M — treat as N-max with step
            let lo: u32 = range_part
                .parse()
                .map_err(|_| CronError::InvalidValue(range_part.to_string()))?;
            if lo < min || lo > max {
                return Err(CronError::OutOfRange(range_part.to_string(), min, max));
            }
            return Ok(CronField::Step(step, max));
        }

        // Range: a-b
        if let Some((lo, hi)) = s.split_once('-') {
            let lo: u32 = lo
                .parse()
                .map_err(|_| CronError::InvalidRange(s.to_string()))?;
            let hi: u32 = hi
                .parse()
                .map_err(|_| CronError::InvalidRange(s.to_string()))?;
            if lo < min || hi > max || lo > hi {
                return Err(CronError::InvalidRange(s.to_string()));
            }
            return Ok(CronField::Range(lo, hi));
        }

        // List: a,b,c
        if s.contains(',') {
            let mut values = Vec::new();
            for part in s.split(',') {
                let v: u32 = part
                    .trim()
                    .parse()
                    .map_err(|_| CronError::InvalidValue(part.trim().to_string()))?;
                if v < min || v > max {
                    return Err(CronError::InvalidValue(part.trim().to_string()));
                }
                values.push(v);
            }
            return Ok(CronField::List(values));
        }

        // Single value
        let v: u32 = s
            .parse()
            .map_err(|_| CronError::InvalidValue(s.to_string()))?;
        if v < min || v > max {
            return Err(CronError::OutOfRange(s.to_string(), min, max));
        }
        Ok(CronField::Specific(v))
    }

    /// Check if a value matches this field.
    pub fn matches(&self, value: u32) -> bool {
        match self {
            CronField::Any => true,
            CronField::Specific(v) => value == *v,
            CronField::Range(lo, hi) => value >= *lo && value <= *hi,
            CronField::List(values) => values.contains(&value),
            CronField::Step(step, max) => value.is_multiple_of(*step) && value <= *max,
        }
    }

    /// The minimum value this field can match (for next-run computation).
    pub fn min_value(&self) -> u32 {
        match self {
            CronField::Any => 0,
            CronField::Specific(v) => *v,
            CronField::Range(lo, _) => *lo,
            CronField::List(values) => values.iter().copied().min().unwrap_or(0),
            CronField::Step(_, _) => 0,
        }
    }

    /// The maximum value this field can match (for next-run computation).
    pub fn max_value(&self, absolute_max: u32) -> u32 {
        match self {
            CronField::Any => absolute_max,
            CronField::Specific(v) => *v,
            CronField::Range(_, hi) => *hi,
            CronField::List(values) => values.iter().copied().max().unwrap_or(absolute_max),
            CronField::Step(_, max) => *max,
        }
    }
}

impl fmt::Display for CronField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CronField::Any => write!(f, "*"),
            CronField::Specific(v) => write!(f, "{v}"),
            CronField::Range(lo, hi) => write!(f, "{lo}-{hi}"),
            CronField::List(values) => {
                let parts: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                write!(f, "{}", parts.join(","))
            }
            CronField::Step(step, _) => write!(f, "*/{step}"),
        }
    }
}

/// A full cron expression with five fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronExpression {
    pub minute: CronField,
    pub hour: CronField,
    pub day_of_month: CronField,
    pub month: CronField,
    pub day_of_week: CronField,
}

impl CronExpression {
    /// Parse a standard 5-field cron expression.
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let expr = expr.trim();

        // Named schedules
        if let Some(result) = Self::parse_named(expr) {
            return result;
        }

        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(CronError::WrongFieldCount(parts.len()));
        }

        Ok(CronExpression {
            minute: CronField::parse(parts[0], 0, 59)?,
            hour: CronField::parse(parts[1], 0, 23)?,
            day_of_month: CronField::parse(parts[2], 1, 31)?,
            month: CronField::parse(parts[3], 1, 12)?,
            day_of_week: CronField::parse(parts[4], 0, 7)?,
        })
    }

    fn parse_named(expr: &str) -> Option<Result<Self, CronError>> {
        let lower = expr.to_lowercase();
        match lower.as_str() {
            "@yearly" | "@annually" => Some(Ok(CronExpression {
                minute: CronField::Specific(0),
                hour: CronField::Specific(0),
                day_of_month: CronField::Specific(1),
                month: CronField::Specific(1),
                day_of_week: CronField::Any,
            })),
            "@monthly" => Some(Ok(CronExpression {
                minute: CronField::Specific(0),
                hour: CronField::Specific(0),
                day_of_month: CronField::Specific(1),
                month: CronField::Any,
                day_of_week: CronField::Any,
            })),
            "@weekly" => Some(Ok(CronExpression {
                minute: CronField::Specific(0),
                hour: CronField::Specific(0),
                day_of_month: CronField::Any,
                month: CronField::Any,
                day_of_week: CronField::Specific(0),
            })),
            "@daily" | "@midnight" => Some(Ok(CronExpression {
                minute: CronField::Specific(0),
                hour: CronField::Specific(0),
                day_of_month: CronField::Any,
                month: CronField::Any,
                day_of_week: CronField::Any,
            })),
            "@hourly" => Some(Ok(CronExpression {
                minute: CronField::Specific(0),
                hour: CronField::Any,
                day_of_month: CronField::Any,
                month: CronField::Any,
                day_of_week: CronField::Any,
            })),
            _ => {
                // @every 30m, @every 2h, @every 1d
                if let Some(rest) = lower.strip_prefix("@every ") {
                    Some(Self::parse_every(rest))
                } else if lower.starts_with('@') {
                    Some(Err(CronError::InvalidNamed(expr.to_string())))
                } else {
                    None
                }
            }
        }
    }

    fn parse_every(s: &str) -> Result<CronExpression, CronError> {
        let s = s.trim();
        let (num_str, unit) = s
            .split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
        let num: u32 = num_str
            .trim()
            .parse()
            .map_err(|_| CronError::InvalidEvery(s.to_string()))?;
        if num == 0 {
            return Err(CronError::InvalidEvery(s.to_string()));
        }

        let unit = unit.trim();
        match unit {
            "m" | "min" | "mins" | "minute" | "minutes" => {
                if num >= 60 {
                    // Every N minutes where N >= 60: express in whole hours
                    let hours = num / 60;
                    if hours >= 24 {
                        if hours.is_multiple_of(24) {
                            let days = hours / 24;
                            Ok(CronExpression {
                                minute: CronField::Specific(0),
                                hour: CronField::Specific(0),
                                day_of_month: CronField::Step(days, 31),
                                month: CronField::Any,
                                day_of_week: CronField::Any,
                            })
                        } else {
                            Err(CronError::InvalidEvery(s.to_string()))
                        }
                    } else {
                        Ok(CronExpression {
                            minute: CronField::Specific(0),
                            hour: CronField::Step(hours, 23),
                            day_of_month: CronField::Any,
                            month: CronField::Any,
                            day_of_week: CronField::Any,
                        })
                    }
                } else {
                    Ok(CronExpression {
                        minute: CronField::Step(num, 59),
                        hour: CronField::Any,
                        day_of_month: CronField::Any,
                        month: CronField::Any,
                        day_of_week: CronField::Any,
                    })
                }
            }
            "h" | "hr" | "hrs" | "hour" | "hours" => {
                if num >= 24 {
                    if num.is_multiple_of(24) {
                        let days = num / 24;
                        Ok(CronExpression {
                            minute: CronField::Specific(0),
                            hour: CronField::Specific(0),
                            day_of_month: CronField::Step(days, 31),
                            month: CronField::Any,
                            day_of_week: CronField::Any,
                        })
                    } else {
                        Err(CronError::InvalidEvery(s.to_string()))
                    }
                } else {
                    Ok(CronExpression {
                        minute: CronField::Specific(0),
                        hour: CronField::Step(num, 23),
                        day_of_month: CronField::Any,
                        month: CronField::Any,
                        day_of_week: CronField::Any,
                    })
                }
            }
            "d" | "day" | "days" => {
                if num >= 31 {
                    Err(CronError::InvalidEvery(s.to_string()))
                } else {
                    Ok(CronExpression {
                        minute: CronField::Specific(0),
                        hour: CronField::Specific(0),
                        day_of_month: CronField::Step(num, 31),
                        month: CronField::Any,
                        day_of_week: CronField::Any,
                    })
                }
            }
            _ => Err(CronError::InvalidEvery(s.to_string())),
        }
    }

    /// Check if a given UTC datetime matches this cron expression.
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        let minute = dt.minute();
        let hour = dt.hour();
        let day_of_month = dt.day();
        let month = dt.month();
        // chrono: Monday=1 .. Sunday=7; cron: Sunday=0, Monday=1 .. Saturday=6
        let weekday = dt.weekday().num_days_from_sunday();

        self.minute.matches(minute)
            && self.hour.matches(hour)
            && self.day_of_month.matches(day_of_month)
            && self.month.matches(month)
            && self.day_of_week.matches(weekday)
    }

    /// Compute the next fire time after `from`, up to 5 years in the future.
    pub fn next_after(&self, from: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        // Start from the next minute
        let mut candidate = from
            .checked_add_signed(chrono::Duration::minutes(1))?
            .with_second(0)?
            .with_nanosecond(0)?;

        // Search up to 5 years ahead
        let limit = from.checked_add_signed(chrono::Duration::days(365 * 5))?;

        while candidate <= limit {
            if self.matches(&candidate) {
                return Some(candidate);
            }
            candidate = candidate.checked_add_signed(chrono::Duration::minutes(1))?;
        }
        None
    }
}

impl fmt::Display for CronExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} {} {}",
            self.minute, self.hour, self.day_of_month, self.month, self.day_of_week
        )
    }
}

/// Errors that can occur when parsing or using cron expressions.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CronError {
    #[error("wrong number of fields: expected 5, got {0}")]
    WrongFieldCount(usize),
    #[error("invalid value: {0}")]
    InvalidValue(String),
    #[error("invalid range: {0}")]
    InvalidRange(String),
    #[error("invalid step: {0}")]
    InvalidStep(String),
    #[error("invalid named schedule: {0}")]
    InvalidNamed(String),
    #[error("invalid @every expression: {0}")]
    InvalidEvery(String),
    #[error("value {0} out of range [{1}, {2}]")]
    OutOfRange(String, u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parse_any() {
        let f = CronField::parse("*", 0, 59).unwrap();
        assert_eq!(f, CronField::Any);
        assert!(f.matches(0));
        assert!(f.matches(30));
        assert!(f.matches(59));
    }

    #[test]
    fn parse_specific() {
        let f = CronField::parse("5", 0, 59).unwrap();
        assert_eq!(f, CronField::Specific(5));
        assert!(!f.matches(4));
        assert!(f.matches(5));
        assert!(!f.matches(6));
    }

    #[test]
    fn parse_range() {
        let f = CronField::parse("1-5", 0, 59).unwrap();
        assert_eq!(f, CronField::Range(1, 5));
        assert!(!f.matches(0));
        assert!(f.matches(1));
        assert!(f.matches(3));
        assert!(f.matches(5));
        assert!(!f.matches(6));
    }

    #[test]
    fn parse_list() {
        let f = CronField::parse("1,3,5", 0, 59).unwrap();
        assert_eq!(f, CronField::List(vec![1, 3, 5]));
        assert!(!f.matches(0));
        assert!(f.matches(1));
        assert!(!f.matches(2));
        assert!(f.matches(3));
        assert!(f.matches(5));
        assert!(!f.matches(6));
    }

    #[test]
    fn parse_step() {
        let f = CronField::parse("*/15", 0, 59).unwrap();
        assert_eq!(f, CronField::Step(15, 59));
        assert!(f.matches(0));
        assert!(!f.matches(1));
        assert!(f.matches(15));
        assert!(f.matches(30));
        assert!(f.matches(45));
        assert!(!f.matches(59));
    }

    #[test]
    fn parse_full_expression() {
        let cron = CronExpression::parse("0 8 * * 1-5").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(8));
        assert_eq!(cron.day_of_month, CronField::Any);
        assert_eq!(cron.month, CronField::Any);
        assert_eq!(cron.day_of_week, CronField::Range(1, 5));
    }

    #[test]
    fn parse_named_hourly() {
        let cron = CronExpression::parse("@hourly").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Any);
    }

    #[test]
    fn parse_named_daily() {
        let cron = CronExpression::parse("@daily").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(0));
    }

    #[test]
    fn parse_named_weekly() {
        let cron = CronExpression::parse("@weekly").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(0));
        assert_eq!(cron.day_of_week, CronField::Specific(0));
    }

    #[test]
    fn parse_named_monthly() {
        let cron = CronExpression::parse("@monthly").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(0));
        assert_eq!(cron.day_of_month, CronField::Specific(1));
    }

    #[test]
    fn parse_every_30m() {
        let cron = CronExpression::parse("@every 30m").unwrap();
        assert_eq!(cron.minute, CronField::Step(30, 59));
        assert_eq!(cron.hour, CronField::Any);
    }

    #[test]
    fn parse_every_2h() {
        let cron = CronExpression::parse("@every 2h").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Step(2, 23));
    }

    #[test]
    fn parse_every_1d() {
        let cron = CronExpression::parse("@every 1d").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(0));
        assert_eq!(cron.day_of_month, CronField::Step(1, 31));
    }

    #[test]
    fn parse_every_90m_converts_to_hours() {
        let cron = CronExpression::parse("@every 90m").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Step(1, 23));
    }

    #[test]
    fn parse_every_48h_converts_to_days() {
        let cron = CronExpression::parse("@every 48h").unwrap();
        assert_eq!(cron.minute, CronField::Specific(0));
        assert_eq!(cron.hour, CronField::Specific(0));
        assert_eq!(cron.day_of_month, CronField::Step(2, 31));
    }

    #[test]
    fn invalid_field_count() {
        assert!(matches!(
            CronExpression::parse("* * *"),
            Err(CronError::WrongFieldCount(3))
        ));
    }

    #[test]
    fn invalid_value() {
        assert!(matches!(
            CronField::parse("abc", 0, 59),
            Err(CronError::InvalidValue(_))
        ));
    }

    #[test]
    fn out_of_range() {
        assert!(matches!(
            CronField::parse("60", 0, 59),
            Err(CronError::OutOfRange(_, 0, 59))
        ));
    }

    #[test]
    fn invalid_range_reversed() {
        assert!(matches!(
            CronField::parse("5-1", 0, 59),
            Err(CronError::InvalidRange(_))
        ));
    }

    #[test]
    fn invalid_step_zero() {
        assert!(matches!(
            CronField::parse("*/0", 0, 59),
            Err(CronError::InvalidStep(_))
        ));
    }

    #[test]
    fn invalid_named() {
        assert!(matches!(
            CronExpression::parse("@fortnightly"),
            Err(CronError::InvalidNamed(_))
        ));
    }

    #[test]
    fn invalid_every_zero() {
        assert!(matches!(
            CronExpression::parse("@every 0m"),
            Err(CronError::InvalidEvery(_))
        ));
    }

    #[test]
    fn matches_specific_datetime() {
        let cron = CronExpression::parse("0 8 * * 1-5").unwrap();
        // Monday 2025-01-06 08:00 UTC
        let dt = Utc.with_ymd_and_hms(2025, 1, 6, 8, 0, 0).unwrap();
        assert!(cron.matches(&dt));
        // Monday 2025-01-06 09:00 UTC — wrong hour
        let dt2 = Utc.with_ymd_and_hms(2025, 1, 6, 9, 0, 0).unwrap();
        assert!(!cron.matches(&dt2));
        // Saturday 2025-01-11 08:00 UTC — weekend
        let dt3 = Utc.with_ymd_and_hms(2025, 1, 11, 8, 0, 0).unwrap();
        assert!(!cron.matches(&dt3));
    }

    #[test]
    fn next_after_finds_next_hourly() {
        let cron = CronExpression::parse("@hourly").unwrap();
        let from = Utc.with_ymd_and_hms(2025, 1, 6, 8, 30, 0).unwrap();
        let next = cron.next_after(&from).unwrap();
        assert_eq!(next.hour(), 9);
        assert_eq!(next.minute(), 0);
    }

    #[test]
    fn next_after_finds_next_daily() {
        let cron = CronExpression::parse("@daily").unwrap();
        let from = Utc.with_ymd_and_hms(2025, 1, 6, 8, 0, 0).unwrap();
        let next = cron.next_after(&from).unwrap();
        assert_eq!(next.day(), 7);
        assert_eq!(next.hour(), 0);
        assert_eq!(next.minute(), 0);
    }

    #[test]
    fn next_after_respects_weekday() {
        let cron = CronExpression::parse("0 8 * * 1-5").unwrap();
        // Friday 2025-01-10 08:00 UTC — should find Monday
        let from = Utc.with_ymd_and_hms(2025, 1, 10, 8, 0, 0).unwrap();
        let next = cron.next_after(&from).unwrap();
        assert_eq!(next.day(), 13); // Monday
        assert_eq!(next.hour(), 8);
    }

    #[test]
    fn display_roundtrip() {
        let expr = "0 8 * * 1-5";
        let cron = CronExpression::parse(expr).unwrap();
        assert_eq!(cron.to_string(), expr);
    }

    #[test]
    fn serde_roundtrip() {
        let cron = CronExpression::parse("*/15 9-17 * * 1-5").unwrap();
        let json = serde_json::to_string(&cron).unwrap();
        let back: CronExpression = serde_json::from_str(&json).unwrap();
        assert_eq!(cron, back);
    }
}
