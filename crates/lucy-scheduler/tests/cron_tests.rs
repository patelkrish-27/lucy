//! Integration tests for the cron parser and matcher.

use chrono::{TimeZone, Timelike, Utc};
use lucy_scheduler::{CronError, CronExpression, CronField};

#[test]
fn parses_star_star_star_star_star() {
    let cron = CronExpression::parse("* * * * *").unwrap();
    assert_eq!(cron.minute, CronField::Any);
    assert_eq!(cron.hour, CronField::Any);
    assert_eq!(cron.day_of_month, CronField::Any);
    assert_eq!(cron.month, CronField::Any);
    assert_eq!(cron.day_of_week, CronField::Any);
}

#[test]
fn parses_step_field() {
    let cron = CronExpression::parse("*/15 * * * *").unwrap();
    assert_eq!(cron.minute, CronField::Step(15, 59));
}

#[test]
fn parses_range_field() {
    let cron = CronExpression::parse("0 9-17 * * *").unwrap();
    assert_eq!(cron.hour, CronField::Range(9, 17));
}

#[test]
fn parses_list_field() {
    let cron = CronExpression::parse("0 0 1,15 * *").unwrap();
    assert_eq!(cron.day_of_month, CronField::List(vec![1, 15]));
}

#[test]
fn parses_named_hourly() {
    let cron = CronExpression::parse("@hourly").unwrap();
    assert_eq!(cron.minute, CronField::Specific(0));
    assert_eq!(cron.hour, CronField::Any);
}

#[test]
fn parses_named_daily() {
    let cron = CronExpression::parse("@daily").unwrap();
    assert_eq!(cron.minute, CronField::Specific(0));
    assert_eq!(cron.hour, CronField::Specific(0));
}

#[test]
fn parses_named_weekly() {
    let cron = CronExpression::parse("@weekly").unwrap();
    assert_eq!(cron.day_of_week, CronField::Specific(0));
}

#[test]
fn parses_named_monthly() {
    let cron = CronExpression::parse("@monthly").unwrap();
    assert_eq!(cron.day_of_month, CronField::Specific(1));
}

#[test]
fn parses_every_30m() {
    let cron = CronExpression::parse("@every 30m").unwrap();
    assert_eq!(cron.minute, CronField::Step(30, 59));
}

#[test]
fn parses_every_2h() {
    let cron = CronExpression::parse("@every 2h").unwrap();
    assert_eq!(cron.hour, CronField::Step(2, 23));
}

#[test]
fn parses_every_1d() {
    let cron = CronExpression::parse("@every 1d").unwrap();
    assert_eq!(cron.day_of_month, CronField::Step(1, 31));
}

#[test]
fn rejects_wrong_field_count() {
    assert!(matches!(
        CronExpression::parse("* * *"),
        Err(CronError::WrongFieldCount(3))
    ));
}

#[test]
fn rejects_out_of_range_value() {
    assert!(matches!(
        CronExpression::parse("60 * * * *"),
        Err(CronError::OutOfRange(_, 0, 59))
    ));
}

#[test]
fn rejects_unknown_named_schedule() {
    assert!(matches!(
        CronExpression::parse("@fortnightly"),
        Err(CronError::InvalidNamed(_))
    ));
}

#[test]
fn matches_a_weekday_morning_expression() {
    let cron = CronExpression::parse("0 8 * * 1-5").unwrap();
    let monday_8am = Utc.with_ymd_and_hms(2025, 1, 6, 8, 0, 0).unwrap();
    assert!(cron.matches(&monday_8am));
    let saturday_8am = Utc.with_ymd_and_hms(2025, 1, 11, 8, 0, 0).unwrap();
    assert!(!cron.matches(&saturday_8am));
}

#[test]
fn next_after_finds_next_slot() {
    let cron = CronExpression::parse("*/15 * * * *").unwrap();
    let from = Utc.with_ymd_and_hms(2025, 3, 1, 10, 7, 30).unwrap();
    let next = cron.next_after(&from).unwrap();
    assert_eq!(next.hour(), 10);
    assert_eq!(next.minute(), 15);
    assert_eq!(next.second(), 0);
}

#[test]
fn display_roundtrips_plain_expression() {
    let expr = "30 4 1 6 *";
    let cron = CronExpression::parse(expr).unwrap();
    assert_eq!(cron.to_string(), expr);
}

#[test]
fn serde_roundtrips() {
    let cron = CronExpression::parse("0 12 * * 3").unwrap();
    let json = serde_json::to_string(&cron).unwrap();
    let back: CronExpression = serde_json::from_str(&json).unwrap();
    assert_eq!(cron, back);
}
