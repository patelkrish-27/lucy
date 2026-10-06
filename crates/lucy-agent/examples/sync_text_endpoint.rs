//! One-shot run of the legacy-endpoint model sync, e.g.:
//!
//!     LUCY_CONFIG=~/.config/lucy/config.toml cargo run -p lucy-agent --example sync_text_endpoint
//!
//! Prints what would be persisted without starting the TUI/CLI runtime.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cfg = lucy_config::LucyConfig::load()?;
    match lucy_agent::sync_text_endpoint(&mut cfg).await {
        Ok(Some(health)) => {
            println!("provider: {}", health.name);
            println!("endpoint: {}", health.models_endpoint);
            println!("models:   {}", health.available_models.join(", "));
            println!("deprecated: {}", health.deprecated_models.join(", "));
            println!("default text model now: {}", cfg.default_text_model());
        }
        Ok(None) => println!("no legacy text endpoint configured"),
        Err(e) => println!("sync failed: {e:#}"),
    }
    Ok(())
}
