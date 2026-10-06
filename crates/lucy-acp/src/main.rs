use anyhow::Result;
use lucy_acp::AcpRunner;

#[tokio::main]
async fn main() -> Result<()> {
    let command = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let runner = if command.trim().is_empty() {
        AcpRunner::opencode()
    } else {
        AcpRunner::from_command_line(&command)?
    };
    let info = runner.initialize().await?;
    println!("ACP connected");
    println!("  runner   : {}", info.runner);
    println!("  command  : {}", info.command);
    println!("  protocol : {}", info.protocol_version);
    println!("  agent    : {}", info.agent_name);
    Ok(())
}
