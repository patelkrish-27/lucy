//! ACP integration for Lucy.
//!
//! Lucy acts as an ACP client when it delegates work to an external agent
//! harness such as OpenCode or another ACP-compatible executable.

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, TextContent,
};
use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo, Error, Responder};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use lucy_core::InterruptSignal;

#[derive(Debug, Clone)]
pub struct AcpRunner {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: Vec<(String, String)>,
    pub auto_approve_permissions: bool,
}

impl AcpRunner {
    pub fn new(command: impl Into<String>, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let command = command.into();
        Self {
            name: command.clone(),
            command,
            args: args.into_iter().map(Into::into).collect(),
            cwd: None,
            env: Vec::new(),
            auto_approve_permissions: false,
        }
    }

    pub fn opencode() -> Self {
        Self::new("opencode", ["acp"])
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn auto_approve_permissions(mut self, enabled: bool) -> Self {
        self.auto_approve_permissions = enabled;
        self
    }

    fn command_line(&self) -> String {
        let mut parts = vec![self.command.clone()];
        parts.extend(self.args.iter().cloned());
        parts.join(" ")
    }

    fn acp_agent(&self) -> Result<AcpAgent> {
        let config = AcpAgentConfig::new(self.command.clone())
            .args(self.args.clone())
            .envs(self.env.clone());
        Ok(AcpAgent::new(config))
    }

    pub async fn prompt(&self, prompt: &str) -> Result<AcpResult> {
        self.prompt_with_interrupt(prompt, InterruptSignal::new()).await
    }

    /// Run one ACP prompt while propagating Lucy's interrupt to the remote
    /// session. Dropping the pending request also triggers the SDK's protocol
    /// cancellation, while `session/cancel` tells ACP-aware agents to stop the
    /// whole prompt turn and its nested work.
    pub async fn prompt_with_interrupt(&self, prompt: &str, interrupt: InterruptSignal) -> Result<AcpResult> {
        self.prompt_with_updates_and_interrupt(prompt, |_| {}, interrupt).await
    }

    pub async fn prompt_with_updates<F>(&self, prompt: &str, on_update: F) -> Result<AcpResult>
    where
        F: FnMut(&SessionNotification),
    {
        self.prompt_with_updates_and_interrupt(prompt, on_update, InterruptSignal::new()).await
    }

    pub async fn prompt_with_updates_and_interrupt<F>(&self, prompt: &str, mut on_update: F, interrupt: InterruptSignal) -> Result<AcpResult>
    where
        F: FnMut(&SessionNotification),
    where
        F: FnMut(&SessionNotification),
    {
        let agent = self.acp_agent()?;
        let cwd = match self.cwd.clone() {
            Some(path) if path.is_absolute() => path,
            Some(path) => std::env::current_dir()
                .context("could not resolve ACP working directory")?
                .join(path),
            None => std::env::current_dir().context("could not determine ACP working directory")?,
        };
        anyhow::ensure!(
            cwd.is_absolute(),
            "ACP session working directory must be absolute: {}",
            cwd.display()
        );
        anyhow::ensure!(
            cwd.is_dir(),
            "ACP session working directory does not exist or is not a directory: {}",
            cwd.display()
        );
        let (tx, mut rx): (
            UnboundedSender<SessionNotification>,
            UnboundedReceiver<SessionNotification>,
        ) = unbounded_channel();

        let auto_approve = self.auto_approve_permissions;
        let result = Client
            .builder()
            .name("lucy")
            .on_receive_notification(
                async move |notification: SessionNotification, _connection: ConnectionTo<Agent>| {
                    tx.send(notification).map_err(Error::into_internal_error)
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                move |request: RequestPermissionRequest,
                      responder: Responder<RequestPermissionResponse>,
                      _connection: ConnectionTo<Agent>| async move {
                    if auto_approve {
                        if let Some(option) = request.options.first() {
                            return responder.respond(RequestPermissionResponse::new(
                                RequestPermissionOutcome::Selected(
                                    SelectedPermissionOutcome::new(option.option_id.clone()),
                                ),
                            ));
                        }
                    }
                    responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Cancelled,
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(agent, async move |connection: ConnectionTo<Agent>| {
                let init = connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let session = connection
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?
                    .session_id;
                let prompt_request = connection.send_request(PromptRequest::new(
                    session.clone(),
                    vec![ContentBlock::Text(TextContent::new(prompt.to_owned()))],
                ));
                let prompt_response = tokio::select! {
                    result = prompt_request.block_task() => result?,
                    _ = interrupt.notified() => {
                        // ACP v1 requires session/cancel for cancelling the
                        // current prompt turn. The request handle is dropped
                        // on this branch as well, which sends protocol-level
                        // $/cancel_request as a second cooperative signal.
                        connection
                            .send_notification(CancelNotification::new(session.clone()))
                            .map_err(Error::into_internal_error)?;
                        anyhow::bail!("ACP prompt cancelled");
                    }
                };
                Ok((init, session, prompt_response))
            })
            .await
            .context("ACP connection failed")?;

        let mut text = String::new();
        while let Ok(notification) = rx.try_recv() {
            on_update(&notification);
            collect_v1_text(&notification, &mut text);
        }
        while let Ok(notification) = rx.try_recv() {
            on_update(&notification);
            collect_v1_text(&notification, &mut text);
        }

        let (init, session_id, prompt_response) = result;
        Ok(AcpResult {
            runner: self.name.clone(),
            command: self.command_line(),
            session_id: session_id.to_string(),
            protocol_version: format!("{:?}", init.protocol_version),
            agent_name: init.agent_info.name,
            text,
            stop_reason: format!("{:?}", prompt_response.stop_reason),
        })
    }

    pub async fn initialize(&self) -> Result<AcpInfo> {
        let agent = self.acp_agent()?;
        let result = Client
            .builder()
            .name("lucy")
            .connect_with(agent, async move |connection: ConnectionTo<Agent>| {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await
                    .map_err(anyhow::Error::from)
            })
            .await
            .context("ACP initialization failed")?;

        Ok(AcpInfo {
            runner: self.name.clone(),
            command: self.command_line(),
            protocol_version: format!("{:?}", result.protocol_version),
            agent_name: result.agent_info.name,
        })
    }

    pub fn from_command_line(command: &str) -> Result<Self> {
        let command = command.trim();
        anyhow::ensure!(!command.is_empty(), "ACP runner command cannot be empty");
        let agent = AcpAgent::from_str(command)
            .with_context(|| format!("invalid ACP runner command: {command}"))?;
        let config = agent.into_config();
        let mut runner = Self::new(
            config.command().to_string_lossy().to_string(),
            config.arguments().iter().cloned().collect::<Vec<_>>(),
        );
        runner.env = config.environment().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        Ok(runner)
    }

    pub fn workspace(self, path: &Path) -> Self {
        self.with_cwd(path)
    }
}

#[derive(Debug, Clone)]
pub struct AcpInfo {
    pub runner: String,
    pub command: String,
    pub protocol_version: String,
    pub agent_name: String,
}

#[derive(Debug, Clone)]
pub struct AcpResult {
    pub runner: String,
    pub command: String,
    pub session_id: String,
    pub protocol_version: String,
    pub agent_name: String,
    pub text: String,
    pub stop_reason: String,
}

fn collect_v1_text(notification: &SessionNotification, output: &mut String) {
    match &notification.update {
        agent_client_protocol::schema::v1::SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(text) = &chunk.content {
                output.push_str(&text.text);
            }
        }
        agent_client_protocol::schema::v1::SessionUpdate::AgentMessage(message) => {
            for content in &message.content {
                if let ContentBlock::Text(text) = content {
                    output.push_str(&text.text);
                }
            }
        }
        _ => {}
    }
}
