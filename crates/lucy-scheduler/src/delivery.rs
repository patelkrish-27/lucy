//! Delivery of task results to various channels.
//!
//! Supports delivery to the TUI, gateway, and webhooks. The delivery system
//! is intentionally generic — it delivers whatever output a task produces
//! to whatever channel the user configured.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::debug;

/// The channel to deliver results to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryChannel {
    /// Deliver to the terminal UI.
    Tui,
    /// Deliver to the gateway (remote API).
    Gateway,
    /// Deliver to a webhook URL.
    Webhook(String),
}

/// Configuration for how task results are delivered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryConfig {
    pub channel: DeliveryChannel,
    pub destination: String,
}

impl DeliveryConfig {
    /// Create a new delivery config.
    pub fn new(channel: DeliveryChannel, destination: impl Into<String>) -> Self {
        Self {
            channel,
            destination: destination.into(),
        }
    }

    /// Create a TUI delivery config.
    pub fn tui() -> Self {
        Self {
            channel: DeliveryChannel::Tui,
            destination: "default".to_string(),
        }
    }

    /// Create a gateway delivery config.
    pub fn gateway(destination: impl Into<String>) -> Self {
        Self {
            channel: DeliveryChannel::Gateway,
            destination: destination.into(),
        }
    }

    /// Create a webhook delivery config.
    pub fn webhook(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            channel: DeliveryChannel::Webhook(url.clone()),
            destination: url,
        }
    }
}

/// A message to be delivered to a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryMessage {
    pub task_id: String,
    pub output: String,
    pub success: bool,
    pub timestamp: u64,
    pub destination: String,
}

impl DeliveryMessage {
    pub fn new(
        task_id: impl Into<String>,
        output: impl Into<String>,
        success: bool,
        timestamp: u64,
        destination: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            output: output.into(),
            success,
            timestamp,
            destination: destination.into(),
        }
    }
}

/// Errors that can occur during delivery.
#[derive(Debug, Error)]
pub enum DeliveryError {
    #[error("delivery channel is closed")]
    ChannelClosed,
    #[error("webhook delivery failed: {0}")]
    WebhookFailed(String),
    #[error("no delivery configured for channel: {0}")]
    NoHandler(String),
}

/// Trait for delivery handlers. Implement this to add new delivery channels.
#[async_trait::async_trait]
pub trait DeliveryHandler: Send + Sync {
    /// Deliver a message to this handler's channel.
    async fn deliver(&self, message: &DeliveryMessage) -> Result<(), DeliveryError>;
}

/// TUI delivery handler — sends messages through a channel to the TUI.
pub struct TuiDelivery {
    sender: mpsc::UnboundedSender<DeliveryMessage>,
}

impl TuiDelivery {
    pub fn new(sender: mpsc::UnboundedSender<DeliveryMessage>) -> Self {
        Self { sender }
    }
}

#[async_trait::async_trait]
impl DeliveryHandler for TuiDelivery {
    async fn deliver(&self, message: &DeliveryMessage) -> Result<(), DeliveryError> {
        self.sender
            .send(message.clone())
            .map_err(|_| DeliveryError::ChannelClosed)
    }
}

/// Gateway delivery handler — sends messages through a channel to the gateway.
pub struct GatewayDelivery {
    sender: mpsc::UnboundedSender<DeliveryMessage>,
}

impl GatewayDelivery {
    pub fn new(sender: mpsc::UnboundedSender<DeliveryMessage>) -> Self {
        Self { sender }
    }
}

#[async_trait::async_trait]
impl DeliveryHandler for GatewayDelivery {
    async fn deliver(&self, message: &DeliveryMessage) -> Result<(), DeliveryError> {
        self.sender
            .send(message.clone())
            .map_err(|_| DeliveryError::ChannelClosed)
    }
}

/// Webhook delivery handler — POSTs messages to a URL.
pub struct WebhookDelivery {
    url: String,
    client: reqwest::Client,
}

impl WebhookDelivery {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl DeliveryHandler for WebhookDelivery {
    async fn deliver(&self, message: &DeliveryMessage) -> Result<(), DeliveryError> {
        let response = self
            .client
            .post(&self.url)
            .json(message)
            .send()
            .await
            .map_err(|e| DeliveryError::WebhookFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DeliveryError::WebhookFailed(format!(
                "HTTP {}",
                response.status()
            )));
        }
        Ok(())
    }
}

/// Routes delivery messages to the appropriate handler.
pub struct DeliveryRouter {
    tui: Option<Arc<dyn DeliveryHandler>>,
    gateway: Option<Arc<dyn DeliveryHandler>>,
    webhooks: Arc<std::sync::Mutex<HashMap<String, Arc<dyn DeliveryHandler>>>>,
}

impl std::fmt::Debug for DeliveryRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeliveryRouter")
            .field("tui", &self.tui.as_ref().map(|_| "..."))
            .field("gateway", &self.gateway.as_ref().map(|_| "..."))
            .field("webhooks", &self.webhooks.lock().map(|w| w.len()))
            .finish()
    }
}

impl DeliveryRouter {
    /// Create a new delivery router.
    pub fn new() -> Self {
        Self {
            tui: None,
            gateway: None,
            webhooks: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Set the TUI delivery handler.
    pub fn with_tui(mut self, handler: Arc<dyn DeliveryHandler>) -> Self {
        self.tui = Some(handler);
        self
    }

    /// Set the gateway delivery handler.
    pub fn with_gateway(mut self, handler: Arc<dyn DeliveryHandler>) -> Self {
        self.gateway = Some(handler);
        self
    }

    /// Register a webhook handler for a specific URL.
    pub fn with_webhook(self, url: &str, handler: Arc<dyn DeliveryHandler>) -> Self {
        if let Ok(mut webhooks) = self.webhooks.lock() {
            webhooks.insert(url.to_string(), handler);
        }
        self
    }

    /// Deliver a message according to its config.
    pub async fn deliver(
        &self,
        config: &DeliveryConfig,
        message: &DeliveryMessage,
    ) -> Result<(), DeliveryError> {
        debug!(
            channel = ?config.channel,
            destination = %config.destination,
            "delivering message"
        );

        match &config.channel {
            DeliveryChannel::Tui => {
                if let Some(handler) = &self.tui {
                    handler.deliver(message).await
                } else {
                    Err(DeliveryError::NoHandler("tui".to_string()))
                }
            }
            DeliveryChannel::Gateway => {
                if let Some(handler) = &self.gateway {
                    handler.deliver(message).await
                } else {
                    Err(DeliveryError::NoHandler("gateway".to_string()))
                }
            }
            DeliveryChannel::Webhook(url) => {
                let handler = {
                    let webhooks = self.webhooks.lock().map_err(|_| {
                        DeliveryError::NoHandler("webhook lock poisoned".to_string())
                    })?;
                    webhooks.get(url).cloned()
                };
                if let Some(handler) = handler {
                    handler.deliver(message).await
                } else {
                    Err(DeliveryError::NoHandler(format!("webhook: {url}")))
                }
            }
        }
    }
}

impl Default for DeliveryRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tui_delivery_sends_message() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handler = TuiDelivery::new(tx);
        let router = DeliveryRouter::new().with_tui(Arc::new(handler));

        let config = DeliveryConfig::tui();
        let msg = DeliveryMessage::new("task-1", "hello", true, 12345, "default");

        router.deliver(&config, &msg).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.task_id, "task-1");
        assert_eq!(received.output, "hello");
        assert!(received.success);
    }

    #[tokio::test]
    async fn gateway_delivery_sends_message() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handler = GatewayDelivery::new(tx);
        let router = DeliveryRouter::new().with_gateway(Arc::new(handler));

        let config = DeliveryConfig::gateway("session-42");
        let msg = DeliveryMessage::new("task-2", "result", true, 12345, "session-42");

        router.deliver(&config, &msg).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.destination, "session-42");
    }

    #[tokio::test]
    async fn webhook_delivery_posts() {
        // Start a mock server
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/webhook");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
            let _ = tokio::io::AsyncWriteExt::write(&mut socket, response.as_bytes()).await;
        });

        let handler = WebhookDelivery::new(&url);
        let router = DeliveryRouter::new().with_webhook(&url, Arc::new(handler));

        let config = DeliveryConfig::webhook(&url);
        let msg = DeliveryMessage::new("task-3", "data", true, 12345, &url);

        router.deliver(&config, &msg).await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn no_handler_returns_error() {
        let router = DeliveryRouter::new();
        let config = DeliveryConfig::tui();
        let msg = DeliveryMessage::new("task-4", "hello", true, 12345, "default");

        let result = router.deliver(&config, &msg).await;
        assert!(matches!(result, Err(DeliveryError::NoHandler(_))));
    }

    #[tokio::test]
    async fn unregistered_webhook_returns_error() {
        let router = DeliveryRouter::new();
        let url = "http://localhost:9999/hook";
        let config = DeliveryConfig::webhook(url);
        let msg = DeliveryMessage::new("task-5", "data", true, 12345, url);

        let result = router.deliver(&config, &msg).await;
        assert!(matches!(result, Err(DeliveryError::NoHandler(_))));
    }

    #[test]
    fn delivery_config_constructors() {
        let tui = DeliveryConfig::tui();
        assert_eq!(tui.channel, DeliveryChannel::Tui);
        assert_eq!(tui.destination, "default");

        let gw = DeliveryConfig::gateway("sess-1");
        assert_eq!(gw.channel, DeliveryChannel::Gateway);
        assert_eq!(gw.destination, "sess-1");

        let wh = DeliveryConfig::webhook("http://example.com/hook");
        assert_eq!(wh.channel, DeliveryChannel::Webhook("http://example.com/hook".to_string()));
        assert_eq!(wh.destination, "http://example.com/hook");
    }

    #[test]
    fn delivery_message_new() {
        let msg = DeliveryMessage::new("t1", "out", true, 999, "dest");
        assert_eq!(msg.task_id, "t1");
        assert_eq!(msg.output, "out");
        assert!(msg.success);
        assert_eq!(msg.timestamp, 999);
        assert_eq!(msg.destination, "dest");
    }

    #[test]
    fn webhook_delivery_new() {
        let handler = WebhookDelivery::new("http://example.com");
        assert_eq!(handler.url, "http://example.com");
    }
}
