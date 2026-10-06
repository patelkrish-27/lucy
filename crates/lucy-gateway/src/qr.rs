//! Pairing QR rendering and LAN address discovery.
//!
//! The QR encodes [`PairingPayload`] as JSON. A phone scanner hands that text to
//! the app, which parses the `url`/`token` fields — no custom URI scheme, and a
//! standard QR scanner on the desktop side can read it too.

use anyhow::{Context, Result};
use qrcode::QrCode;
use qrcode::render::unicode;

use crate::protocol::PairingPayload;

/// Render `text` as a terminal QR code (two rows per line, dark modules as
/// block glyphs). Works in any UTF-8 terminal.
pub fn render_terminal(text: &str) -> Result<String> {
    let code = QrCode::new(text.as_bytes()).context("encoding QR code")?;
    Ok(code
        .render::<unicode::Dense1x2>()
        .dark_color(unicode::Dense1x2::Light)
        .light_color(unicode::Dense1x2::Dark)
        .build())
}

/// Build the payload for a pairing session.
pub fn pairing_payload(
    server_name: &str,
    host: &str,
    port: u16,
    token: &str,
    server_id: &str,
) -> PairingPayload {
    PairingPayload {
        lucy: 1,
        name: server_name.to_string(),
        url: format!("ws://{host}:{port}/ws"),
        http: format!("http://{host}:{port}"),
        token: token.to_string(),
        server_id: server_id.to_string(),
    }
}

/// The address to advertise in a pairing payload.
///
/// A loopback bind is only reachable by the desktop itself, so when the user
/// bound `0.0.0.0` we advertise a concrete LAN address. Discovery is a UDP
/// "connect" to a public address with no packets sent: the kernel picks the
/// interface it would route through, which is the one a phone on the same LAN
/// can reach. No external request happens.
pub fn advertise_host(bind: &str) -> String {
    let bind = bind.trim();
    if bind.is_empty() || bind == "127.0.0.1" || bind == "localhost" || bind == "::1" {
        return "127.0.0.1".to_string();
    }
    if bind != "0.0.0.0" && bind != "::" {
        // An explicit bind address is exactly what the user wants advertised.
        return bind.to_string();
    }
    primary_lan_ip().unwrap_or_else(|| "127.0.0.1".to_string())
}

fn primary_lan_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    // 192.0.2.0/24 is TEST-NET-1: reserved for documentation, so even if a
    // packet were emitted it would go nowhere. `connect` only selects a route.
    socket.connect("192.0.2.1:9").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    if ip.is_loopback() || ip.is_unspecified() {
        None
    } else {
        Some(ip.to_string())
    }
}

/// Hostname for display in the QR and `lucy serve` banner.
pub fn server_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.trim().is_empty())
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "lucy-desktop".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_qr_renders_for_a_real_payload() {
        let payload = pairing_payload("desktop", "10.0.0.5", 9847, "lucy_pair_x", "srv-1");
        let json = serde_json::to_string(&payload).unwrap();
        let qr = render_terminal(&json).expect("payload fits in a QR");
        assert!(qr.lines().count() > 10, "a QR is many rows");
        assert!(qr.contains('\u{2588}') || qr.contains('\u{2584}'));
    }

    #[test]
    fn loopback_bind_advertises_loopback() {
        assert_eq!(advertise_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(advertise_host("localhost"), "127.0.0.1");
        assert_eq!(advertise_host(""), "127.0.0.1");
    }

    #[test]
    fn an_explicit_bind_is_advertised_verbatim() {
        assert_eq!(advertise_host("192.168.1.50"), "192.168.1.50");
        assert_eq!(advertise_host("100.64.0.9"), "100.64.0.9");
    }

    #[test]
    fn the_payload_urls_are_consistent() {
        let p = pairing_payload("desktop", "10.0.0.5", 9911, "tok", "srv-9");
        assert_eq!(p.url, "ws://10.0.0.5:9911/ws");
        assert_eq!(p.http, "http://10.0.0.5:9911");
        assert_eq!(p.token, "tok");
        assert_eq!(p.server_id, "srv-9");
        assert_eq!(p.lucy, 1);
    }
}
