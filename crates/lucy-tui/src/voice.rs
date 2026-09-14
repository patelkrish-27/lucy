//! Push-to-talk voice: hotkey matching, hold-to-talk recording lifecycle.

use std::sync::Arc;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use tokio::sync::mpsc;

use lucy_stt::{GroqStt, HoldCapture};

use super::model::App;

pub(crate) fn is_voice_hotkey(key: &KeyEvent, ptt_config: &str) -> bool {
    let ptt = ptt_config.trim().to_ascii_lowercase();
    if ptt.starts_with('f') {
        if let Ok(num) = ptt[1..].parse::<u8>() {
            return key.code == KeyCode::F(num);
        }
    }
    key.code == KeyCode::F(2)
}

// A recording started by a toggle press of the PTT key.
pub(crate) struct Recording {
    pub(crate) cap: HoldCapture,
    pub(crate) started_at: Instant,
}

// Stop recording and dispatch transcription. Safe to call with hold == None.
pub(crate) fn stop_recording(
    hold: &mut Option<Recording>,
    stt: Option<&Arc<GroqStt>>,
    app: &mut App,
    voice_tx: &mpsc::UnboundedSender<Result<String, String>>,
) {
    if let Some(rec) = hold.take() {
        app.listening = false;
        if let Some(stt) = stt {
            let stt = Arc::clone(stt);
            let tx = voice_tx.clone();
            app.status = "Transcribing…".into();
            tokio::spawn(async move {
                let res = stt.finish_hold(rec.cap).await.map_err(|e| e.to_string());
                let _ = tx.send(res);
            });
        } else {
            app.status = "Voice disabled — set GROQ_API_KEY".into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    #[test]
    fn matches_voice_hotkey_case_insensitively_and_fallbacks() {
        let f2 = KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE);
        let f3 = KeyEvent::new(KeyCode::F(3), KeyModifiers::NONE);
        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert!(is_voice_hotkey(&f2, "f2"));
        assert!(is_voice_hotkey(&f2, "F2"));
        assert!(is_voice_hotkey(&f3, "f3"));
        assert!(!is_voice_hotkey(&f3, "f2"));
        assert!(!is_voice_hotkey(&esc, "f2"));
    }
}
