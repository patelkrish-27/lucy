//! Voice-command audio feedback: the whiteboard's `own STT / play TTS beep` box.
//!
//! Lucy ships its own STT (`GroqStt`); this module owns the other half of the
//! voice loop — short non-speech acknowledgment beeps plus a `speak_text`
//! extension point for a real TTS voice.
//!
//! Design notes:
//! - No new dependencies: beeps are synthesized as in-memory WAV and played
//!   through `paplay` / `aplay` / `play` when present, falling back to the
//!   terminal bell (`\x07`) so headless/CI runs never fail.
//! - `speak_text` shells out to `$LUCY_TTS_COMMAND` with the text on stdin
//!   when set (e.g. a Piper/ElevenLabs CLI). Unset → no-op `Ok(())`, so TTS
//!   is opt-in and never blocks the command pipeline.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Synthesize a mono 16-bit 22.05kHz sine beep and play it.
///
/// Never fails the caller: every I/O error degrades to a terminal bell.
pub fn play_beep(freq_hz: f32, duration: Duration) {
    if play_beep_wav(freq_hz, duration).is_err() {
        // Fallback: terminal bell. Flushed so it lands immediately.
        let _ = write!(std::io::stderr(), "\x07");
        let _ = std::io::stderr().flush();
    }
}

/// Short high blip: voice recording started / command accepted.
pub fn ack_beep() {
    play_beep(880.0, Duration::from_millis(120));
}

/// Two-tone rising blip: plan finished, goal complete.
pub fn done_beep() {
    play_beep(660.0, Duration::from_millis(110));
    std::thread::sleep(Duration::from_millis(90));
    play_beep(990.0, Duration::from_millis(160));
}

/// Low buzz: command denied / failed / no speech detected.
pub fn error_beep() {
    play_beep(220.0, Duration::from_millis(220));
}

fn play_beep_wav(freq_hz: f32, duration: Duration) -> anyhow::Result<()> {
    let wav = synth_sine_wav(freq_hz, duration);
    for player in [
        ("paplay", vec!["/dev/stdin"]),
        ("aplay", vec!["-q", "/dev/stdin"]),
        ("play", vec!["-q", "-t", "wav", "-"]),
    ] {
        if let Ok(mut child) = Command::new(player.0)
            .args(&player.1)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(&wav);
            }
            if child.wait().is_ok() {
                return Ok(());
            }
        }
    }
    anyhow::bail!("no audio player found")
}

fn synth_sine_wav(freq_hz: f32, duration: Duration) -> Vec<u8> {
    const RATE: u32 = 22_050;
    let n = ((duration.as_millis() as u32 * RATE) / 1000).max(1) as usize;
    let mut pcm: Vec<i16> = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        // 10ms raised-cosine fade in/out to avoid clicks.
        let fade = ((i as f32 / (0.01 * RATE as f32)).min(1.0))
            .min(((n - i) as f32 / (0.01 * RATE as f32)).min(1.0));
        let s = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * fade;
        pcm.push((s * 12_000.0) as i16);
    }
    let data_len = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&RATE.to_le_bytes());
    out.extend_from_slice(&(RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Speak `text` through an external TTS command when configured.
///
/// Set `LUCY_TTS_COMMAND` to a player, e.g.
/// `LUCY_TTS_COMMAND="piper --model en.onnx --output-raw | paplay --raw --rate 22050"`.
/// The text is piped on stdin; `sh -c` runs the pipeline. Unset → `Ok(())`.
pub fn speak_text(text: &str) -> anyhow::Result<()> {
    let cmd = std::env::var("LUCY_TTS_COMMAND").unwrap_or_default();
    let cmd = cmd.trim();
    if cmd.is_empty() || text.trim().is_empty() {
        return Ok(());
    }
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes());
    }
    let _ = child.wait();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_wav_has_valid_header() {
        let wav = synth_sine_wav(440.0, Duration::from_millis(50));
        assert!(wav.len() > 44);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn speak_is_noop_without_command() {
        unsafe { std::env::remove_var("LUCY_TTS_COMMAND") };
        assert!(speak_text("hello").is_ok());
        assert!(speak_text("   ").is_ok());
    }

    #[test]
    fn beep_never_panics_headless() {
        // No audio daemon in CI: must degrade to bell, not panic.
        play_beep(440.0, Duration::from_millis(10));
        ack_beep();
        done_beep();
        error_beep();
    }
}
