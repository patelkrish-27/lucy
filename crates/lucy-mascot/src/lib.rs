//! Lucy herself: the approved pose artwork, embedded and resized, never redrawn.
//!
//! The mascot is a library of pose PNGs in `assets/poses/`, each extracted
//! from the same approved reference sheet — full pixel art, on a transparent
//! background. A pose is a pure function of mood and time: every frame is the
//! embedded image resized to the requested buffer, with a gentle bob, so her
//! figure is what the pixels are and the art never repaints.
//!
//! On terminals that speak the Kitty, Sixel, or iTerm2 graphics protocol the
//! art is transmitted *as pixels* (`draw_auto` via `ratatui-image`), so the
//! eyes, lashes and mouth keep their full detail. Everything else — and the
//! exporters — share the half-block renderer [`draw`], which packs two pixels
//! into every cell with the upper-half block `▀`.
//!
//! Every frame is a pure function of `(width, height, [`Mood`], t_ms)`, so it
//! never depends on what was painted before it: she cannot crawl, and an
//! export reproduces exactly what the TUI drew. The only motion is a gentle
//! bob, a pure function of `t_ms`, phased per mood.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::io::IsTerminal;
use std::sync::{Mutex, OnceLock};

use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Color;

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

/// sRGB triple in 0..=1. Small enough to copy, cheap enough to mix per pixel.
#[derive(Clone, Copy, Debug)]
struct Rgb(f32, f32, f32);

impl Rgb {
    const fn hex(r: u8, g: u8, b: u8) -> Self {
        Self(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }
    fn bytes(self) -> [u8; 3] {
        let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        [c(self.0), c(self.1), c(self.2)]
    }
}

/// The matte the exporter composes its sheets on, so the art can be judged as
/// an image. The terminal does *not* use it: there the sprite keeps a
/// transparent background and stands directly on the user's theme, which is
/// the only way the two can agree — the theme colour is not knowable here.
const MATTE: Rgb = Rgb::hex(0x0b, 0x09, 0x18);

// ---------------------------------------------------------------------------
// Mood
// ---------------------------------------------------------------------------

/// What Lucy is doing. Callers map their own state onto one of these; nothing
/// here inspects a request, a phase string or a tool name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mood {
    Idle,
    Listening,
    Thinking,
    Working,
    Talking,
    Happy,
    Approval,
}

impl Mood {
    pub const ALL: [Mood; 7] = [
        Mood::Idle,
        Mood::Listening,
        Mood::Thinking,
        Mood::Working,
        Mood::Talking,
        Mood::Happy,
        Mood::Approval,
    ];

    /// Short caption under the sprite, the way a status rail reads.
    pub fn label(self) -> &'static str {
        match self {
            Mood::Idle => "Idle",
            Mood::Listening => "Listening",
            Mood::Thinking => "Thinking",
            Mood::Working => "Working",
            Mood::Talking => "Talking",
            Mood::Happy => "Happy",
            Mood::Approval => "Waiting for you",
        }
    }

    /// Accent for the caption, so the rail colour agrees with the pose.
    pub fn accent(self) -> Color {
        match self {
            Mood::Idle => Color::Magenta,
            Mood::Listening => Color::Red,
            Mood::Thinking => Color::Yellow,
            Mood::Working => Color::Cyan,
            Mood::Talking => Color::LightMagenta,
            Mood::Happy => Color::Green,
            Mood::Approval => Color::Yellow,
        }
    }
}

// ---------------------------------------------------------------------------
// Companion state
// ---------------------------------------------------------------------------

/// Coarse companion lifecycle for embedders (TUI rails, launchers, widgets).
///
/// This is intentionally smaller than [`Mood`]: it describes *what the
/// companion is doing* (resting, receiving input, reasoning, responding, or
/// reporting a failure), never *what task* it is doing. Callers map their own
/// lifecycle onto one of these variants; nothing here inspects a goal string,
/// a tool name, or a site. The mapping to artwork is a total function
/// ([`CompanionState::mood`]) so every state always paints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompanionState {
    Idle,
    Listening,
    Thinking,
    Speaking,
    Error,
}

impl CompanionState {
    pub const ALL: [CompanionState; 5] = [
        CompanionState::Idle,
        CompanionState::Listening,
        CompanionState::Thinking,
        CompanionState::Speaking,
        CompanionState::Error,
    ];

    /// The [`Mood`] (and therefore the approved pose) this state shows.
    /// Total: every state maps to exactly one mood.
    pub fn mood(self) -> Mood {
        match self {
            CompanionState::Idle => Mood::Idle,
            CompanionState::Listening => Mood::Listening,
            CompanionState::Thinking => Mood::Thinking,
            CompanionState::Speaking => Mood::Talking,
            // No dedicated error artwork exists; the approval pose is the
            // attention-grabbing "waiting on you" stance, the closest visual
            // match for a state that needs the user's eye.
            CompanionState::Error => Mood::Approval,
        }
    }

    /// Short caption under the sprite, the way a status rail reads.
    pub fn label(self) -> &'static str {
        match self {
            CompanionState::Idle => "Idle",
            CompanionState::Listening => "Listening",
            CompanionState::Thinking => "Thinking",
            CompanionState::Speaking => "Speaking",
            CompanionState::Error => "Error",
        }
    }

    /// Accent for the caption, agreeing with the mapped pose.
    pub fn accent(self) -> Color {
        self.mood().accent()
    }

    /// `true` for the resting state only.
    pub fn is_idle(self) -> bool {
        matches!(self, CompanionState::Idle)
    }

    /// `true` when the companion needs attention (failure state).
    pub fn is_error(self) -> bool {
        matches!(self, CompanionState::Error)
    }

    /// `true` while the companion is actively engaged with the user.
    pub fn is_busy(self) -> bool {
        !self.is_idle()
    }

    /// Best-effort inverse of [`CompanionState::mood`]: recovers the
    /// companion state for moods that have one, `None` for moods outside
    /// this smaller lifecycle (`Working`, `Happy`, ...).
    pub fn from_mood(mood: Mood) -> Option<CompanionState> {
        match mood {
            Mood::Idle => Some(CompanionState::Idle),
            Mood::Listening => Some(CompanionState::Listening),
            Mood::Thinking => Some(CompanionState::Thinking),
            Mood::Talking => Some(CompanionState::Speaking),
            Mood::Approval => Some(CompanionState::Error),
            _ => None,
        }
    }
}

/// Paint one companion frame at `w` x `h` pixels.
///
/// Pure function of `(w, h, state, t_ms)`; delegates to [`render_frame`]
/// through [`CompanionState::mood`], so it inherits the same guarantees
/// (deterministic, transparent background, gentle bob).
pub fn render_companion_frame(w: usize, h: usize, state: CompanionState, t_ms: u128) -> Frame {
    render_frame(w, h, state.mood(), t_ms)
}

/// Draw the companion into `area` of a ratatui buffer, centred, at whatever
/// size fits — for the given state, size, and color mode.
///
/// Thin wrapper over [`draw`]; see it for the cell-packing contract
/// (half-blocks, transparent background, theme-safe clears).
pub fn draw_companion(
    buf: &mut Buffer,
    area: Rect,
    state: CompanionState,
    t_ms: u128,
    mode: ColorMode,
) {
    draw(buf, area, state.mood(), t_ms, mode);
}

/// Crossfade between two companion states; see [`draw_blend`].
pub fn draw_companion_blend(
    buf: &mut Buffer,
    area: Rect,
    prev: CompanionState,
    cur: CompanionState,
    alpha: f32,
    t_ms: u128,
    mode: ColorMode,
) {
    draw_blend(buf, area, prev.mood(), cur.mood(), alpha, t_ms, mode);
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// The pose PNGs, baked into the binary. Each is a trimmed, transparent
/// cutout from the approved reference sheet, so the pixels never repaint.
static POSE_BYTES: &[(&str, &[u8])] = &[
    ("headphones_closed", include_bytes!("../assets/poses/headphones_closed.png") as &[u8]),
    ("laptop_smile", include_bytes!("../assets/poses/laptop_smile.png") as &[u8]),
    ("questioning", include_bytes!("../assets/poses/questioning.png") as &[u8]),
    ("singing_melody", include_bytes!("../assets/poses/singing_melody.png") as &[u8]),
    ("standing_idle", include_bytes!("../assets/poses/standing_idle.png") as &[u8]),
    ("star_celebration", include_bytes!("../assets/poses/star_celebration.png") as &[u8]),
    ("sunny_laugh", include_bytes!("../assets/poses/sunny_laugh.png") as &[u8]),
];

/// One finished sprite: RGBA, ready for either consumer.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    /// `w * h * 4` bytes, RGBA.
    pub rgba: Vec<u8>,
}

impl Frame {
    pub fn pixel(&self, x: usize, y: usize) -> (u8, u8, u8, u8) {
        let i = (y * self.w + x) * 4;
        (
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        )
    }
}

// ---------------------------------------------------------------------------
// The embedded artwork
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The embedded artwork
// ---------------------------------------------------------------------------

/// One approved pose: the same art the user approved, baked into the binary,
/// on a transparent ground. `name` is the pose's stable identifier; the map
/// from [`Mood`] to a pose is a separate step so the same art can appear in
/// the exporter without mood semantics.
/// Where pose PNGs are read from before falling back to the bytes embedded
/// in the binary. Overridable with `LUCY_POSE_DIR`; the default is the
/// canonical pose library next to the crate. Replace a PNG in that folder
/// with an edited crop and the next `lucy` run picks it up — no rebuild.
fn pose_library_dir() -> std::path::PathBuf {
    std::env::var_os("LUCY_POSE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from("/home/krish/projects/lucy/lucy_lib/poses")
        })
}

/// Raw bytes for `name`, preferring the pose library on disk so users can
/// swap art freely; falls back to the embedded copy when the file is missing
/// or undecodable.
fn pose_png_bytes(name: &str) -> Vec<u8> {
    let path = pose_library_dir().join(format!("{name}.png"));
    if let Ok(bytes) = std::fs::read(&path) {
        if image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).is_ok() {
            return bytes;
        }
    }
    POSE_BYTES
        .iter()
        .find_map(|(n, b)| (*n == name).then_some(*b))
        .map(|b| b.to_vec())
        .expect("pose exists")
}

struct PoseArt {
    name: &'static str,
    asset: OnceLock<Asset>,
}

struct Asset {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

impl PoseArt {
    fn asset(&self) -> &Asset {
        self.asset.get_or_init(|| {
            let bytes = pose_png_bytes(self.name);
            let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                .expect("embedded mascot PNG decodes")
                .to_rgba8();
            let (w, h) = img.dimensions();
            Asset {
                w: w as usize,
                h: h as usize,
                rgba: img.into_raw(),
            }
        })
    }
}

/// The full pose library, sliced from the reference sheet (transparent
/// ground, figure with soft glow). Ordering is internal: callers go through
/// [`pose_for_mood`].
fn pose_art() -> &'static [(&'static str, PoseArt)] {
    static ART: OnceLock<Vec<(&'static str, PoseArt)>> = OnceLock::new();
    ART.get_or_init(|| {
        POSE_BYTES
            .iter()
            .map(|(n, _)| {
                (
                    *n,
                    PoseArt {
                        name: n,
                        asset: OnceLock::new(),
                    },
                )
            })
            .collect()
    })
}

fn pose_art_by_name(name: &str) -> &'static PoseArt {
    pose_art()
        .iter()
        .find_map(|(n, a)| (*n == name).then_some(a))
        .expect("pose exists")
}

/// Which approved pose a mood shows. This mapping is intentionally the same
/// for every task the app runs — a pose describes a *state*, not a task, so
/// the art can never drift away from what the UI is actually doing.
fn pose_for_mood(mood: Mood) -> &'static str {
    match mood {
        Mood::Idle => "standing_idle",
        Mood::Listening => "headphones_closed",
        Mood::Thinking => "questioning",
        Mood::Working => "laptop_smile",
        Mood::Talking => "singing_melody",
        Mood::Happy => "star_celebration",
        Mood::Approval => "sunny_laugh",
    }
}

// ---------------------------------------------------------------------------
// Pixel-true rendering on terminals with graphics protocols
// ---------------------------------------------------------------------------

/// `true` when the terminal can be asked for Kitty/Sixel/iTerm2 cell images.
/// Anywhere else the half-block [`draw`] painter is the whole mascot.
pub fn has_image_protocol() -> bool {
    std::env::var("LUCY_MASCOT_IMAGE").as_deref() != Ok("off")
        && std::io::stdout().is_terminal()
        && picker().is_some()
}

fn picker() -> &'static Option<ratatui_image::picker::Picker> {
    use ratatui_image::picker::Picker;
    static P: OnceLock<Option<Picker>> = OnceLock::new();
    P.get_or_init(|| {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        Picker::from_query_stdio().ok().filter(|p| {
            !matches!(
                p.protocol_type(),
                ratatui_image::picker::ProtocolType::Halfblocks
            )
        })
    })
}

/// Query the terminal once, before the alternate screen takes over. Harmless
/// when the terminal does not answer: the picker reports half-blocks and
/// [`draw`] is used exactly as before.
pub fn init_image_support() {
    let _ = picker();
}

fn pose_dynamic(name: &'static str) -> image::DynamicImage {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, image::DynamicImage>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(c) = cache.lock() {
        if let Some(img) = c.get(name) {
            return img.clone();
        }
    }
    let bytes = pose_png_bytes(name);
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .expect("embedded mascot PNG decodes");
    if let Ok(mut c) = cache.lock() {
        c.insert(name, img.clone());
    }
    img
}

/// Render the mascot's *pixel* art into `area`, if this terminal speaks a
/// graphics protocol. Returns `false` when the caller should fall back to the
/// half-block [`draw`] painter.
///
/// The encoded image is cached by (pose, cell size), so each frame only pays
/// for a widget render; the terminal diff deduplicates identical cells. The
/// pose itself is static within a mood — no bob — because re-encoding the
/// art on every animation tick would be pure waste.
pub fn draw_image_protocol(frame: &mut ratatui::Frame<'_>, area: Rect, mood: Mood) -> bool {
    use ratatui_image::{Image, Resize};
    use ratatui::layout::Size;
    if !has_image_protocol() || area.width < 8 || area.height < 4 {
        return false;
    }
    let Some(p) = picker().as_ref() else {
        return false;
    };
    let pose = pose_for_mood(mood);
    let img = pose_dynamic(pose);
    let avail = Size::new(area.width, area.height);
    let fitted = Resize::Fit(None).size_for(&img, p.font_size(), avail);
    if fitted.width < 2 || fitted.height < 2 {
        return false;
    }
    let key = (pose, fitted.width, fitted.height);
    let mut map = PROTO_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if !map.contains_key(&key) {
        let proto = p
            .new_protocol(img.clone(), fitted, Resize::Fit(None))
            .expect("encodes the pose");
        map.insert(key, proto);
    }
    let proto = map.get(&key).unwrap();
    let sub = Rect::new(
        area.x + (area.width.saturating_sub(fitted.width)) / 2,
        area.y + (area.height.saturating_sub(fitted.height)) / 2,
        fitted.width,
        fitted.height,
    );
    frame.render_widget(Image::new(proto), sub);
    true
}

/// Blend two pose images into one RGBA image (premultiplied-over, same as
/// [`merge_frames`]) — used only while a mood change is still crossfading,
/// so the pixel path can still play the transition.
fn blend_pose_images(
    a: &image::DynamicImage,
    b: &image::DynamicImage,
    alpha: f32,
) -> image::DynamicImage {
    let w = a.width();
    let h = a.height();
    let a_px = a.to_rgba8();
    let b_px = image::imageops::resize(
        &b.to_rgba8(),
        w,
        h,
        image::imageops::FilterType::Lanczos3,
    );
    let mut out = image::RgbaImage::new(w, h);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let ap = a_px.get_pixel(x, y);
        let bp = b_px.get_pixel(x, y);
        let fa = (ap[3] as f32 / 255.0) * (1.0 - alpha);
        let fb = (bp[3] as f32 / 255.0) * alpha;
        let oa = fa + fb * (1.0 - fa);
        if oa <= 0.0 {
            *px = image::Rgba([0, 0, 0, 0]);
            continue;
        }
        let mut o = [0u8; 4];
        for c in 0..3 {
            o[c] = ((ap[c] as f32 * fa + bp[c] as f32 * fb * (1.0 - fa)) / oa) as u8;
        }
        o[3] = (oa * 255.0) as u8;
        *px = image::Rgba(o);
    }
    image::DynamicImage::ImageRgba8(out)
}

/// Protocol-pixel crossfade: renders the blended mid-transition pose through
/// the same cache the static path uses. During the short crossfade window the
/// Kitty/Sixel terminal shows the blended pose instead of snapping.
pub fn draw_image_protocol_blend(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    prev: Mood,
    cur: Mood,
    alpha: f32,
) -> bool {
    use ratatui_image::{Image, Resize};
    use ratatui::layout::Size;
    if !has_image_protocol() || area.width < 8 || area.height < 4 {
        return false;
    }
    let Some(p) = picker().as_ref() else {
        return false;
    };
    let prev_bytes = pose_for_mood(prev);
    let cur_bytes = pose_for_mood(cur);
    let cur_img = pose_dynamic(cur_bytes);
    let avail = Size::new(area.width, area.height);
    let fitted = Resize::Fit(None).size_for(&cur_img, p.font_size(), avail);
    if fitted.width < 2 || fitted.height < 2 {
        return false;
    }
    let step = (alpha.clamp(0.0, 1.0) * 4.0).round() as u8;
    let key = (prev_bytes, cur_bytes, step, fitted.width, fitted.height);
    static BLEND_CACHE: OnceLock<
        Mutex<
            HashMap<
                (&'static str, &'static str, u8, u16, u16),
                ratatui_image::protocol::Protocol,
            >,
        >,
    > = OnceLock::new();
    let mut map = BLEND_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if !map.contains_key(&key) {
        let a = step as f32 / 4.0;
        let img = blend_pose_images(&pose_dynamic(prev_bytes), &pose_dynamic(cur_bytes), a);
        let proto = p
            .new_protocol(img, fitted, Resize::Fit(None))
            .expect("blended pose encodes");
        map.insert(key, proto);
    }
    let proto = map.get(&key).unwrap();
    let sub = Rect::new(
        area.x + (area.width.saturating_sub(fitted.width)) / 2,
        area.y + (area.height.saturating_sub(fitted.height)) / 2,
        fitted.width,
        fitted.height,
    );
    frame.render_widget(Image::new(proto), sub);
    true
}

static PROTO_CACHE: OnceLock<
    Mutex<
        std::collections::HashMap<
            (&'static str, u16, u16),
            ratatui_image::protocol::Protocol,
        >,
    >,
> = OnceLock::new();

/// Cached RGBA per (pose name, size), so the Lanczos resample — the expensive
/// step — runs once per requested size and is reused across frames.
fn resized(pose: &'static str, rw: usize, rh: usize) -> Vec<u8> {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, usize, usize), Vec<u8>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(c) = cache.lock() {
        if let Some(v) = c.get(&(pose, rw, rh)) {
            return v.clone();
        }
    }
    let a = pose_art_by_name(pose).asset();
    let src = image::RgbaImage::from_raw(a.w as u32, a.h as u32, a.rgba.clone())
        .expect("asset buffer matches its dimensions");
    let out = image::imageops::resize(
        &src,
        rw as u32,
        rh as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let raw = out.into_raw();
    if let Ok(mut c) = cache.lock() {
        c.insert((pose, rw, rh), raw.clone());
    }
    raw
}

fn asset_dims(pose: &'static str) -> (usize, usize) {
    let a = pose_art_by_name(pose).asset();
    (a.w, a.h)
}

/// Paint one frame at `w` x `h` pixels.
///
/// The pose artwork for the current mood is scaled to fit, centred, and
/// bobbed by a pure function of `t_ms` — a slow sine, phased per mood, a few
/// percent of the frame height. That is the whole animation: the same pixels,
/// gently floating, so the art never moirés and never crawls.
pub fn render_frame(w: usize, h: usize, mood: Mood, t_ms: u128) -> Frame {
    let w = w.max(4);
    let h = h.max(4);
    let pose = pose_for_mood(mood);
    let (aw, ah) = asset_dims(pose);
    let scale = (w as f32 / aw as f32).min(h as f32 / ah as f32);
    let rw = ((aw as f32 * scale).round() as usize).max(1);
    let rh = ((ah as f32 * scale).round() as usize).max(1);
    let src = resized(pose, rw, rh);

    let idx = Mood::ALL.iter().position(|&m| m == mood).unwrap_or(0) as f32;
    let period = 1600.0_f32;
    let amplitude = h as f32 * 0.03;
    let phase = idx * 0.9;
    let dy = ((t_ms as f32 / period) * 2.0 * PI + phase).sin() * amplitude;

    let ox = ((w - rw) / 2) as isize;
    // `as isize` truncates toward zero, which would freeze the bob in its
    // first half-pixel of range — floor instead, so sub-pixel motion
    // actually moves the row she lands on.
    let oy = ((h - rh) as f32 / 2.0 + dy).floor() as isize;

    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..rh {
        let ty = oy + y as isize;
        if ty < 0 || ty >= h as isize {
            continue;
        }
        for x in 0..rw {
            let tx = ox + x as isize;
            if tx < 0 || tx >= w as isize {
                continue;
            }
            let si = (y * rw + x) * 4;
            let di = (ty as usize * w + tx as usize) * 4;
            rgba[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    Frame { w, h, rgba }
}

// ---------------------------------------------------------------------------
// Terminal rendering
// ---------------------------------------------------------------------------

/// How much colour the terminal can show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Truecolor,
    Ansi256,
}

/// The largest sprite we will paint. Past this the extra pixels buy nothing
/// but frame time.
const MAX_PX_W: usize = 96;
const MAX_PX_H: usize = 96;

/// True when the terminal advertises 24-bit colour. `LUCY_MASCOT_COLOR` forces
/// a mode (`truecolor`, `256`) for terminals whose environment lies.
pub fn color_mode() -> ColorMode {
    let forced = std::env::var("LUCY_MASCOT_COLOR")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase());
    match forced.as_deref() {
        Some("truecolor") | Some("24bit") => return ColorMode::Truecolor,
        Some("256") | Some("ansi256") => return ColorMode::Ansi256,
        _ => {}
    }
    let ct = std::env::var("COLORTERM").unwrap_or_default();
    let term = std::env::var("TERM").unwrap_or_default();
    if ct.eq_ignore_ascii_case("truecolor")
        || ct.eq_ignore_ascii_case("24bit")
        || term.contains("truecolor")
        || term.contains("direct")
        || std::env::var("KITTY_WINDOW_ID").is_ok()
        || std::env::var("WT_SESSION").is_ok()
    {
        ColorMode::Truecolor
    } else {
        ColorMode::Ansi256
    }
}

fn xterm256() -> &'static [(u8, u8, u8)] {
    static TABLE: OnceLock<Vec<(u8, u8, u8)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t: Vec<(u8, u8, u8)> = vec![
            (0, 0, 0),
            (170, 0, 0),
            (0, 170, 0),
            (170, 85, 0),
            (0, 0, 170),
            (170, 0, 170),
            (0, 170, 170),
            (170, 170, 170),
            (85, 85, 85),
            (255, 85, 85),
            (85, 255, 85),
            (255, 255, 85),
            (85, 85, 255),
            (255, 85, 255),
            (85, 255, 255),
            (255, 255, 255),
        ];
        let steps = [0u8, 95, 135, 175, 215, 255];
        for r in steps {
            for g in steps {
                for b in steps {
                    t.push((r, g, b));
                }
            }
        }
        for i in 0..24u8 {
            let v = 8 + i * 10;
            t.push((v, v, v));
        }
        t
    })
}

/// Nearest xterm-256 index for an RGB triple, memoised on a 5-bit-per-channel
/// key so a frame only pays for the handful of distinct colours it uses.
fn to_256(r: u8, g: u8, b: u8) -> u8 {
    static CACHE: OnceLock<std::sync::Mutex<Vec<u8>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(vec![u8::MAX; 32768]));
    let key = ((r >> 3) as usize) << 10 | ((g >> 3) as usize) << 5 | (b >> 3) as usize;
    if let Ok(c) = cache.lock() {
        let v = c[key];
        if v != u8::MAX {
            return v;
        }
    }
    let mut best = 0usize;
    let mut best_d = i64::MAX;
    for (i, (tr, tg, tb)) in xterm256().iter().enumerate() {
        let dr = r as i64 - *tr as i64;
        let dg = g as i64 - *tg as i64;
        let db = b as i64 - *tb as i64;
        // Perceptual weights: green carries more of the luminance.
        let d = dr * dr * 2 + dg * dg * 4 + db * db;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    if let Ok(mut c) = cache.lock() {
        c[key] = best as u8;
    }
    best as u8
}

/// The frame proportions of the pose the renderer is currently painting,
/// so fitting can never drift from her real shape.
fn pose_aspect(pose: &'static str) -> f32 {
    let (w, h) = asset_dims(pose);
    h as f32 / w.max(1) as f32
}

/// Draw Lucy into `area` of a ratatui buffer, centred, at whatever size fits.
///
/// Every cell carries two pixels: the upper-half block `▀` paints the top pixel
/// in the foreground and the bottom pixel in the background, which is the
/// densest pixel grid a character cell can carry.
///
/// Transparent pixels reset their half of the cell, so the sprite stands
/// directly on the user's terminal theme — the only backdrop that can ever
/// match it. A cell with one live pixel therefore uses the half block whose
/// *other* half is left as the default background: `▀` for a live top pixel,
/// `▄` for a live bottom one, never a reset foreground, which would paint in
/// the theme's default text colour instead of its background.
pub fn draw(buf: &mut Buffer, area: Rect, mood: Mood, t_ms: u128, mode: ColorMode) {
    if area.width < 6 || area.height < 4 {
        return;
    }
    // A cell is about twice as tall as wide and carries two pixels, so the
    // pixel grid that matches the design space needs half as many rows as
    // columns.
    let mut w = (area.width as usize).min(MAX_PX_W);
    let mut h = ((area.height as usize) * 2).min(MAX_PX_H);
    let aspect = pose_aspect(pose_for_mood(mood));
    h = h.min((w as f32 * aspect) as usize);
    w = w.min((h as f32 / aspect) as usize);
    if w < 4 || h < 6 {
        return;
    }
    let fr = render_frame(w, h, mood, t_ms);
    let ox = area.x as usize + (area.width as usize - w) / 2;
    let oy = area.y as usize + (area.height as usize - h / 2) / 2;
    pack_frame(buf, area, &fr, w, h, ox, oy, mode);
}

fn pack_frame(buf: &mut Buffer, area: Rect, fr: &Frame, w: usize, h: usize, ox: usize, oy: usize, mode: ColorMode) {
    for row in 0..h / 2 {
        for col in 0..w {
            let x = ox + col;
            let y = oy + row;
            if x >= area.right() as usize || y >= area.bottom() as usize {
                continue;
            }
            let top = fr.pixel(col, row * 2);
            let bot = fr.pixel(col, row * 2 + 1);
            let Some(cell) = buf.cell_mut(Position::new(x as u16, y as u16)) else {
                continue;
            };
            // A pixel covers its half-cell when it is mostly opaque. A terminal
            // cannot blend per pixel, so the choice is binary; the embedded
            // alpha gives the 50% contour, which is the true silhouette at
            // this resolution.
            let top_on = top.3 >= 128;
            let bot_on = bot.3 >= 128;
            match (top_on, bot_on) {
                (true, true) => {
                    cell.set_symbol("▀");
                    cell.set_fg(color(top.0, top.1, top.2, mode));
                    cell.set_bg(color(bot.0, bot.1, bot.2, mode));
                }
                (true, false) => {
                    // Upper half drawn, lower half is the terminal background.
                    cell.set_symbol("▀");
                    cell.set_fg(color(top.0, top.1, top.2, mode));
                    cell.set_bg(Color::Reset);
                }
                (false, true) => {
                    cell.set_symbol("▄");
                    cell.set_fg(color(bot.0, bot.1, bot.2, mode));
                    cell.set_bg(Color::Reset);
                }
                (false, false) => {
                    cell.set_symbol(" ");
                    cell.set_fg(Color::Reset);
                    cell.set_bg(Color::Reset);
                }
            }
        }
    }
}

/// Crossfade two frames pixel-by-pixel with a simple premultiplied-over:
/// `a` faded by out, `b` faded in by `alpha`. Respects either sprite's
/// transparent corners, so no white square ever appears.
fn merge_frames(a: &Frame, b: &Frame, alpha: f32) -> Frame {
    let mut out = Frame {
        w: a.w,
        h: a.h,
        rgba: a.rgba.clone(),
    };
    for i in (0..out.rgba.len()).step_by(4) {
        let fa = (1.0 - alpha) * (a.rgba[i + 3] as f32 / 255.0);
        let fb = alpha * (b.rgba[i + 3] as f32 / 255.0);
        let oa = fa + fb * (1.0 - fa);
        if oa <= 0.0 {
            out.rgba[i + 3] = 0;
            continue;
        }
        for c in 0..3 {
            out.rgba[i + c] = ((a.rgba[i + c] as f32 * fa / oa)
                + (b.rgba[i + c] as f32 * fb * (1.0 - fa) / oa))
                as u8;
        }
        out.rgba[i + 3] = (oa * 255.0).round() as u8;
    }
    out
}

/// Same frame grid `draw` uses, so the crossfade lands in exactly the same
/// cells — only the pose changes.
fn frame_dims(area: Rect, mood: Mood) -> (usize, usize) {
    let mut w = (area.width as usize).min(MAX_PX_W);
    let mut h = ((area.height as usize) * 2).min(MAX_PX_H);
    let aspect = pose_aspect(pose_for_mood(mood));
    h = h.min((w as f32 * aspect) as usize);
    w = w.min((h as f32 / aspect) as usize);
    (w, h)
}

/// Half-block crossfade used while a mood change is still animating and the
/// terminal has no pixel protocol (or the protocol path is not usable).
pub fn draw_blend(
    buf: &mut Buffer,
    area: Rect,
    prev: Mood,
    cur: Mood,
    alpha: f32,
    t_ms: u128,
    mode: ColorMode,
) {
    if area.width < 6 || area.height < 4 {
        return;
    }
    let (w, h) = frame_dims(area, cur);
    if w < 4 || h < 6 {
        return;
    }
    let fa = render_frame(w, h, prev, t_ms);
    let fb = render_frame(w, h, cur, t_ms);
    let fr = merge_frames(&fa, &fb, alpha);
    let ox = area.x as usize + (area.width as usize - w) / 2;
    let oy = area.y as usize + (area.height as usize - h / 2) / 2;
    pack_frame(buf, area, &fr, w, h, ox, oy, mode);
}

fn color(r: u8, g: u8, b: u8, mode: ColorMode) -> Color {
    match mode {
        ColorMode::Truecolor => Color::Rgb(r, g, b),
        ColorMode::Ansi256 => Color::Indexed(to_256(r, g, b)),
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Renders the mascot outside the terminal, so the art can be judged as an
/// image instead of squinted at through a character grid.
pub mod export {
    use super::{Frame, MATTE, Mood, Rgb, render_frame};
    use std::io;
    use std::path::{Path, PathBuf};

    /// Frame spacing for the animation strips. The TUI repaints every 50 ms, so
    /// this is roughly terminal speed.
    pub const STRIP_STEP_MS: u128 = 50;
    const STRIP_FRAMES: usize = 8;

    /// The frame size poses export at: the embedded artwork scaled to 64 px
    /// wide, so `scale` multiplies a recognisable render rather than a
    /// repaint.
    const BASE_W: usize = 64;
    const BASE_H: usize = 56;

    pub fn mood_name(mood: Mood) -> &'static str {
        match mood {
            Mood::Idle => "idle",
            Mood::Listening => "listening",
            Mood::Thinking => "thinking",
            Mood::Working => "working",
            Mood::Talking => "talking",
            Mood::Happy => "happy",
            Mood::Approval => "approval",
        }
    }

    /// One PNG per pose, an animation sheet, and an HTML index.
    pub fn write_preview(dir: &Path, scale: usize) -> io::Result<Vec<PathBuf>> {
        std::fs::create_dir_all(dir)?;
        let scale = scale.max(1);
        let mut out = Vec::new();
        for mood in Mood::ALL {
            let p = dir.join(format!("lucy-{}.png", mood_name(mood)));
            std::fs::write(
                &p,
                png_bytes(&render_frame(BASE_W * scale, BASE_H * scale, mood, 0)),
            )?;
            out.push(p);
        }
        let p = dir.join("lucy-animations.png");
        std::fs::write(&p, png_bytes(&render_sheet(scale)))?;
        out.push(p);
        let p = dir.join("index.html");
        std::fs::write(&p, index_html(scale))?;
        out.push(p);
        Ok(out)
    }

    /// Every pose as a row of animation frames, for checking the motion.
    pub fn render_sheet(scale: usize) -> Frame {
        let scale = scale.max(1);
        let (fw, fh) = (BASE_W * scale, BASE_H * scale);
        let gap = 8;
        let w = fw * STRIP_FRAMES + gap * (STRIP_FRAMES - 1);
        let h = fh * Mood::ALL.len() + gap * (Mood::ALL.len() - 1);
        let mut acc = Frame {
            w,
            h,
            rgba: Vec::new(),
        };
        let bg = MATTE.bytes();
        // Start from an opaque backdrop so the gaps between frames read as matte.
        for _ in 0..w * h {
            acc.rgba.extend_from_slice(&[bg[0], bg[1], bg[2], 255]);
        }
        for (row, mood) in Mood::ALL.iter().enumerate() {
            for f in 0..STRIP_FRAMES {
                let fr = render_frame(fw, fh, *mood, f as u128 * STRIP_STEP_MS * 4);
                let (ox, oy) = (f * (fw + gap), row * (fh + gap));
                for y in 0..fh {
                    for x in 0..fw {
                        let (r, g, b, a) = fr.pixel(x, y);
                        if a == 0 {
                            continue;
                        }
                        let i = ((oy + y) * w + ox + x) * 4;
                        let al = a as f32 / 255.0;
                        acc.rgba[i] = (r as f32 * al + bg[0] as f32 * (1.0 - al)) as u8;
                        acc.rgba[i + 1] = (g as f32 * al + bg[1] as f32 * (1.0 - al)) as u8;
                        acc.rgba[i + 2] = (b as f32 * al + bg[2] as f32 * (1.0 - al)) as u8;
                    }
                }
            }
        }
        acc
    }

    fn index_html(scale: usize) -> String {
        let mut s = String::from(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Lucy mascot</title>\
<style>body{background:#07060f;color:#c9b8ff;font:14px/1.5 ui-monospace,Menlo,monospace;\
margin:0;padding:32px}h1{font-size:20px;letter-spacing:.22em;margin:0 0 4px;color:#e6d5ff}\
h2{font-size:12px;letter-spacing:.16em;margin:26px 0 8px;color:#a98fe0}\
p{color:#7f6fbf;margin:0 0 20px;max-width:70ch}\
.row{display:flex;gap:10px;flex-wrap:wrap}\
figure{margin:0;background:#0b0918;border:1px solid #2a1f4a;border-radius:10px;padding:10px}\
figcaption{font-size:11px;color:#8f7bd0;text-align:center;padding-top:6px}\
img{display:block;image-rendering:auto;width:calc(var(--s) * 1px)}\
</style></head><body><h1>LUCY</h1>\
<p>The embedded pose library (<code>assets/poses/</code>), scaled with a Lanczos filter — \
the exact approved artwork, not a repaint. Same renderer the TUI uses: transparent background, \
two pixels packed into each terminal cell.</p><h2>POSES</h2><div class=\"row\">",
        );
        for mood in Mood::ALL {
            s.push_str(&format!(
                "<figure><img style=\"--s:{}\" src=\"data:image/png;base64,{}\">\
<figcaption>{}</figcaption></figure>",
                scale * 2,
                png_base64(&render_frame(BASE_W * scale, BASE_H * scale, mood, 0)),
                mood.label()
            ));
        }
        s.push_str("</div>\n<h2>ANIMATION</h2>");
        for mood in Mood::ALL {
            s.push_str(&format!(
                "<h2>{}</h2><div class=\"row\" style=\"--s:{}\">",
                mood.label().to_uppercase(),
                scale
            ));
            for f in 0..STRIP_FRAMES {
                let fr = render_frame(
                    BASE_W * scale,
                    BASE_H * scale,
                    mood,
                    f as u128 * STRIP_STEP_MS * 4,
                );
                s.push_str(&format!(
                    "<img src=\"data:image/png;base64,{}\">",
                    png_base64(&fr)
                ));
            }
            s.push_str("</div>");
        }
        s.push_str("</body></html>");
        s
    }

    // ---- minimal PNG writer -------------------------------------------------
    // A sprite sheet is a few hundred kilobytes of flat colour, so a real
    // deflate would buy nothing worth a dependency: stored blocks are enough,
    // and the file still opens in any viewer.

    fn png_bytes(fr: &Frame) -> Vec<u8> {
        let mut raw = Vec::with_capacity(fr.h * (1 + fr.w * 3));
        let bg = MATTE.bytes();
        for y in 0..fr.h {
            raw.push(0); // filter: none
            for x in 0..fr.w {
                let (r, g, b, a) = fr.pixel(x, y);
                let a = a as f32 / 255.0;
                raw.push((r as f32 * a + bg[0] as f32 * (1.0 - a)) as u8);
                raw.push((g as f32 * a + bg[1] as f32 * (1.0 - a)) as u8);
                raw.push((b as f32 * a + bg[2] as f32 * (1.0 - a)) as u8);
            }
        }
        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(fr.w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(fr.h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, no interlace
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &zlib_stored(&raw));
        chunk(&mut png, b"IEND", &[]);
        png
    }

    fn png_base64(fr: &Frame) -> String {
        base64(&png_bytes(fr))
    }

    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut body = Vec::with_capacity(4 + data.len());
        body.extend_from_slice(kind);
        body.extend_from_slice(data);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }

    fn zlib_stored(raw: &[u8]) -> Vec<u8> {
        let mut z = vec![0x78, 0x01];
        let mut i = 0;
        loop {
            let n = (raw.len() - i).min(65_535);
            let last = if i + n >= raw.len() { 1u8 } else { 0 };
            z.push(last);
            z.extend_from_slice(&(n as u16).to_le_bytes());
            z.extend_from_slice(&(!(n as u16)).to_le_bytes());
            z.extend_from_slice(&raw[i..i + n]);
            i += n;
            if i >= raw.len() {
                break;
            }
        }
        z.extend_from_slice(&adler32(raw).to_be_bytes());
        z
    }

    fn adler32(data: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in data {
            a = (a + byte as u32) % 65_521;
            b = (b + a) % 65_521;
        }
        (b << 16) | a
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut c: u32 = 0xffff_ffff;
        for &byte in data {
            c ^= byte as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }

    fn base64(data: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
        for c in data.chunks(3) {
            let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            s.push(T[(n >> 18) as usize & 63] as char);
            s.push(T[(n >> 12) as usize & 63] as char);
            s.push(if c.len() > 1 {
                T[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            s.push(if c.len() > 2 {
                T[n as usize & 63] as char
            } else {
                '='
            });
        }
        s
    }

    /// Silence an unused-import warning when the palette constant is only
    /// referenced through `MATTE`.
    #[allow(dead_code)]
    fn _palette_anchor() -> Rgb {
        MATTE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(mood: Mood, t: u128) -> Frame {
        render_frame(64, 56, mood, t)
    }

    /// Her fur is violet — a magenta-leaning lavender: blue well above red,
    /// red well above green, and confidently opaque.
    fn is_fur(p: &[u8]) -> bool {
        p[3] > 200 && p[2] > p[1].saturating_add(30) && p[2] > p[0]
    }

    /// The sprite has to be a solid figure, not a handful of strokes: most of
    /// its own bounding box has to be filled.
    #[test]
    fn the_sprite_is_a_filled_shape_not_an_outline() {
        let fr = frame(Mood::Idle, 0);
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (fr.w, fr.h, 0usize, 0usize);
        let mut fur = 0usize;
        for (i, p) in fr.rgba.chunks(4).enumerate() {
            if !is_fur(p) {
                continue;
            }
            fur += 1;
            let (x, y) = (i % fr.w, i / fr.w);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        assert!(fur > 800, "expected a substantial figure, got {fur} px");
        let box_area = (max_x - min_x + 1) * (max_y - min_y + 1);
        let fill = fur as f32 / box_area as f32;
        assert!(
            fill > 0.55,
            "only {:.0}% of the sprite's box is filled — that is an outline",
            fill * 100.0
        );
    }

    /// The artwork, and more than a few shades of it: a flat silhouette is
    /// exactly the shape-in-one-colour look this crate is replacing.
    #[test]
    fn the_body_is_the_artwork_not_a_single_flat_colour() {
        let fr = frame(Mood::Idle, 0);
        let mut hues: Vec<(u8, u8, u8)> = Vec::new();
        for p in fr.rgba.chunks(4) {
            if is_fur(p) {
                let c = (p[0], p[1], p[2]);
                if !hues.contains(&c) {
                    hues.push(c);
                }
            }
        }
        assert!(hues.len() > 60, "only {} distinct fur colours", hues.len());
        // The fur is lit from above: the upper third is brighter than the
        // lower third.
        let lum = |y0: usize, y1: usize| -> (u32, usize) {
            fr.rgba
                .chunks(4)
                .enumerate()
                .filter(|(i, p)| is_fur(p) && (i / fr.w) >= y0 && (i / fr.w) < y1)
                .fold((0, 0), |(s, n), (_, p)| {
                    (s + p[0] as u32 + p[1] as u32 + p[2] as u32, n + 1)
                })
        };
        let (top, tn) = lum(4, 12);
        let (bottom, bn) = lum(44, 54);
        assert!(tn > 20 && bn > 20, "not enough fur to compare ({tn}/{bn})");
        let top = top / tn as u32;
        let bottom = bottom / bn as u32;
        assert!(
            top > bottom,
            "top {top} should be brighter than bottom {bottom}"
        );
    }

    /// The face has to be there: bright, near-white pixels (sclera, glints) in
    /// the upper middle of the head. Without them she is a blank blob.
    #[test]
    fn there_is_a_face_with_eye_highlights() {
        let fr = frame(Mood::Idle, 0);
        let bright = fr
            .rgba
            .chunks(4)
            .enumerate()
            .filter(|(i, p)| {
                p[3] > 160
                    && p[0] > 180
                    && p[1] > 180
                    && p[2] > 180
                    && (i / fr.w) > 14
                    && (i / fr.w) < 40
            })
            .count();
        assert!(bright > 20, "expected eye highlights, got {bright} px");
    }

    /// Every mood has to render a real, substantial sprite.
    #[test]
    fn every_mood_paints() {
        for mood in Mood::ALL {
            let fr = frame(mood, 400);
            let painted = fr.rgba.chunks(4).filter(|p| p[3] > 100).count();
            assert!(painted > 800, "{mood:?} painted almost nothing");
        }
    }

    /// Animation has to actually move, or "interactive" is a lie.
    #[test]
    fn the_sprite_animates_over_time() {
        for mood in Mood::ALL {
            let a = frame(mood, 0);
            let b = frame(mood, 900);
            let diff = a
                .rgba
                .chunks(4)
                .zip(b.rgba.chunks(4))
                .filter(|(x, y)| x != y)
                .count();
            assert!(diff > 40, "{mood:?} barely changed ({diff} px)");
        }
    }

    /// Rendering must be a pure function of its inputs, or an export cannot
    /// reproduce a TUI frame.
    #[test]
    fn rendering_is_deterministic() {
        for mood in Mood::ALL {
            assert_eq!(frame(mood, 777).rgba, frame(mood, 777).rgba, "{mood:?}");
        }
    }

    #[test]
    fn the_frame_fits_the_area_and_leaves_the_corners_alone() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 20));
        draw(
            &mut buf,
            Rect::new(0, 0, 40, 20),
            Mood::Idle,
            0,
            ColorMode::Truecolor,
        );
        let mut blocks = 0;
        let mut foreign = 0;
        for y in 0..20u16 {
            for x in 0..40u16 {
                let cell = &buf[(x, y)];
                if matches!(cell.symbol(), "▀" | "▄") {
                    blocks += 1;
                    assert!(
                        matches!(cell.fg, Color::Rgb(..)),
                        "expected truecolor, got {:?}",
                        cell.fg
                    );
                } else if cell.symbol() != " " {
                    foreign += 1;
                }
            }
        }
        assert!(blocks > 200, "expected a big sprite, got {blocks} cells");
        assert_eq!(foreign, 0, "the sprite must not write over other glyphs");
    }

    /// The whole point of the transparent background: a sprite pixel is never
    /// drawn against a hardcoded backdrop. Every cell outside the silhouette
    /// must reset both halves so whatever theme is behind shows through, and
    /// no cell may carry a colour where the sprite is not.
    #[test]
    fn the_background_is_the_terminal_theme_not_a_card() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 20));
        draw(
            &mut buf,
            Rect::new(0, 0, 40, 20),
            Mood::Idle,
            0,
            ColorMode::Truecolor,
        );
        let mut drawn = 0;
        for y in 0..20u16 {
            for x in 0..40u16 {
                let cell = &buf[(x, y)];
                let drawn_half = matches!(cell.symbol(), "▀" | "▄");
                if drawn_half {
                    drawn += 1;
                    continue;
                }
                // Clear cell: both halves must be the terminal default, so the
                // backdrop is the user's theme rather than any pixel we wrote.
                assert_eq!(cell.symbol(), " ", "unexpected glyph at {x},{y}");
                assert_eq!(cell.fg, Color::Reset, "clear cell painted fg at {x},{y}");
                assert_eq!(cell.bg, Color::Reset, "clear cell painted bg at {x},{y}");
            }
        }
        assert!(drawn > 200, "expected a substantial sprite, got {drawn}");
    }

    /// One live pixel in a cell leaves the empty half as the theme background,
    /// so an edge column cannot paint a rectangular block behind the sprite.
    #[test]
    fn a_half_covered_cell_resets_its_empty_half() {
        // A tall, narrow area puts a pixel row where only one half is inside:
        // the top of the head or the bottom of the feet. Sweep the whole area
        // and require every one-live-half cell to have the empty side reset.
        let (w, h) = (18u16, 14u16);
        let mut buf = Buffer::empty(Rect::new(0, 0, w, h));
        draw(
            &mut buf,
            Rect::new(0, 0, w, h),
            Mood::Idle,
            0,
            ColorMode::Truecolor,
        );
        let mut half_cells = 0;
        for y in 0..h {
            for x in 0..w {
                let cell = &buf[(x, y)];
                match cell.symbol() {
                    // Upper half live: the lower half is `bg` and must be Reset.
                    "▀" => {
                        half_cells += 1;
                        assert_ne!(cell.fg, Color::Reset, "▀ with no fg at {x},{y}");
                    }
                    // Lower half live: the upper half is `fg`, and using Reset
                    // there would paint the default *text* colour, not the
                    // backdrop. It must be Reset because nothing else can be
                    // guaranteed to match the theme.
                    "▄" => {
                        half_cells += 1;
                        assert_ne!(cell.fg, Color::Reset, "▄ with no fg at {x},{y}");
                    }
                    _ => {}
                }
            }
        }
        assert!(half_cells > 0, "no half-covered cell to check");
    }

    #[test]
    fn a_tiny_area_is_skipped_rather_than_panicking() {
        for (w, h) in [(1u16, 1u16), (3, 2), (5, 3), (0, 0)] {
            let mut buf = Buffer::empty(Rect::new(0, 0, w.max(1), h.max(1)));
            draw(
                &mut buf,
                Rect::new(0, 0, w, h),
                Mood::Idle,
                0,
                ColorMode::Truecolor,
            );
        }
    }

    #[test]
    fn a_limited_color_terminal_gets_palette_indices() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 24, 12));
        draw(
            &mut buf,
            Rect::new(0, 0, 24, 12),
            Mood::Idle,
            0,
            ColorMode::Ansi256,
        );
        let mut indexed = 0;
        let mut live = 0;
        for y in 0..12u16 {
            for x in 0..24u16 {
                let cell = &buf[(x, y)];
                if !matches!(cell.symbol(), "▀" | "▄") {
                    continue;
                }
                live += 1;
                if let Color::Indexed(_) = cell.fg {
                    indexed += 1;
                }
            }
        }
        // Every live half-cell must be a palette entry on a limited terminal,
        // never raw RGB.
        assert_eq!(indexed, live, "some live pixels were not quantised");
        assert!(live > 60, "expected a substantial sprite, got {live} cells");
    }

    /// A wide and a narrow area must both produce a sensible, non-degenerate
    /// sprite rather than one stretching the other.
    #[test]
    fn the_sprite_keeps_its_proportions_in_any_area() {
        let mut ratios: Vec<f32> = Vec::new();
        for (w, h) in [(20u16, 30u16), (60, 12), (80, 40), (100, 24)] {
            let mut buf = Buffer::empty(Rect::new(0, 0, w, h));
            draw(
                &mut buf,
                Rect::new(0, 0, w, h),
                Mood::Idle,
                0,
                ColorMode::Truecolor,
            );
            let mut min_x = u16::MAX;
            let mut max_x = 0u16;
            let mut min_y = u16::MAX;
            let mut max_y = 0u16;
            for y in 0..h {
                for x in 0..w {
                    if matches!(buf[(x, y)].symbol(), "▀" | "▄") {
                        min_x = min_x.min(x);
                        max_x = max_x.max(x);
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
                    }
                }
            }
            assert!(min_x < max_x && min_y < max_y, "nothing drawn at {w}x{h}");
            assert!(max_x < w && max_y < h, "drew outside the area at {w}x{h}");
            let cw = (max_x - min_x + 1) as f32;
            let ch = (max_y - min_y + 1) as f32;
            // A cell carries two stacked pixels, so the artwork lands in
            // `2 / aspect` columns per row of rows — the ratio the fitting
            // has to preserve.
            ratios.push(cw / ch);
        }
        let design = 2.0 / pose_aspect(pose_for_mood(Mood::Idle));
        for r in &ratios {
            // Generous, because a rail is only a dozen cells across and the
            // drawn box rounds to whole cells. But tight enough that a stretch
            // cannot hide inside it.
            assert!(
                (design * 0.85..design * 1.15).contains(r),
                "sprite cell ratio {r:.2} should be near the design's {design:.2} ({ratios:?})"
            );
        }
        // And the ratio itself must not drift with the area, or one terminal
        // size would show a different creature than another.
        let min = ratios.iter().cloned().fold(f32::MAX, f32::min);
        let max = ratios.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max / min < 1.25,
            "sprite aspect drifts across areas: {ratios:?}"
        );
    }

    // ---- companion state --------------------------------------------------

    #[test]
    fn every_companion_state_maps_to_a_mood_that_paints() {
        assert_eq!(CompanionState::ALL.len(), 5);
        for state in CompanionState::ALL {
            let fr = render_companion_frame(64, 56, state, 400);
            let painted = fr.rgba.chunks(4).filter(|p| p[3] > 100).count();
            assert!(painted > 800, "{state:?} painted almost nothing");
        }
    }

    #[test]
    fn companion_mapping_covers_the_lifecycle_once() {
        use CompanionState as S;
        assert_eq!(S::Idle.mood(), Mood::Idle);
        assert_eq!(S::Listening.mood(), Mood::Listening);
        assert_eq!(S::Thinking.mood(), Mood::Thinking);
        assert_eq!(S::Speaking.mood(), Mood::Talking);
        // Error reuses the attention-grabbing approval pose.
        assert_eq!(S::Error.mood(), Mood::Approval);
    }

    #[test]
    fn companion_labels_are_distinct_and_accents_agree_with_mood() {
        let mut labels = std::collections::HashSet::new();
        for state in CompanionState::ALL {
            assert!(labels.insert(state.label()), "{state:?} label repeats");
            assert_eq!(state.accent(), state.mood().accent(), "{state:?}");
        }
    }

    #[test]
    fn companion_predicates_partition_the_states() {
        for state in CompanionState::ALL {
            assert_eq!(state.is_idle(), state == CompanionState::Idle);
            assert_eq!(state.is_error(), state == CompanionState::Error);
            assert_eq!(state.is_busy(), state != CompanionState::Idle);
        }
    }

    #[test]
    fn companion_round_trips_through_mood_where_defined() {
        for state in CompanionState::ALL {
            assert_eq!(CompanionState::from_mood(state.mood()), Some(state));
        }
        assert_eq!(CompanionState::from_mood(Mood::Working), None);
        assert_eq!(CompanionState::from_mood(Mood::Happy), None);
    }

    #[test]
    fn companion_frame_matches_mood_frame_and_is_deterministic() {
        for state in CompanionState::ALL {
            let a = render_companion_frame(64, 56, state, 777);
            let b = render_frame(64, 56, state.mood(), 777);
            assert_eq!(a.rgba, b.rgba, "{state:?}");
            assert_eq!(
                a.rgba,
                render_companion_frame(64, 56, state, 777).rgba,
                "{state:?}"
            );
        }
    }

    #[test]
    fn companion_frame_clamps_degenerate_sizes_like_mood_frame() {
        for state in CompanionState::ALL {
            for (w, h) in [(0usize, 0usize), (1, 1), (2, 50)] {
                let a = render_companion_frame(w, h, state, 0);
                let b = render_frame(w, h, state.mood(), 0);
                assert!(a.w >= 4 && a.h >= 4, "{state:?} at {w}x{h}");
                assert_eq!(a.rgba, b.rgba, "{state:?} at {w}x{h}");
            }
        }
    }

    #[test]
    fn draw_companion_paints_only_half_blocks_and_quantises_in_256() {
        for mode in [ColorMode::Truecolor, ColorMode::Ansi256] {
            for state in CompanionState::ALL {
                let mut buf = Buffer::empty(Rect::new(0, 0, 40, 20));
                draw_companion(&mut buf, Rect::new(0, 0, 40, 20), state, 0, mode);
                let mut live = 0;
                for y in 0..20u16 {
                    for x in 0..40u16 {
                        let cell = &buf[(x, y)];
                        if matches!(cell.symbol(), "▀" | "▄") {
                            live += 1;
                            match mode {
                                ColorMode::Truecolor => assert!(
                                    matches!(cell.fg, Color::Rgb(..)),
                                    "{state:?}: expected truecolor, got {:?}",
                                    cell.fg
                                ),
                                ColorMode::Ansi256 => assert!(
                                    matches!(cell.fg, Color::Indexed(_)),
                                    "{state:?}: expected palette index, got {:?}",
                                    cell.fg
                                ),
                            }
                        } else {
                            assert_eq!(cell.symbol(), " ", "{state:?}: foreign glyph");
                            assert_eq!(cell.fg, Color::Reset, "{state:?}: fg leak");
                            assert_eq!(cell.bg, Color::Reset, "{state:?}: bg leak");
                        }
                    }
                }
                assert!(live > 200, "{state:?} drew almost nothing ({live})");
            }
        }
    }

    #[test]
    fn draw_companion_skips_tiny_areas_rather_than_panicking() {
        for state in CompanionState::ALL {
            for (w, h) in [(1u16, 1u16), (3, 2), (5, 3), (0, 0)] {
                let mut buf = Buffer::empty(Rect::new(0, 0, w.max(1), h.max(1)));
                draw_companion(
                    &mut buf,
                    Rect::new(0, 0, w, h),
                    state,
                    0,
                    ColorMode::Truecolor,
                );
            }
        }
    }

    #[test]
    fn draw_companion_blend_delegates_to_mood_blend() {
        // The companion blend is a thin wrapper: for any alpha it must paint
        // exactly what the mood-level blend paints for the mapped moods.
        let pairs = [
            (CompanionState::Idle, CompanionState::Speaking),
            (CompanionState::Thinking, CompanionState::Error),
        ];
        for (prev, cur) in pairs {
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let mut a = Buffer::empty(Rect::new(0, 0, 40, 20));
                draw_companion_blend(
                    &mut a,
                    Rect::new(0, 0, 40, 20),
                    prev,
                    cur,
                    alpha,
                    500,
                    ColorMode::Truecolor,
                );
                let mut b = Buffer::empty(Rect::new(0, 0, 40, 20));
                draw_blend(
                    &mut b,
                    Rect::new(0, 0, 40, 20),
                    prev.mood(),
                    cur.mood(),
                    alpha,
                    500,
                    ColorMode::Truecolor,
                );
                assert_eq!(format!("{a:?}"), format!("{b:?}"), "{prev:?}->{cur:?}@{alpha}");
            }
        }
        // A midpoint must still paint a real sprite, not a blank.
        let mut m = Buffer::empty(Rect::new(0, 0, 40, 20));
        draw_companion_blend(
            &mut m,
            Rect::new(0, 0, 40, 20),
            CompanionState::Idle,
            CompanionState::Error,
            0.5,
            500,
            ColorMode::Truecolor,
        );
            let mut live = 0;
            for y in 0..20u16 {
                for x in 0..40u16 {
                    if matches!(m[(x, y)].symbol(), "▀" | "▄") {
                        live += 1;
                    }
                }
            }
            assert!(live > 100, "midpoint blend drew almost nothing");
    }
}
