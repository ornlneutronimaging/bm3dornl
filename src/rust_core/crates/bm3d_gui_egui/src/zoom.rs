//! Normal / large text-size preference, shared by every VENUS rust tool.
//!
//! Some users found the text too small, so a toolbar button toggles the whole
//! UI between the normal size and a 25% larger one (egui's zoom factor scales
//! fonts and widgets together, so nothing gets clipped). Like the theme, the
//! preference lives in one file (`~/.config/venus_rust_tools/zoom`) so
//! switching it in any of the tools switches all of them — the next time each
//! one starts. Normal size is the default.
//!
//! This module is deliberately self-contained (egui + std only) so it can be
//! copied verbatim into the other tools' crates.

use eframe::egui;
use std::path::PathBuf;

/// The zoom factor the "large text" mode applies to the whole UI.
const LARGE: f32 = 1.25;
const NORMAL: f32 = 1.0;

/// The preference file, under `$XDG_CONFIG_HOME` (or `~/.config`).
fn pref_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
        })?;
    Some(base.join("venus_rust_tools").join("zoom"))
}

/// The saved zoom factor, or the normal size when there is none
/// (or it is unreadable).
pub fn load() -> f32 {
    match pref_path().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(s) if s.trim().eq_ignore_ascii_case("large") => LARGE,
        _ => NORMAL,
    }
}

/// Persist the preference. Best effort: a read-only home directory only
/// costs the user their choice on the next start, not an error dialog.
pub fn save(factor: f32) {
    let Some(path) = pref_path() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, if factor > NORMAL { "large\n" } else { "normal\n" });
}

/// A button that toggles the whole application between the normal and the
/// large text size and saves the choice for every VENUS rust tool. Drop it
/// anywhere in a toolbar, next to the theme toggle.
pub fn toggle_button(ui: &mut egui::Ui) {
    let large_now = ui.ctx().zoom_factor() > NORMAL;
    let (label, tip, next) = if large_now {
        ("🔍−", "Switch back to the normal text size", NORMAL)
    } else {
        ("🔍+", "Make all the text larger", LARGE)
    };
    if ui
        .button(label)
        .on_hover_text(format!("{tip} (applies to all the VENUS rust tools)"))
        .clicked()
    {
        ui.ctx().set_zoom_factor(next);
        save(next);
    }
}
