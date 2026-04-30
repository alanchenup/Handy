//! Extract mono 16 kHz WAV from local files or URLs using ffmpeg,
//! optionally apply the same Silero VAD stack as live recording, then feed f32
//! samples to the existing transcription engine.

use crate::audio_toolkit::constants;
use crate::audio_toolkit::read_wav_samples;
use crate::audio_toolkit::vad::{SileroVad, SmoothedVad, VoiceActivityDetector};
use anyhow::{Context, Result};
use log::{debug, info, warn};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::{AppHandle, Manager};
use tempfile::NamedTempFile;

const SILERO_FRAME_SAMPLES: usize = (constants::WHISPER_SAMPLE_RATE * 30 / 1000) as usize; // 30 ms @ 16 kHz

fn is_http_url(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("https://") || t.starts_with("http://")
}

const FFMPEG_OUTPUT_ARGS: &[&str] = &[
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-f",
    "wav",
    "-loglevel",
    "error",
];

/// Run ffmpeg to decode `input` (file path or URL) to mono 16 kHz WAV bytes.
fn ffmpeg_to_wav_bytes(input: &str) -> Result<Vec<u8>> {
    info!(
        target: "handy::media_transcribe",
        "ffmpeg decode start input_len={}",
        input.len()
    );
    let mut child = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .args(FFMPEG_OUTPUT_ARGS)
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "Failed to spawn ffmpeg. Is ffmpeg installed and on PATH?")?;

    let mut stdout = child.stdout.take().context("ffmpeg: missing stdout")?;
    let stderr_handle = child.stderr.take();
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut err) = stderr_handle {
            let _ = std::io::Read::read_to_string(&mut err, &mut buf);
        }
        buf
    });

    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut stdout, &mut out).context("Failed to read ffmpeg stdout")?;

    let status = child.wait().context("ffmpeg wait failed")?;
    let stderr_msg = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        anyhow::bail!(
            "ffmpeg failed (status {:?}): {}",
            status.code(),
            stderr_msg.trim()
        );
    }

    info!(
        target: "handy::media_transcribe",
        "ffmpeg decode ok bytes={}",
        out.len()
    );
    Ok(out)
}

/// Stream best audio from URL via yt-dlp into ffmpeg (when yt-dlp is available).
fn ytdlp_pipe_to_wav_bytes(url: &str) -> Result<Vec<u8>> {
    info!(
        target: "handy::media_transcribe",
        "yt-dlp pipe decode start url_len={}",
        url.len()
    );
    let mut ytdlp = Command::new("yt-dlp")
        .args([
            "-f",
            "bestaudio/ba/b",
            "-o",
            "-",
            "--no-playlist",
            "--no-warnings",
            url,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| {
            "Failed to spawn yt-dlp. Install yt-dlp and add it to PATH, or use a direct media URL / local file."
        })?;

    let ytdlp_stdout = ytdlp
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("yt-dlp: missing stdout"))?;

    let mut ffmpeg = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-y")
        .arg("-i")
        .arg("pipe:0")
        .args(FFMPEG_OUTPUT_ARGS)
        .arg("pipe:1")
        .stdin(Stdio::from(ytdlp_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg for yt-dlp stream")?;

    let mut ffmpeg_out = ffmpeg.stdout.take().context("ffmpeg: missing stdout")?;
    let ffmpeg_stderr = ffmpeg.stderr.take();
    let ffmpeg_stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut err) = ffmpeg_stderr {
            let _ = std::io::Read::read_to_string(&mut err, &mut buf);
        }
        buf
    });

    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut ffmpeg_out, &mut buf).context("read ffmpeg output")?;

    let ffmpeg_status = ffmpeg.wait().context("ffmpeg wait")?;
    let ffmpeg_stderr_msg = ffmpeg_stderr_thread.join().unwrap_or_default();
    let ytdlp_status = ytdlp.wait().context("yt-dlp wait")?;

    if !ffmpeg_status.success() {
        anyhow::bail!(
            "ffmpeg failed while reading yt-dlp stream: {}",
            ffmpeg_stderr_msg.trim()
        );
    }
    if !ytdlp_status.success() {
        anyhow::bail!("yt-dlp failed (check the URL and that the site is supported)");
    }

    info!(
        target: "handy::media_transcribe",
        "yt-dlp pipe decode ok bytes={}",
        buf.len()
    );
    Ok(buf)
}

fn wav_bytes_to_f32(wav_bytes: &[u8]) -> Result<Vec<f32>> {
    let mut tmp = NamedTempFile::new().context("temp file for wav")?;
    tmp.write_all(wav_bytes)?;
    tmp.flush()?;
    let path: PathBuf = tmp.path().to_path_buf();
    let samples = read_wav_samples(&path).context("parse ffmpeg WAV output")?;
    info!(
        target: "handy::media_transcribe",
        "parsed ffmpeg wav samples={}",
        samples.len()
    );
    drop(tmp);
    Ok(samples)
}

fn resolve_vad_model_path(app: &AppHandle) -> Result<PathBuf> {
    app.path()
        .resolve(
            "resources/models/silero_vad_v4.onnx",
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| anyhow::anyhow!("Failed to resolve VAD path: {}", e))
}

/// Apply Silero + smoothed VAD identical to live recording (30 ms frames @ 16 kHz).
pub fn apply_file_vad(app: &AppHandle, samples: &[f32]) -> Result<Vec<f32>> {
    let vad_path = resolve_vad_model_path(app)?;
    let silero = SileroVad::new(&vad_path, 0.3).context("SileroVad::new")?;
    let mut smoothed = SmoothedVad::new(Box::new(silero), 15, 15, 2);

    let in_len = samples.len();
    let mut out = Vec::with_capacity(samples.len());
    let mut i = 0;
    while i + SILERO_FRAME_SAMPLES <= samples.len() {
        let frame = &samples[i..i + SILERO_FRAME_SAMPLES];
        match smoothed.push_frame(frame)? {
            crate::audio_toolkit::vad::VadFrame::Speech(buf) => out.extend_from_slice(buf),
            crate::audio_toolkit::vad::VadFrame::Noise => {}
        }
        i += SILERO_FRAME_SAMPLES;
    }
    if i < samples.len() {
        debug!(
            "VAD: appending {} trailing samples without partial frame VAD",
            samples.len() - i
        );
        out.extend_from_slice(&samples[i..]);
    }
    info!(
        target: "handy::media_transcribe",
        "file VAD in_samples={} out_samples={}",
        in_len,
        out.len()
    );
    Ok(out)
}

/// Extract mono 16 kHz f32 PCM from a local media file path.
pub fn extract_audio_from_file(path: &Path) -> Result<Vec<f32>> {
    info!(
        target: "handy::media_transcribe",
        "extract from file path={}",
        path.display()
    );
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid file path (non-UTF8)"))?;
    let wav = ffmpeg_to_wav_bytes(path_str)?;
    wav_bytes_to_f32(&wav)
}

/// Extract audio from a remote URL: try yt-dlp first, then plain ffmpeg.
pub fn extract_audio_from_url(url: &str) -> Result<Vec<f32>> {
    info!(
        target: "handy::media_transcribe",
        "extract from url len={}",
        url.len()
    );
    let wav = match ytdlp_pipe_to_wav_bytes(url) {
        Ok(b) => b,
        Err(e_ytdlp) => {
            warn!(
                target: "handy::media_transcribe",
                "yt-dlp failed: {}; trying ffmpeg -i URL",
                e_ytdlp
            );
            ffmpeg_to_wav_bytes(url.trim())?
        }
    };
    wav_bytes_to_f32(&wav)
}

pub fn extract_audio_from_source(_app: &AppHandle, source: &str) -> Result<Vec<f32>> {
    if is_http_url(source) {
        extract_audio_from_url(source.trim())
    } else {
        extract_audio_from_file(Path::new(source.trim()))
    }
}

pub fn extract_and_prepare_for_transcription(
    app: &AppHandle,
    source: &str,
    apply_vad: bool,
) -> Result<Vec<f32>> {
    info!(
        target: "handy::media_transcribe",
        "extract_and_prepare apply_vad={}",
        apply_vad
    );
    let mut samples = extract_audio_from_source(app, source)?;
    if samples.is_empty() {
        return Ok(samples);
    }
    if apply_vad {
        samples = apply_file_vad(app, &samples)?;
    }
    info!(
        target: "handy::media_transcribe",
        "extract_and_prepare done samples={}",
        samples.len()
    );
    Ok(samples)
}
