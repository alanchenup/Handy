use crate::actions::process_transcription_output;
use crate::audio_toolkit::save_wav_file;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use log::{error, info, warn};
use serde::Serialize;
use specta::Type;
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, State};

#[derive(Serialize, Type)]
pub struct MediaTranscriptionResult {
    pub text: String,
}

/// Transcribe local media or a remote URL: ffmpeg extracts mono 16 kHz audio,
/// optional VAD (same stack as live recording), then the loaded ASR model.
#[tauri::command]
#[specta::specta]
pub async fn transcribe_media_source(
    app: AppHandle,
    source: String,
    apply_vad: bool,
    post_process: bool,
    transcription_manager: State<'_, Arc<TranscriptionManager>>,
    history_manager: State<'_, Arc<HistoryManager>>,
) -> Result<MediaTranscriptionResult, String> {
    let app_extract = app.clone();
    let source_trim = source.trim().to_string();
    if source_trim.is_empty() {
        return Err("No file or URL provided".to_string());
    }

    let source_kind = if source_trim.starts_with("http://") || source_trim.starts_with("https://") {
        "url"
    } else {
        "file"
    };
    let t_cmd = Instant::now();
    info!(
        target: "handy::media_transcribe",
        "transcribe_media_source start kind={} apply_vad={} post_process={}",
        source_kind,
        apply_vad,
        post_process
    );

    transcription_manager.initiate_model_load();

    let t_extract = Instant::now();
    let samples = tauri::async_runtime::spawn_blocking(move || {
        crate::media_transcription::extract_and_prepare_for_transcription(
            &app_extract,
            &source_trim,
            apply_vad,
        )
    })
    .await
    .map_err(|e| format!("Media extraction task panicked: {}", e))?
    .map_err(|e| e.to_string())?;
    info!(
        target: "handy::media_transcribe",
        "stage extract_blocking elapsed_ms={}",
        t_extract.elapsed().as_millis()
    );

    if samples.is_empty() {
        warn!(
            target: "handy::media_transcribe",
            "transcribe_media_source: no samples after extract (empty audio)"
        );
        return Err("No audio extracted".to_string());
    }

    let duration_s = samples.len() as f64 / 16_000.0;
    info!(
        target: "handy::media_transcribe",
        "extract done samples={} duration_sec={:.2} cmd_elapsed_so_far_ms={}",
        samples.len(),
        duration_s,
        t_cmd.elapsed().as_millis()
    );

    let tm = Arc::clone(&transcription_manager);
    let samples_for_wav = samples.clone();
    let t_asr = Instant::now();
    let transcription = tauri::async_runtime::spawn_blocking(move || tm.transcribe(samples))
        .await
        .map_err(|e| format!("Transcription task panicked: {}", e))?
        .map_err(|e| e.to_string())?;
    info!(
        target: "handy::media_transcribe",
        "stage transcribe_blocking elapsed_ms={}",
        t_asr.elapsed().as_millis()
    );

    if transcription.is_empty() {
        warn!(
            target: "handy::media_transcribe",
            "transcribe_media_source: model returned empty text"
        );
        return Err("Transcription is empty".to_string());
    }

    info!(
        target: "handy::media_transcribe",
        "transcription chars={}",
        transcription.chars().count()
    );

    let t_pp = Instant::now();
    let processed = process_transcription_output(&app, &transcription, post_process).await;
    info!(
        target: "handy::media_transcribe",
        "stage post_process elapsed_ms={}",
        t_pp.elapsed().as_millis()
    );

    let file_name = format!("media-{}.wav", chrono::Utc::now().timestamp_millis());
    let wav_path = history_manager.recordings_dir().join(&file_name);
    let wav_path_verify = wav_path.clone();
    let sample_count = samples_for_wav.len();
    info!(
        target: "handy::media_transcribe",
        "saving wav path={} samples={}",
        wav_path.display(),
        sample_count
    );

    let t_wav = Instant::now();
    let wav_save_result = tauri::async_runtime::spawn_blocking(move || {
        save_wav_file(&wav_path, &samples_for_wav).map_err(|e| e.to_string())?;
        crate::audio_toolkit::verify_wav_file(&wav_path_verify, sample_count)
            .map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("WAV save task panicked: {}", e))?;
    info!(
        target: "handy::media_transcribe",
        "stage wav_save_verify_blocking elapsed_ms={}",
        t_wav.elapsed().as_millis()
    );

    if let Err(ref e) = wav_save_result {
        error!(
            target: "handy::media_transcribe",
            "WAV save/verify failed ({}); history will still be saved",
            e
        );
    } else {
        info!(
            target: "handy::media_transcribe",
            "WAV save and verify ok file_name={}",
            file_name
        );
    }

    // Always persist history when transcription succeeded — previously we only saved when
    // WAV verification passed, which left the UI showing text but an empty history list.
    let t_hist = Instant::now();
    match history_manager.save_entry(
        file_name.clone(),
        transcription.clone(),
        post_process,
        processed.post_processed_text.clone(),
        processed.post_process_prompt.clone(),
    ) {
        Ok(entry) => {
            info!(
                target: "handy::media_transcribe",
                "history saved id={} file_name={} save_entry_elapsed_ms={}",
                entry.id,
                entry.file_name,
                t_hist.elapsed().as_millis()
            );
        }
        Err(err) => {
            error!(
                target: "handy::media_transcribe",
                "history save_entry failed after_ms={}: {}",
                t_hist.elapsed().as_millis(),
                err
            );
        }
    }

    info!(
        target: "handy::media_transcribe",
        "transcribe_media_source done final_text_chars={} total_elapsed_ms={}",
        processed.final_text.chars().count(),
        t_cmd.elapsed().as_millis()
    );

    Ok(MediaTranscriptionResult {
        text: processed.final_text,
    })
}
