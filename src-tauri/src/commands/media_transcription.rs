use crate::actions::process_transcription_output;
use crate::audio_toolkit::save_wav_file;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use serde::Serialize;
use specta::Type;
use std::sync::Arc;
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

    transcription_manager.initiate_model_load();

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

    if samples.is_empty() {
        return Err("No audio extracted".to_string());
    }

    let tm = Arc::clone(&transcription_manager);
    let samples_for_wav = samples.clone();
    let transcription = tauri::async_runtime::spawn_blocking(move || tm.transcribe(samples))
        .await
        .map_err(|e| format!("Transcription task panicked: {}", e))?
        .map_err(|e| e.to_string())?;

    if transcription.is_empty() {
        return Err("Transcription is empty".to_string());
    }

    let processed =
        process_transcription_output(&app, &transcription, post_process).await;

    let file_name = format!("media-{}.wav", chrono::Utc::now().timestamp_millis());
    let wav_path = history_manager.recordings_dir().join(&file_name);
    let wav_path_verify = wav_path.clone();
    let sample_count = samples_for_wav.len();

    let wav_ok = tauri::async_runtime::spawn_blocking(move || {
        save_wav_file(&wav_path, &samples_for_wav)?;
        crate::audio_toolkit::verify_wav_file(&wav_path_verify, sample_count)?;
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("WAV save task panicked: {}", e))?
    .is_ok();

    if wav_ok {
        if let Err(err) = history_manager.save_entry(
            file_name,
            transcription.clone(),
            post_process,
            processed.post_processed_text.clone(),
            processed.post_process_prompt.clone(),
        ) {
            log::error!("Failed to save media transcription history: {}", err);
        }
    }

    Ok(MediaTranscriptionResult {
        text: processed.final_text,
    })
}
