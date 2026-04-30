use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use log::debug;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

/// Read a WAV file and return normalised f32 samples.
pub fn read_wav_samples<P: AsRef<Path>>(file_path: P) -> Result<Vec<f32>> {
    let reader = WavReader::open(file_path.as_ref())?;
    let samples = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<f32>, _>>()?;
    Ok(samples)
}

/// Read WAV bytes (e.g. ffmpeg stdout) from memory — avoids slow or blocking second open on disk.
pub fn read_wav_bytes(wav_bytes: &[u8]) -> Result<Vec<f32>> {
    let t0 = Instant::now();
    let reader = WavReader::new(Cursor::new(wav_bytes)).context("WavReader::new")?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let total_i16_values = reader.len() as usize;
    let frames = reader.duration() as usize;
    log::info!(
        target: "handy::media_transcribe",
        "read_wav_bytes header channels={} rate={} bits={} format={:?} total_i16_values={} frames={} open_ms={}",
        spec.channels,
        spec.sample_rate,
        spec.bits_per_sample,
        spec.sample_format,
        total_i16_values,
        frames,
        t0.elapsed().as_millis()
    );

    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        anyhow::bail!(
            "unsupported WAV format: bits={} format={:?}",
            spec.bits_per_sample,
            spec.sample_format
        );
    }

    let t1 = Instant::now();
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve(frames)
        .map_err(|e| anyhow::anyhow!("reserve f32 vec: {}", e))?;

    if ch == 1 {
        for s in reader.samples::<i16>() {
            let v = s?;
            out.push(v as f32 / i16::MAX as f32);
        }
    } else {
        let mut frame = Vec::with_capacity(ch);
        for s in reader.samples::<i16>() {
            frame.push(s?);
            if frame.len() == ch {
                let mono = frame.iter().map(|&v| v as f32).sum::<f32>() / ch as f32;
                out.push(mono / i16::MAX as f32);
                frame.clear();
            }
        }
        if !frame.is_empty() {
            anyhow::bail!(
                "WAV has incomplete final frame: {} samples in buffer, expected multiple of {}",
                frame.len(),
                ch
            );
        }
    }

    log::info!(
        target: "handy::media_transcribe",
        "read_wav_bytes collect done samples={} collect_ms={} total_ms={}",
        out.len(),
        t1.elapsed().as_millis(),
        t0.elapsed().as_millis()
    );
    Ok(out)
}

/// Verify a WAV file by reading it back and checking the sample count.
pub fn verify_wav_file<P: AsRef<Path>>(file_path: P, expected_samples: usize) -> Result<()> {
    let reader = WavReader::open(file_path.as_ref())?;
    let actual_samples = reader.len() as usize;
    if actual_samples != expected_samples {
        anyhow::bail!(
            "WAV sample count mismatch: expected {}, got {}",
            expected_samples,
            actual_samples
        );
    }
    Ok(())
}

/// Save audio samples as a WAV file
pub fn save_wav_file<P: AsRef<Path>>(file_path: P, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path.as_ref(), spec)?;

    // Convert f32 samples to i16 for WAV
    for sample in samples {
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    debug!("Saved WAV file: {:?}", file_path.as_ref());
    Ok(())
}
