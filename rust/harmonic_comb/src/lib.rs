use std::collections::VecDeque;
use std::f64::consts::PI;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use num_complex::Complex;
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_asyncio::tokio::future_into_py;
use realfft::RealFftPlanner;
use realfft::num_traits::Zero;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
struct HarmonicCombConfig {
    frame_size: usize,
    hop_size: usize,
    candidate_count: usize,
    harmonic_weight_count: usize,
    min_harmonics: usize,
    on_rmax: f64,
    off_rmax: f64,
    sfm_max: f64,
    on_frames: usize,
    off_frames: usize,
}

impl Default for HarmonicCombConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 1024,
            candidate_count: 36,
            harmonic_weight_count: 10,
            min_harmonics: 4,
            on_rmax: 0.001,
            off_rmax: 0.0005,
            sfm_max: 0.6,
            on_frames: 3,
            off_frames: 3,
        }
    }
}

impl HarmonicCombConfig {
    fn harmonic_weights(&self) -> Vec<f64> {
        let count = self.harmonic_weight_count.max(1);
        (1..=count).map(|i| 1.0 / i as f64).collect()
    }

    fn update_from_dict(&mut self, dict: &PyDict) -> PyResult<()> {
        macro_rules! set_if_present {
            ($key:literal, $field:ident, $convert:expr) => {
                if let Some(value) = dict.get_item($key) {
                    self.$field = $convert(value)?;
                }
            };
        }

        set_if_present!("frame_size", frame_size, |v: &PyAny| v.extract::<usize>());
        set_if_present!("hop_size", hop_size, |v: &PyAny| v.extract::<usize>());
        set_if_present!("candidate_count", candidate_count, |v: &PyAny| {
            v.extract::<usize>()
        });
        set_if_present!(
            "harmonic_weight_count",
            harmonic_weight_count,
            |v: &PyAny| v.extract::<usize>()
        );
        set_if_present!("min_harmonics", min_harmonics, |v: &PyAny| {
            v.extract::<usize>()
        });
        set_if_present!("on_rmax", on_rmax, |v: &PyAny| v.extract::<f64>());
        set_if_present!("off_rmax", off_rmax, |v: &PyAny| v.extract::<f64>());
        set_if_present!("sfm_max", sfm_max, |v: &PyAny| v.extract::<f64>());
        set_if_present!("on_frames", on_frames, |v: &PyAny| v.extract::<usize>());
        set_if_present!("off_frames", off_frames, |v: &PyAny| v.extract::<usize>());
        Ok(())
    }
}

#[derive(Debug, Error)]
enum CombError {
    #[error("no audio input device available")]
    NoInputDevice,
    #[error("requested sample rate {0} Hz is not supported by the input device")]
    UnsupportedSampleRate(u32),
    #[error("audio stream build failed: {0}")]
    StreamBuild(#[from] cpal::BuildStreamError),
    #[error("audio stream play failed: {0}")]
    StreamPlay(#[from] cpal::PlayStreamError),
    #[error("failed to query input configs: {0}")]
    ConfigQuery(#[from] cpal::SupportedStreamConfigsError),
    #[error("failed to query default input config: {0}")]
    DefaultConfig(#[from] cpal::DefaultStreamConfigError),
    #[error("no audio captured above the comb trigger thresholds")]
    NoAudioCaptured,
    #[error("FFT processing failed")]
    Fft,
}

impl From<CombError> for PyErr {
    fn from(err: CombError) -> Self {
        match err {
            CombError::UnsupportedSampleRate(_) | CombError::NoInputDevice => {
                PyErr::new::<PyValueError, _>(err.to_string())
            }
            CombError::NoAudioCaptured => PyErr::new::<PyRuntimeError, _>(err.to_string()),
            _ => PyErr::new::<PyRuntimeError, _>(err.to_string()),
        }
    }
}

struct CombAnalyzer {
    fft_forward: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    fft_input: Vec<f64>,
    fft_output: Vec<Complex<f64>>,
    window: Vec<f64>,
    weights: Vec<f64>,
    bin_width: f64,
    nyquist: f64,
}

impl CombAnalyzer {
    fn new(frame_size: usize, sample_rate: u32, weights: Vec<f64>) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(frame_size);
        let fft_output = fft_forward.make_output_vec();
        let fft_input = vec![0.0; frame_size];
        let window: Vec<f64> = (0..frame_size)
            .map(|i| {
                let n = frame_size as f64;
                0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1.0)).cos()
            })
            .collect();
        let freq_bins: Vec<f64> = (0..=frame_size / 2)
            .map(|i| i as f64 * sample_rate as f64 / frame_size as f64)
            .collect();
        let bin_width = if freq_bins.len() > 1 {
            freq_bins[1] - freq_bins[0]
        } else {
            sample_rate as f64 / 2.0
        };
        let nyquist = sample_rate as f64 / 2.0;
        Self {
            fft_forward,
            fft_input,
            fft_output,
            window,
            weights,
            bin_width,
            nyquist,
        }
    }

    fn analyze(
        &mut self,
        frame: &[f32],
        candidates: &[f64],
        min_harmonics: usize,
    ) -> Result<(f64, f64, bool), CombError> {
        if frame.len() != self.fft_input.len() {
            return Err(CombError::Fft);
        }

        for (dst, (sample, win)) in self
            .fft_input
            .iter_mut()
            .zip(frame.iter().zip(self.window.iter()))
        {
            *dst = *sample as f64 * *win;
        }

        self.fft_output.fill(Complex::zero());
        self.fft_forward
            .process(&mut self.fft_input, &mut self.fft_output)
            .map_err(|_| CombError::Fft)?;

        let magnitude: Vec<f64> = self.fft_output.iter().map(|c| c.norm()).collect();
        let eps = 1e-12;
        let magnitude_eps: Vec<f64> = magnitude.iter().map(|&m| m.max(eps)).collect();
        let geom_mean =
            magnitude_eps.iter().map(|m| m.ln()).sum::<f64>() / magnitude_eps.len() as f64;
        let geom_mean = geom_mean.exp();
        let arith_mean = magnitude_eps.iter().sum::<f64>() / magnitude_eps.len() as f64;
        let sfm = geom_mean / (arith_mean + eps);

        let magnitude_db: Vec<f64> = magnitude_eps.iter().map(|&m| 20.0 * m.log10()).collect();
        let max_mag = magnitude_eps
            .iter()
            .copied()
            .fold(0.0_f64, |acc, v| acc.max(v));

        let mut best_r = 0.0_f64;
        let mut found = false;

        for &candidate in candidates {
            if !candidate.is_finite() || candidate <= 0.0 {
                continue;
            }

            let mut harmonics: Vec<f64> = (1..=self.weights.len())
                .map(|i| candidate * i as f64)
                .collect();
            harmonics.retain(|&f| f <= self.nyquist);
            if harmonics.is_empty() {
                continue;
            }

            let local_weights = &self.weights[..harmonics.len()];

            let mut sampled = Vec::with_capacity(harmonics.len());
            for &freq in &harmonics {
                let pos = freq / self.bin_width.max(1e-12);
                let lower = pos.floor() as usize;
                let upper = pos.ceil() as usize;
                let frac = pos - lower as f64;
                let lower_val = magnitude
                    .get(lower)
                    .copied()
                    .unwrap_or_else(|| *magnitude.last().unwrap_or(&0.0));
                let upper_val = magnitude
                    .get(upper)
                    .copied()
                    .unwrap_or_else(|| *magnitude.last().unwrap_or(&0.0));
                sampled.push(lower_val + frac * (upper_val - lower_val));
            }

            if sampled.len() < min_harmonics {
                continue;
            }

            let amps_db: Vec<f64> = sampled
                .iter()
                .map(|&amp| 20.0 * amp.max(1e-12).log10())
                .collect();
            let mut floor_db = Vec::with_capacity(sampled.len());
            for &freq in &harmonics {
                let bin = (freq / self.bin_width.max(1e-12)).round() as isize;
                let mut values = Vec::new();
                for offset in -3..=3 {
                    let idx = bin + offset;
                    if idx >= 0 && (idx as usize) < magnitude_db.len() {
                        values.push(magnitude_db[idx as usize]);
                    }
                }
                if values.is_empty() {
                    values.push(0.0);
                }
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if values.len() % 2 == 1 {
                    values[values.len() / 2]
                } else {
                    let mid = values.len() / 2;
                    (values[mid - 1] + values[mid]) / 2.0
                };
                floor_db.push(median);
            }

            if amps_db
                .iter()
                .zip(floor_db.iter())
                .filter(|(amp, floor)| (*amp - *floor) >= 8.0)
                .count()
                < min_harmonics
            {
                continue;
            }

            let weight_sum: f64 = local_weights.iter().copied().sum();
            let weighted_sum: f64 = local_weights
                .iter()
                .zip(sampled.iter())
                .map(|(w, amp)| *w * amp)
                .sum();

            if weight_sum > 0.0 && max_mag > 0.0 {
                let r_value = weighted_sum / (weight_sum * max_mag);
                if r_value > best_r {
                    best_r = r_value;
                    found = true;
                }
            }
        }

        Ok((best_r, sfm, found))
    }
}

fn geomspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let log_start = start.ln();
    let log_end = end.ln();
    (0..count)
        .map(|i| {
            let t = i as f64 / (count as f64 - 1.0);
            (log_start + (log_end - log_start) * t).exp()
        })
        .collect()
}

fn build_audio_stream(
    sample_rate: u32,
    hop_size: usize,
) -> Result<(cpal::Stream, mpsc::Receiver<Vec<f32>>), CombError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(CombError::NoInputDevice)?;
    let supported = device.supported_input_configs()?;
    let mut selected = None;
    for range in supported {
        if range.channels() != 1 {
            continue;
        }
        if (range.min_sample_rate().0..=range.max_sample_rate().0).contains(&sample_rate) {
            let sample_format = range.sample_format();
            let mut cfg = range
                .with_sample_rate(cpal::SampleRate(sample_rate))
                .config();
            cfg.channels = 1;
            cfg.sample_rate = cpal::SampleRate(sample_rate);
            cfg.buffer_size = cpal::BufferSize::Fixed(hop_size as u32);
            selected = Some((sample_format, cfg));
            break;
        }
    }

    let (sample_format, config) = if let Some(pair) = selected {
        pair
    } else {
        return Err(CombError::UnsupportedSampleRate(sample_rate));
    };

    let (tx, rx) = mpsc::channel::<Vec<f32>>(128);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, tx)?,
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, tx)?,
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, tx)?,
        _ => {
            return Err(CombError::StreamBuild(
                cpal::BuildStreamError::StreamConfigNotSupported,
            ));
        }
    };

    stream.play()?;
    Ok((stream, rx))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, CombError>
where
    T: cpal::Sample + Send + 'static,
{
    let mut sender = sender;
    let err_fn = |err: cpal::StreamError| {
        eprintln!("[ERROR] Audio stream error: {err}");
    };

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            let chunk: Vec<f32> = data.iter().map(|sample| sample.to_f32()).collect();
            let _ = sender.try_send(chunk);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

async fn run_comb_trigger(
    expected_f0: f64,
    sample_rate: u32,
    max_record_seconds: f64,
    mut config: HarmonicCombConfig,
) -> Result<Vec<f32>, CombError> {
    config.frame_size = config.frame_size.max(1);
    config.hop_size = config.hop_size.max(1);
    config.candidate_count = config.candidate_count.max(1);
    config.harmonic_weight_count = config.harmonic_weight_count.max(1);
    config.min_harmonics = config.min_harmonics.max(1);
    config.on_frames = config.on_frames.max(1);
    config.off_frames = config.off_frames.max(1);

    let frame_size = config.frame_size;
    let hop = config.hop_size;

    let max_samples = (max_record_seconds * sample_rate as f64) as usize;
    if max_samples == 0 {
        return Err(CombError::NoAudioCaptured);
    }

    let nyquist = sample_rate as f64 / 2.0;

    let mut f_min = expected_f0 / 2.0;
    let f_max = (expected_f0 * 2.0).min(nyquist);
    if frame_size > 1 {
        let first_bin = sample_rate as f64 / frame_size as f64;
        f_min = f_min.max(first_bin);
    }

    if !f_min.is_finite() || !f_max.is_finite() || f_max <= f_min {
        return Err(CombError::UnsupportedSampleRate(sample_rate));
    }

    let candidates = geomspace(f_min, f_max, config.candidate_count);
    let weights = config.harmonic_weights();

    let (stream, mut rx) = build_audio_stream(sample_rate, hop)?;
    println!("[INFO] Listening for audio events (harmonic comb trigger)...");

    let mut analyzer = CombAnalyzer::new(frame_size, sample_rate, weights);

    let mut frame_buffer: Vec<f32> = Vec::new();
    let mut recent_chunks: VecDeque<Vec<f32>> = VecDeque::new();
    let mut recent_samples = 0usize;

    let mut collected: Vec<f32> = Vec::new();
    let mut collected_samples = 0usize;

    let mut on_counter = 0usize;
    let mut off_counter = 0usize;
    let mut triggered = false;

    while collected_samples < max_samples {
        let chunk = match rx.recv().await {
            Some(c) => c,
            None => break,
        };
        if chunk.is_empty() {
            continue;
        }

        frame_buffer.extend_from_slice(&chunk);
        recent_samples += chunk.len();
        recent_chunks.push_back(chunk.clone());
        while recent_samples > frame_size {
            if let Some(removed) = recent_chunks.pop_front() {
                if recent_samples >= removed.len() {
                    recent_samples -= removed.len();
                } else {
                    recent_samples = 0;
                }
            }
        }

        let was_triggered = triggered;
        let mut chunk_included = false;
        let mut stop_recording = false;

        while frame_buffer.len() >= frame_size {
            let frame = frame_buffer[..frame_size].to_vec();
            let (r_value, sfm, valid) =
                analyzer.analyze(&frame, &candidates, config.min_harmonics)?;
            let drain_len = hop.min(frame_buffer.len());
            frame_buffer.drain(..drain_len);

            if !triggered {
                if valid && r_value > config.on_rmax && sfm < config.sfm_max {
                    on_counter += 1;
                } else {
                    on_counter = 0;
                }

                if on_counter >= config.on_frames {
                    triggered = true;
                    on_counter = 0;
                    off_counter = 0;
                    chunk_included = true;

                    let mut pre_audio: Vec<f32> = Vec::new();
                    for stored in recent_chunks.iter() {
                        pre_audio.extend_from_slice(stored);
                    }
                    if !pre_audio.is_empty() {
                        collected.extend_from_slice(&pre_audio);
                        collected_samples += pre_audio.len();
                    }
                    recent_chunks.clear();
                    recent_samples = 0;
                    println!("[INFO] Recording started (harmonic comb trigger).");
                    if collected_samples >= max_samples {
                        stop_recording = true;
                        break;
                    }
                }
            } else if r_value < config.off_rmax {
                off_counter += 1;
                if off_counter >= config.off_frames {
                    triggered = false;
                    stop_recording = true;
                    println!("[INFO] Recording stopped (comb trigger released).");
                    break;
                }
            } else {
                off_counter = 0;
            }
        }

        if was_triggered && !chunk_included {
            let remaining = max_samples.saturating_sub(collected_samples);
            if remaining == 0 {
                stop_recording = true;
            } else {
                if chunk.len() > remaining {
                    collected.extend_from_slice(&chunk[..remaining]);
                    collected_samples += remaining;
                } else {
                    collected.extend_from_slice(&chunk);
                    collected_samples += chunk.len();
                }
            }
        }

        if collected_samples >= max_samples {
            println!("[WARN] Max recording length reached.");
            break;
        }

        if stop_recording {
            break;
        }
    }

    drop(stream);

    if collected.is_empty() {
        return Err(CombError::NoAudioCaptured);
    }

    Ok(collected)
}

#[pyfunction]
fn record_with_harmonic_comb<'py>(
    py: Python<'py>,
    expected_f0: f64,
    sample_rate: u32,
    max_record_seconds: f64,
    comb_cfg: Option<&PyDict>,
) -> PyResult<&'py PyAny> {
    let mut config = HarmonicCombConfig::default();
    if let Some(dict) = comb_cfg {
        config.update_from_dict(dict)?;
    }

    future_into_py(py, async move {
        let audio = run_comb_trigger(expected_f0, sample_rate, max_record_seconds, config)
            .await
            .map_err::<PyErr, _>(Into::into)?;
        Python::with_gil(|py| {
            let array = PyArray1::from_vec(py, audio);
            Ok(array.to_owned())
        })
    })
}

#[pymodule]
fn harmonic_comb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(record_with_harmonic_comb, m)?)?;
    Ok(())
}
