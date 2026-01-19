//! Wav2Vec2 ONNX Inference for Phoneme Recognition
//! 
//! This is a Rust port of the Python wav2vec-onnx-inference.py script.
//! It performs speech-to-phoneme recognition using a wav2vec2 ONNX model.
//! 
//! Supports both CLI mode and OpenAI-compatible HTTP API server.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use hound::WavReader;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, Axis};
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

// ============================================================================
// CLI Arguments
// ============================================================================

/// Wav2Vec2 ASR Phoneme Recognizer
#[derive(Parser, Debug)]
#[command(name = "asr")]
#[command(version = "1.0")]
#[command(about = "Speech-to-Phoneme recognition using Wav2Vec2 ONNX")]
struct Args {
    #[command(subcommand)]
    mode: Mode,

    /// Path to the ONNX model directory. If model files are missing,
    /// they will be automatically downloaded from HuggingFace (tyanfarm/aimate-asr-ONNX)
    #[arg(short, long, default_value = "aimate-asr-onnx", global = true)]
    model_dir: String,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Transcribe a single audio file (CLI mode)
    #[command(alias = "t")]
    Transcribe {
        /// Path to the audio file (WAV format)
        #[arg(short, long)]
        audio: String,
    },

    /// Start OpenAI-compatible HTTP API server
    #[command(alias = "o")]
    Openai {
        /// Host/IP address to bind the server to (use 0.0.0.0 for all interfaces)
        #[arg(long, default_value = "127.0.0.1")]
        ip: IpAddr,

        /// Port to listen on for HTTP requests
        #[arg(long, default_value = "3001")]
        port: u16,
    },
}

// ============================================================================
// Model Configuration
// ============================================================================

/// Configuration for ASR model downloads
#[derive(Clone)]
pub struct ASRConfig {
    pub model_url: String,
    pub vocab_url: String,
    pub config_url: String,
}

impl Default for ASRConfig {
    fn default() -> Self {
        Self {
            model_url: "https://huggingface.co/tyanfarm/aimate-asr-ONNX/resolve/main/model.onnx".into(),
            vocab_url: "https://huggingface.co/tyanfarm/aimate-asr-ONNX/resolve/main/vocab.json".into(),
            config_url: "https://huggingface.co/tyanfarm/aimate-asr-ONNX/resolve/main/config.json".into(),
        }
    }
}

/// Download a file from URL with progress bar
async fn download_file_from_url(url: &str, path: &Path) -> Result<()> {
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {:?}", parent))?;
    }

    println!("Downloading {} -> {:?}", url, path);
    
    let resp = reqwest::get(url).await
        .with_context(|| format!("Failed to request: {}", url))?;
    
    if !resp.status().is_success() {
        anyhow::bail!("Failed to download file: HTTP {}", resp.status());
    }
    
    let total_size = resp.content_length().unwrap_or(0);
    
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .unwrap()
        .progress_chars("#>-"));
    
    let mut file = tokio::fs::File::create(path).await
        .with_context(|| format!("Failed to create file: {:?}", path))?;
    
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| "Error downloading chunk")?;
        file.write_all(&chunk).await
            .with_context(|| "Failed to write chunk to file")?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }
    
    pb.finish_with_message("Download completed");
    println!("Downloaded: {:?}", path);
    
    Ok(())
}

/// Ensure all required model files exist, downloading if necessary
async fn ensure_model_files(model_dir: &Path, config: &ASRConfig) -> Result<()> {
    let model_path = model_dir.join("model.onnx");
    let vocab_path = model_dir.join("vocab.json");
    let config_path = model_dir.join("config.json");
    
    if !model_path.exists() {
        download_file_from_url(&config.model_url, &model_path).await?;
    }
    
    if !vocab_path.exists() {
        download_file_from_url(&config.vocab_url, &vocab_path).await?;
    }
    
    if !config_path.exists() {
        download_file_from_url(&config.config_url, &config_path).await?;
    }
    
    Ok(())
}

/// Model configuration from config.json
#[derive(Deserialize, Debug, Clone)]
struct ModelConfig {
    pad_token_id: i64,
    #[serde(default = "default_vocab_size")]
    #[allow(dead_code)]
    vocab_size: usize,
}

fn default_vocab_size() -> usize {
    79
}

// ============================================================================
// ASR Engine (shared between CLI and Server)
// ============================================================================

struct ASREngine {
    session: Session,
    id_to_token: HashMap<i64, String>,
    config: ModelConfig,
}

impl ASREngine {
    fn new(model_dir: &Path) -> Result<Self> {
        println!("Loading vocabulary...");
        let vocab_path = model_dir.join("vocab.json");
        let id_to_token = load_vocab(&vocab_path)?;
        println!("Loaded {} tokens", id_to_token.len());

        println!("Loading config...");
        let config_path = model_dir.join("config.json");
        let config = load_config(&config_path)?;
        println!("Pad token ID: {}", config.pad_token_id);

        println!("\nLoading ONNX model...");
        let model_path = model_dir.join("model.onnx");
        
        #[cfg(feature = "directml")]
        let session = {
            println!("Using DirectML execution provider (GPU)");
            Session::builder()?
                .with_execution_providers([ort::ep::directml::DirectML::default().build()])?
                .commit_from_file(&model_path)
                .with_context(|| format!("Failed to load model with DirectML: {:?}", model_path))?
        };
        
        #[cfg(all(feature = "cuda", not(feature = "directml")))]
        let session = {
            println!("Using CUDA execution provider (GPU)");
            Session::builder()?
                .with_execution_providers([ort::ep::cuda::CUDA::default().build()])?
                .commit_from_file(&model_path)
                .with_context(|| format!("Failed to load model with CUDA: {:?}", model_path))?
        };
        
        #[cfg(all(feature = "coreml", not(feature = "directml"), not(feature = "cuda")))]
        let session = {
            println!("Using CoreML execution provider (Apple GPU)");
            Session::builder()?
                .with_execution_providers([ort::ep::coreml::CoreML::default().build()])?
                .commit_from_file(&model_path)
                .with_context(|| format!("Failed to load model with CoreML: {:?}", model_path))?
        };
        
        #[cfg(not(any(feature = "cuda", feature = "directml", feature = "coreml")))]
        let session = {
            println!("Using CPU execution provider");
            Session::builder()?
                .commit_from_file(&model_path)
                .with_context(|| format!("Failed to load model: {:?}", model_path))?
        };
        
        println!("Model loaded successfully!");

        Ok(Self {
            session,
            id_to_token,
            config,
        })
    }

    fn transcribe_audio(&mut self, audio_samples: Vec<f32>) -> Result<String> {
        // Normalize audio
        let normalized = normalize_audio(&audio_samples);

        // Prepare input tensor [batch_size=1, sequence_length]
        let input = Array2::from_shape_vec((1, normalized.len()), normalized)
            .with_context(|| "Failed to create input array")?;

        // Run inference
        let logits = run_inference(&mut self.session, input)?;

        // Decode with CTC
        let predicted_ids = argmax_axis1(&logits);
        let phonemes = ctc_decode(&predicted_ids, &self.id_to_token, self.config.pad_token_id);

        Ok(phonemes.trim().to_string())
    }

    fn transcribe_file(&mut self, audio_path: &Path) -> Result<String> {
        println!("Loading audio: {:?}", audio_path);
        let audio = load_audio(audio_path, 16000)?;
        println!("Audio length: {} samples ({:.2}s)", audio.len(), audio.len() as f32 / 16000.0);
        
        self.transcribe_audio(audio)
    }

    fn transcribe_bytes(&mut self, audio_bytes: &[u8]) -> Result<String> {
        let audio = load_audio_from_bytes(audio_bytes, 16000)?;
        self.transcribe_audio(audio)
    }
}

// ============================================================================
// Audio Processing Functions
// ============================================================================

/// CTC Greedy Decoding
fn ctc_decode(predicted_ids: &[i64], id_to_token: &HashMap<i64, String>, pad_token_id: i64) -> String {
    // Step 1: Remove consecutive duplicates
    let mut collapsed: Vec<i64> = Vec::new();
    let mut prev_id: i64 = -1;
    
    for &idx in predicted_ids {
        if idx != prev_id {
            collapsed.push(idx);
            prev_id = idx;
        }
    }
    
    // Step 2: Remove blank/pad tokens and special tokens
    let special_tokens = ["<pad>", "<s>", "</s>", "<unk>"];
    let mut tokens: Vec<String> = Vec::new();
    
    for idx in collapsed {
        if idx != pad_token_id {
            if let Some(token) = id_to_token.get(&idx) {
                if !special_tokens.contains(&token.as_str()) {
                    tokens.push(token.clone());
                }
            }
        }
    }
    
    // Join tokens and replace '|' with space (word boundary marker)
    tokens.join("").replace('|', " ")
}

/// Load vocabulary from vocab.json
fn load_vocab(vocab_path: &Path) -> Result<HashMap<i64, String>> {
    let file = File::open(vocab_path)
        .with_context(|| format!("Failed to open vocab file: {:?}", vocab_path))?;
    let reader = BufReader::new(file);
    
    let vocab: HashMap<String, i64> = serde_json::from_reader(reader)
        .with_context(|| "Failed to parse vocab.json")?;
    
    // Invert the mapping: token -> id becomes id -> token
    let id_to_token: HashMap<i64, String> = vocab.into_iter()
        .map(|(k, v)| (v, k))
        .collect();
    
    Ok(id_to_token)
}

/// Load model configuration from config.json
fn load_config(config_path: &Path) -> Result<ModelConfig> {
    let file = File::open(config_path)
        .with_context(|| format!("Failed to open config file: {:?}", config_path))?;
    let reader = BufReader::new(file);
    
    let config: ModelConfig = serde_json::from_reader(reader)
        .with_context(|| "Failed to parse config.json")?;
    
    Ok(config)
}

/// Load and preprocess audio from WAV file
fn load_audio(audio_path: &Path, target_sample_rate: u32) -> Result<Vec<f32>> {
    let reader = WavReader::open(audio_path)
        .with_context(|| format!("Failed to open audio file: {:?}", audio_path))?;
    
    load_audio_from_reader(reader, target_sample_rate)
}

/// Load and preprocess audio from bytes
fn load_audio_from_bytes(audio_bytes: &[u8], target_sample_rate: u32) -> Result<Vec<f32>> {
    let cursor = Cursor::new(audio_bytes);
    let reader = WavReader::new(cursor)
        .with_context(|| "Failed to parse audio bytes as WAV")?;
    
    load_audio_from_reader(reader, target_sample_rate)
}

/// Common audio loading logic
fn load_audio_from_reader<R: std::io::Read>(reader: WavReader<R>, target_sample_rate: u32) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    
    // Read samples based on format
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(Result::ok)
                .collect()
        }
    };
    
    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if channels > 1 {
        samples.chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };
    
    // Resample to target sample rate if needed
    let resampled = if sample_rate != target_sample_rate {
        resample_audio(&mono_samples, sample_rate, target_sample_rate)?
    } else {
        mono_samples
    };
    
    Ok(resampled)
}

/// Resample audio using soxr (same as librosa's default soxr_hq)
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use libsoxr::Soxr;
    
    // Create soxr resampler with HQ quality (same as librosa's soxr_hq)
    let soxr = Soxr::create(
        from_rate as f64,
        to_rate as f64,
        1,  // 1 channel (mono)
        None,  // default IO spec
        None,  // default quality spec (HQ)
        None,  // default runtime spec
    ).with_context(|| "Failed to create soxr resampler")?;
    
    // Calculate output size
    let out_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
    let mut output = vec![0.0f32; out_len + 1024];  // Add buffer for safety
    
    // Resample
    let (_, out_done) = soxr.process(Some(samples), &mut output)
        .with_context(|| "Resampling failed")?;
    
    output.truncate(out_done);
    Ok(output)
}

/// Normalize audio (zero mean, unit variance)
fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let n = samples.len() as f32;
    let mean: f32 = samples.iter().sum::<f32>() / n;
    
    let variance: f32 = samples.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / n;
    let std = variance.sqrt() + 1e-5;
    
    samples.iter()
        .map(|&x| (x - mean) / std)
        .collect()
}

/// Run ONNX inference
fn run_inference(session: &mut Session, input: Array2<f32>) -> Result<Array2<f32>> {
    let shape: Vec<i64> = input.shape().iter().map(|&x| x as i64).collect();
    let (input_data, _offset): (Vec<f32>, _) = input.into_raw_vec_and_offset();
    
    let tensor = Tensor::from_array(([shape[0] as usize, shape[1] as usize], input_data))
        .with_context(|| "Failed to create input tensor")?;
    
    let outputs = session.run(ort::inputs![tensor])
        .with_context(|| "Failed to run inference")?;
    
    let output = outputs[0].try_extract_tensor::<f32>()
        .with_context(|| "Failed to extract output tensor")?;
    
    let (shape, data) = output;
    let output_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    
    if output_shape.len() == 3 {
        let seq_len = output_shape[1];
        let vocab_size = output_shape[2];
        let array = Array2::from_shape_vec((seq_len, vocab_size), data.to_vec())
            .with_context(|| "Failed to reshape output")?;
        Ok(array)
    } else if output_shape.len() == 2 {
        let array = Array2::from_shape_vec((output_shape[0], output_shape[1]), data.to_vec())
            .with_context(|| "Failed to reshape output")?;
        Ok(array)
    } else {
        anyhow::bail!("Unexpected output shape: {:?}", output_shape);
    }
}

/// Argmax along axis 1 to get predicted token IDs
fn argmax_axis1(logits: &Array2<f32>) -> Vec<i64> {
    logits.axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0)
        })
        .collect()
}

// ============================================================================
// OpenAI-Compatible HTTP Server
// ============================================================================

use axum::{
    Json, Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use tower_http::cors::CorsLayer;

type SharedASREngine = Arc<Mutex<ASREngine>>;

/// OpenAI-compatible transcription response
#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

/// OpenAI-compatible error response
#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
}

/// GET /health - Health check endpoint
async fn handle_health() -> &'static str {
    "OK"
}

/// GET /v1/models - List available models
async fn handle_models() -> impl IntoResponse {
    let models = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "wav2vec2-asr",
                "object": "model",
                "created": 1704067200,
                "owned_by": "aimate"
            }
        ]
    });
    Json(models)
}

/// POST /v1/audio/transcriptions - OpenAI Whisper-compatible endpoint
async fn handle_transcription(
    State(engine): State<SharedASREngine>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Parse multipart form data
    let mut audio_data: Option<Vec<u8>> = None;
    let mut _model: Option<String> = None;
    let mut _language: Option<String> = None;
    let mut response_format: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        
        match name.as_str() {
            "file" => {
                if let Ok(bytes) = field.bytes().await {
                    audio_data = Some(bytes.to_vec());
                }
            }
            "model" => {
                if let Ok(text) = field.text().await {
                    _model = Some(text);
                }
            }
            "language" => {
                if let Ok(text) = field.text().await {
                    _language = Some(text);
                }
            }
            "response_format" => {
                if let Ok(text) = field.text().await {
                    response_format = Some(text);
                }
            }
            _ => {}
        }
    }

    // Validate audio data
    let audio_bytes = match audio_data {
        Some(data) => data,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "No audio file provided".to_string(),
                        r#type: "invalid_request_error".to_string(),
                        code: "missing_audio".to_string(),
                    },
                }),
            ).into_response();
        }
    };

    // Transcribe
    let result = {
        let mut engine = engine.lock().await;
        engine.transcribe_bytes(&audio_bytes)
    };

    match result {
        Ok(text) => {
            // Handle different response formats (OpenAI compatibility)
            let format = response_format.as_deref().unwrap_or("json");
            
            match format {
                "text" => {
                    // Plain text response
                    text.into_response()
                }
                "verbose_json" => {
                    // Verbose JSON with additional metadata
                    let response = serde_json::json!({
                        "task": "transcribe",
                        "language": "en",
                        "duration": 0.0,
                        "text": text,
                        "words": [],
                        "segments": []
                    });
                    Json(response).into_response()
                }
                _ => {
                    // Default JSON response
                    Json(TranscriptionResponse { text }).into_response()
                }
            }
        }
        Err(e) => {
            tracing::error!("Transcription error: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Transcription failed: {}", e),
                        r#type: "processing_error".to_string(),
                        code: "transcription_failed".to_string(),
                    },
                }),
            ).into_response()
        }
    }
}

/// Create the HTTP server
async fn create_server(engine: SharedASREngine) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/v1/audio/transcriptions", post(handle_transcription))
        .route("/v1/models", get(handle_models))
        .layer(CorsLayer::permissive())
        .with_state(engine)
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let model_dir = Path::new(&args.model_dir);
    
    // Ensure model files are downloaded
    println!("Model directory: {}", model_dir.display());
    println!("Models will be auto-downloaded from HuggingFace if missing.\n");
    let asr_config = ASRConfig::default();
    ensure_model_files(model_dir, &asr_config).await?;
    
    match args.mode {
        Mode::Transcribe { audio } => {
            // CLI mode - single file transcription
            println!("=== Wav2Vec2 ASR Phoneme Recognition ===\n");
            
            let mut engine = ASREngine::new(model_dir)?;
            let audio_path = Path::new(&audio);
            
            println!("\nRunning inference...");
            let phonemes = engine.transcribe_file(audio_path)?;
            
            println!("\n========================================");
            println!("Result: {}", phonemes);
            println!("========================================");
        }
        
        Mode::Openai { ip, port } => {
            // Initialize tracing
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into())
                )
                .init();

            // Server mode - OpenAI-compatible API
            println!("=== Wav2Vec2 ASR OpenAI-Compatible Server ===\n");
            
            let engine = ASREngine::new(model_dir)?;
            let shared_engine = Arc::new(Mutex::new(engine));
            
            let addr = SocketAddr::from((ip, port));
            let app = create_server(shared_engine).await;
            
            println!("\n🚀 Server starting on http://{}", addr);
            println!("\nEndpoints:");
            println!("  POST /v1/audio/transcriptions - Transcribe audio files");
            println!("  GET  /v1/models               - List available models");
            println!("  GET  /health                  - Health check");
            println!("\nExample usage:");
            println!("  curl -X POST http://{}/v1/audio/transcriptions \\", addr);
            println!("    -F \"file=@audio.wav\" \\");
            println!("    -F \"model=wav2vec2-asr\"");
            
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
        }
    }
    
    Ok(())
}
