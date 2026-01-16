import json
import onnxruntime as ort
import numpy as np
import soundfile as sf

def ctc_decode(predicted_ids, id_to_token, pad_token_id):
    """CTC Greedy Decoding"""
    # Bước 1: Loại bỏ token lặp liên tiếp
    collapsed = []
    prev_id = -1
    for idx in predicted_ids:
        if idx != prev_id:
            collapsed.append(idx)
            prev_id = idx

    # Bước 2: Loại bỏ blank/pad tokens
    tokens = []
    for idx in collapsed:
        if idx != pad_token_id and idx in id_to_token:
            token = id_to_token[idx]
            # Loại bỏ các special tokens
            if token not in ['<pad>', '<s>', '</s>', '<unk>']:
                tokens.append(token)

    return ''.join(tokens).replace('|', ' ')

# Load vocab
with open('aimate-asr-onnx/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}

# Load config
with open('aimate-asr-onnx/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
pad_token_id = config.get('pad_token_id', 0)

# Load ONNX model
session = ort.InferenceSession('aimate-asr-onnx/model.onnx')

# Load audio
audio, sr = sf.read('./audio_files/In 1920, the company-america.wav')

# Resample nếu cần (thường là 16000 Hz)
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Normalize
audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-5)
input_values = np.expand_dims(audio.astype(np.float32), axis=0)

# Inference
input_name = session.get_inputs()[0].name
logits = session.run(None, {input_name: input_values})[0]

# Decode với CTC
predicted_ids = np.argmax(logits, axis=-1)[0]
phonemes = ctc_decode(predicted_ids, id_to_token, pad_token_id)

print(f"Result: {phonemes.strip()}")