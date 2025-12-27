import torch
import torchaudio

# Spectrogram transform
spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=400,
    hop_length=160
)

def audio_to_spectrogram(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Generate spectrogram
    spec = spectrogram_transform(waveform)

    # Log-scale
    spec = torch.log(spec + 1e-9)

    # Remove channel dimension
    return spec.squeeze(0).transpose(0, 1)
