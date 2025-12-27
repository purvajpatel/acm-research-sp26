import torch
from model.phode import PhoDe
from preprocessing.spectrogram import audio_to_spectrogram

model = PhoDe()
model.load_state_dict(torch.load("phode_model.pt"))
model.eval()

spec = audio_to_spectrogram("data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac")

with torch.no_grad():
    output = model(spec.unsqueeze(0))

print(output.shape)

predicted = output.argmax(dim=-1)
print(predicted[:20])
