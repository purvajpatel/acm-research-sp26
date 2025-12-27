import numpy as np
from scipy.signal import butter, lfilter

def bandpass(signal, low, high, fs):
    b, a = butter(4, [low/fs*2, high/fs*2], btype='band')
    return lfilter(b, a, signal)

def ci_vocoder(signal, fs, bands=16):
    output = np.zeros_like(signal)
    band_edges = np.linspace(200, 8000, bands + 1)

    for i in range(bands):
        band = bandpass(signal, band_edges[i], band_edges[i+1], fs)
        envelope = np.abs(band)
        noise = np.random.randn(len(signal))
        output += envelope * noise

    return output / np.max(np.abs(output))
