import numpy as np


def manchester_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    out = np.empty(bits.size * 2, dtype=float)
    out[0::2] = bits            # 1 -> [1,0], 0 -> [0,1]
    out[1::2] = 1 - bits
    return out


def manchester_decode(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    first_half = signal[0::2]
    return (first_half > 0.5).astype(np.uint8)


def ami_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    out = np.zeros(bits.size, dtype=float)
    polarity = 1.0
    for i, b in enumerate(bits):
        if b == 1:
            out[i] = polarity
            polarity = -polarity
    return out


def ami_decode(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    return (np.abs(signal) > 0.5).astype(np.uint8)
