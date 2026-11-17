import numpy as np


def text_to_bits(s: str) -> np.ndarray:
    data = s.encode("ascii")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")
    bytes_ = np.packbits(bits)
    return bytes_.tobytes().decode("ascii")


def bit_errors(tx: np.ndarray, rx: np.ndarray):
    tx = np.asarray(tx, dtype=np.uint8)
    rx = np.asarray(rx, dtype=np.uint8)
    n = min(tx.size, rx.size)
    diff = tx[:n] != rx[:n]
    idx = np.nonzero(diff)[0]
    return int(idx.size), idx
