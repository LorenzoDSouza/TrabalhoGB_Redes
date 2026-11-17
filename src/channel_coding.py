import numpy as np

CODE_RATE = 4 / 7

# Generator matrix G (4x7): systematic [I4 | P]. Rows map data bits d0..d3.
_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

# Parity-check matrix H (3x7): [P^T | I3].
_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)


def hamming74_encode(bits: np.ndarray):
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.size) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits.reshape(-1, 4)
    coded = (blocks @ _G) % 2
    return coded.reshape(-1).astype(np.uint8), pad


def hamming74_decode(coded: np.ndarray, n_padding: int):
    coded = np.asarray(coded, dtype=np.uint8).reshape(-1, 7)
    n_corrected = 0
    for row in coded:
        syndrome = (_H @ row) % 2
        if syndrome.any():
            # syndrome matches a column of H -> that bit is in error
            for col in range(7):
                if np.array_equal(_H[:, col], syndrome):
                    row[col] ^= 1
                    n_corrected += 1
                    break
    data = coded[:, :4].reshape(-1)
    if n_padding:
        data = data[:-n_padding]
    return data.astype(np.uint8), n_corrected
