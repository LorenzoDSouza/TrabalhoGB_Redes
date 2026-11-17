import numpy as np
from src.channel_coding import hamming74_encode, hamming74_decode, CODE_RATE


def test_code_rate():
    assert abs(CODE_RATE - 4/7) < 1e-12


def test_roundtrip_no_error():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=37, dtype=np.uint8)  # not a multiple of 4
    coded, pad = hamming74_encode(bits)
    assert coded.size % 7 == 0
    decoded, n_corr = hamming74_decode(coded, pad)
    assert list(decoded) == list(bits)
    assert n_corr == 0


def test_corrects_single_bit_error_per_block():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    coded, pad = hamming74_encode(bits)
    corrupted = coded.copy()
    corrupted[2] ^= 1  # flip one bit in the single block
    decoded, n_corr = hamming74_decode(corrupted, pad)
    assert list(decoded) == list(bits)
    assert n_corr == 1
