import numpy as np
from src.message import text_to_bits, bits_to_text, bit_errors


def test_roundtrip_ascii():
    msg = "Hello, Redes!"
    bits = text_to_bits(msg)
    assert bits.dtype == np.uint8
    assert bits.size == len(msg) * 8
    assert set(np.unique(bits)).issubset({0, 1})
    assert bits_to_text(bits) == msg


def test_known_encoding_A():
    # 'A' = 0x41 = 0b01000001, MSB-first
    bits = text_to_bits("A")
    assert list(bits) == [0, 1, 0, 0, 0, 0, 0, 1]


def test_bit_errors_counts_and_locates():
    tx = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    rx = np.array([0, 0, 0, 1, 0], dtype=np.uint8)
    n, idx = bit_errors(tx, rx)
    assert n == 2
    assert list(idx) == [1, 4]
