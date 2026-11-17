import numpy as np
from src.line_coding import (
    manchester_encode, manchester_decode, ami_encode, ami_decode,
)


def test_manchester_mapping():
    sig = manchester_encode(np.array([0, 1], dtype=np.uint8))
    assert list(sig) == [0.0, 1.0, 1.0, 0.0]


def test_manchester_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    assert list(manchester_decode(manchester_encode(bits))) == list(bits)


def test_ami_alternates_polarity():
    sig = ami_encode(np.array([1, 0, 1, 1], dtype=np.uint8))
    assert list(sig) == [1.0, 0.0, -1.0, 1.0]


def test_ami_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    assert list(ami_decode(ami_encode(bits))) == list(bits)
