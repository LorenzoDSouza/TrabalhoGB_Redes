import numpy as np
from src.modulation import get_modulation


def test_bits_per_symbol():
    assert get_modulation("BPSK").bits_per_symbol == 1
    assert get_modulation("QPSK").bits_per_symbol == 2
    assert get_modulation("16-QAM").bits_per_symbol == 4


def test_unit_average_energy():
    for name in ["BPSK", "QPSK", "16-QAM"]:
        c = get_modulation(name).constellation
        assert abs(np.mean(np.abs(c) ** 2) - 1.0) < 1e-9


def test_modulate_demodulate_noiseless_is_lossless():
    rng = np.random.default_rng(1)
    for name in ["BPSK", "QPSK", "16-QAM"]:
        mod = get_modulation(name)
        n = mod.bits_per_symbol * 1000
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        syms = mod.modulate(bits)
        assert syms.size == n // mod.bits_per_symbol
        out = mod.demodulate(syms)
        assert list(out) == list(bits)
