import numpy as np
from src.pipeline import run_modulation_trial, run_line_trial


def test_modulation_ideal_channel_is_lossless():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=4000, dtype=np.uint8)
    for mod in ["BPSK", "QPSK", "16-QAM"]:
        rx, b = run_modulation_trial(bits, mod, "awgn", 30.0, False, rng)
        assert b == 0.0  # 30 dB ~ error-free


def test_modulation_higher_ebn0_lower_ber():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=20000, dtype=np.uint8)
    low = run_modulation_trial(bits, "16-QAM", "awgn", 2.0, False,
                               np.random.default_rng(1))[1]
    high = run_modulation_trial(bits, "16-QAM", "awgn", 12.0, False,
                                np.random.default_rng(1))[1]
    assert low > high


def test_line_ideal_channel_is_lossless():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=2000, dtype=np.uint8)
    for code in ["manchester", "ami"]:
        rx, b = run_line_trial(bits, code, "awgn", 30.0, rng)
        assert b == 0.0
