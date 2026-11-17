import numpy as np
from src.channel import awgn, burst


def test_awgn_noise_variance_matches_target():
    rng = np.random.default_rng(42)
    syms = np.ones(200000, dtype=complex)  # unit energy
    ebn0_db = 6.0
    bps, R = 1, 1.0
    out = awgn(syms, ebn0_db, bps, R, rng)
    noise = out - syms
    ebn0_lin = 10 ** (ebn0_db / 10)
    expected_N0 = 1 / (ebn0_lin * bps * R)
    measured = np.var(noise)  # total complex variance ~ N0
    assert abs(measured - expected_N0) / expected_N0 < 0.05


def test_awgn_is_reproducible_with_seed():
    syms = np.ones(1000, dtype=complex)
    a = awgn(syms, 5.0, 2, 1.0, np.random.default_rng(7))
    b = awgn(syms, 5.0, 2, 1.0, np.random.default_rng(7))
    assert np.allclose(a, b)


def test_burst_adds_more_noise_than_awgn():
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    syms = np.ones(50000, dtype=complex)
    a = awgn(syms, 8.0, 2, 1.0, rng1)
    b = burst(syms, 8.0, 2, 1.0, rng2)
    assert np.var(b - syms) > np.var(a - syms)
