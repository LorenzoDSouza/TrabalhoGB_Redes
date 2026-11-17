import numpy as np
from src.metrics import ber, spectral_efficiency, theoretical_ber, qfunc


def test_ber_basic():
    tx = np.array([0, 1, 0, 1], dtype=np.uint8)
    rx = np.array([0, 0, 0, 0], dtype=np.uint8)
    assert ber(tx, rx) == 0.5


def test_spectral_efficiency():
    assert spectral_efficiency(1) == 1
    assert spectral_efficiency(4) == 4


def test_qfunc_known_value():
    # Q(0) = 0.5
    assert abs(qfunc(0.0) - 0.5) < 1e-9


def test_theoretical_bpsk_decreases_with_ebn0():
    ebn0 = np.array([0.0, 4.0, 8.0])
    vals = theoretical_ber("BPSK", ebn0)
    assert vals[0] > vals[1] > vals[2]
    # Known: BPSK BER at 8 dB ~ 1.9e-4
    assert abs(vals[2] - 1.9e-4) < 1e-4
