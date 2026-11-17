import math
import numpy as np
from src.message import bit_errors

# Vectorized error-function complement (scipy-free; see plan note in Task 7).
_erfc = np.vectorize(math.erfc)


def qfunc(x):
    return 0.5 * _erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def ber(tx, rx):
    n, _ = bit_errors(tx, rx)
    total = min(np.asarray(tx).size, np.asarray(rx).size)
    return n / total if total else 0.0


def spectral_efficiency(bits_per_symbol: int) -> float:
    return float(bits_per_symbol)


def theoretical_ber(name: str, ebn0_db):
    ebn0 = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    if name in ("BPSK", "QPSK"):
        return qfunc(np.sqrt(2 * ebn0))
    if name == "16-QAM":
        M = 16
        k = 4
        # Standard approximation for square M-QAM BER vs Eb/N0
        arg = np.sqrt(3 * k * ebn0 / (M - 1))
        return (4 / k) * (1 - 1 / np.sqrt(M)) * qfunc(arg)
    raise ValueError(f"unknown modulation {name}")
