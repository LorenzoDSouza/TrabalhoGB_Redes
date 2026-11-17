import numpy as np


def _n0(ebn0_db, bits_per_symbol, code_rate):
    ebn0_lin = 10 ** (ebn0_db / 10)
    return 1.0 / (ebn0_lin * bits_per_symbol * code_rate)


def awgn(symbols, ebn0_db, bits_per_symbol, code_rate, rng):
    symbols = np.asarray(symbols, dtype=complex)
    n0 = _n0(ebn0_db, bits_per_symbol, code_rate)
    sigma = np.sqrt(n0 / 2)
    noise = rng.normal(0, sigma, symbols.shape) + 1j * rng.normal(0, sigma, symbols.shape)
    return symbols + noise


def burst(symbols, ebn0_db, bits_per_symbol, code_rate, rng,
          burst_prob=0.05, burst_len=20, burst_factor=10.0):
    symbols = np.asarray(symbols, dtype=complex)
    n0 = _n0(ebn0_db, bits_per_symbol, code_rate)
    sigma = np.sqrt(n0 / 2)
    scale = np.ones(symbols.size)
    i = 0
    while i < symbols.size:
        if rng.random() < burst_prob:
            end = min(i + burst_len, symbols.size)
            scale[i:end] = burst_factor
            i = end
        else:
            i += 1
    noise = (rng.normal(0, sigma, symbols.shape) * scale
             + 1j * rng.normal(0, sigma, symbols.shape) * scale)
    return symbols + noise
