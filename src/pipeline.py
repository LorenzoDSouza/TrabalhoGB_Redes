import numpy as np
from src import channel, metrics
from src.modulation import get_modulation
from src.channel_coding import hamming74_encode, hamming74_decode, CODE_RATE
from src.line_coding import (
    manchester_encode, manchester_decode, ami_encode, ami_decode,
)

CHANNELS = {"awgn": channel.awgn, "burst": channel.burst}
_LINE = {
    "manchester": (manchester_encode, manchester_decode),
    "ami": (ami_encode, ami_decode),
}


def run_modulation_trial(bits, mod_name, channel_name, ebn0_db,
                         use_hamming, rng, **channel_kwargs):
    bits = np.asarray(bits, dtype=np.uint8)
    mod = get_modulation(mod_name)
    if use_hamming:
        tx_bits, pad = hamming74_encode(bits)
        code_rate = CODE_RATE
    else:
        tx_bits, pad = bits, 0
        code_rate = 1.0
    # ensure multiple of bits_per_symbol by zero-padding (tracked)
    k = mod.bits_per_symbol
    sym_pad = (-tx_bits.size) % k
    if sym_pad:
        tx_bits = np.concatenate([tx_bits, np.zeros(sym_pad, dtype=np.uint8)])
    syms = mod.modulate(tx_bits)
    noisy = CHANNELS[channel_name](syms, ebn0_db, k, code_rate, rng,
                                   **channel_kwargs)
    rx_coded = mod.demodulate(noisy)
    if sym_pad:
        rx_coded = rx_coded[:tx_bits.size - sym_pad]
    if use_hamming:
        rx_bits, _ = hamming74_decode(rx_coded, pad)
    else:
        rx_bits = rx_coded
    rx_bits = rx_bits[:bits.size]
    return rx_bits, metrics.ber(bits, rx_bits)


def run_line_trial(bits, code_name, channel_name, ebn0_db, rng, **channel_kwargs):
    bits = np.asarray(bits, dtype=np.uint8)
    enc, dec = _LINE[code_name]
    signal = enc(bits).astype(complex)  # real-valued samples as complex
    noisy = CHANNELS[channel_name](signal, ebn0_db, 1, 1.0, rng, **channel_kwargs)
    rx_bits = dec(noisy.real)
    rx_bits = rx_bits[:bits.size]
    return rx_bits, metrics.ber(bits, rx_bits)
