import numpy as np
from src.message import text_to_bits, bits_to_text
from src.pipeline import run_modulation_trial


def test_message_recovered_at_high_ebn0():
    msg = "Redes de Computadores - UNISINOS"
    bits = text_to_bits(msg)
    for mod in ["BPSK", "QPSK", "16-QAM"]:
        rng = np.random.default_rng(99)
        rx, b = run_modulation_trial(bits, mod, "awgn", 30.0, True, rng)
        assert b == 0.0
        assert bits_to_text(rx[:bits.size]) == msg
