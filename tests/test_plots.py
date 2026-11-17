import os
import numpy as np
from src.experiment import run_sweep
from src.plots import plot_ber_curves, plot_constellations, tradeoff_table

_CFG = {
    "master_seed": 1, "n_bits": 2000, "n_reps": 2,
    "ebn0_db": [0, 6, 12], "message": "hi",
    "modulations": ["BPSK", "16-QAM"], "line_codes": ["manchester"],
    "channels": ["awgn"], "use_hamming_options": [False],
    "burst": {"burst_prob": 0.05, "burst_len": 20, "burst_factor": 10.0},
}


def test_plot_ber_curves_writes_png(tmp_path):
    csv_path = run_sweep(_CFG, str(tmp_path))
    pngs = plot_ber_curves(csv_path, str(tmp_path), "modulation", "awgn", False)
    assert pngs and all(os.path.exists(p) for p in pngs)


def test_constellations_write_png(tmp_path):
    pngs = plot_constellations(str(tmp_path), 10.0, 0)
    assert len(pngs) == 3 and all(os.path.exists(p) for p in pngs)


def test_tradeoff_table(tmp_path):
    csv_path = run_sweep(_CFG, str(tmp_path))
    table = tradeoff_table(csv_path, 12)
    names = {r["modulation"] for r in table}
    assert {"BPSK", "16-QAM"}.issubset(names)
    for r in table:
        assert "spectral_efficiency" in r and "ber" in r
