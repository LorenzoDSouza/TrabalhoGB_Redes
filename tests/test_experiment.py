import os
import csv
import numpy as np
from src.experiment import load_config, run_sweep


def test_load_config(tmp_path):
    cfg = load_config("config.yaml")
    assert cfg["n_reps"] >= 5
    assert len(cfg["ebn0_db"]) >= 8
    assert cfg["n_bits"] >= 100000


def test_run_sweep_small(tmp_path):
    cfg = {
        "master_seed": 1, "n_bits": 800, "n_reps": 2,
        "ebn0_db": [0, 10], "message": "hi",
        "modulations": ["BPSK"], "line_codes": ["manchester"],
        "channels": ["awgn"], "use_hamming_options": [False],
        "burst": {"burst_prob": 0.05, "burst_len": 20, "burst_factor": 10.0},
    }
    out = run_sweep(cfg, str(tmp_path))
    assert os.path.exists(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    # BPSK awgn: 2 ebn0 x 2 reps = 4 ; line manchester awgn: 2x2 = 4
    assert len(rows) == 8
    assert {r["experiment"] for r in rows} == {"modulation", "line"}
