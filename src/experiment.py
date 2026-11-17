import os
import csv
import yaml
import numpy as np
from src import pipeline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sweep(config: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ber_results.csv")
    ss = np.random.SeedSequence(config["master_seed"])
    # one child seed per rep, reused across conditions for fair comparison
    rep_seeds = ss.spawn(config["n_reps"])
    rows = []
    msg_bits_target = config["n_bits"]

    for rep in range(config["n_reps"]):
        base = rep_seeds[rep]
        for ch in config["channels"]:
            ckw = config["burst"] if ch == "burst" else {}
            # modulation experiments
            for mod in config["modulations"]:
                for use_h in config["use_hamming_options"]:
                    for ebn0 in config["ebn0_db"]:
                        rng = np.random.default_rng(base.spawn(1)[0])
                        bits = np.random.default_rng(base.spawn(1)[0]).integers(
                            0, 2, size=msg_bits_target, dtype=np.uint8)
                        _, b = pipeline.run_modulation_trial(
                            bits, mod, ch, ebn0, use_h, rng, **ckw)
                        rows.append(["modulation", mod, ch, use_h, ebn0, rep, b, bits.size])
            # line-coding experiments (no hamming dimension)
            for code in config["line_codes"]:
                for ebn0 in config["ebn0_db"]:
                    rng = np.random.default_rng(base.spawn(1)[0])
                    bits = np.random.default_rng(base.spawn(1)[0]).integers(
                        0, 2, size=msg_bits_target, dtype=np.uint8)
                    _, b = pipeline.run_line_trial(bits, code, ch, ebn0, rng, **ckw)
                    rows.append(["line", code, ch, False, ebn0, rep, b, bits.size])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "modulation_or_code", "channel",
                    "use_hamming", "ebn0_db", "rep", "ber", "n_bits"])
        w.writerows(rows)
    return out_path
