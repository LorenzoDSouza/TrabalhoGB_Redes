import os
import numpy as np
from src.experiment import load_config, run_sweep
from src import plots
from src.message import text_to_bits, bits_to_text
from src.pipeline import run_modulation_trial

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def main():
    cfg = load_config("config.yaml")
    print("Running BER sweep (this may take a few minutes)...")
    csv_path = run_sweep(cfg, RESULTS_DIR)
    print(f"Raw results written to {csv_path}")

    # Figures
    for ch in cfg["channels"]:
        for use_h in cfg["use_hamming_options"]:
            plots.plot_ber_curves(csv_path, FIGURES_DIR, "modulation", ch, use_h)
        plots.plot_ber_curves(csv_path, FIGURES_DIR, "line", ch, None)
    plots.plot_constellations(FIGURES_DIR, ebn0_db=10.0, rng_seed=cfg["master_seed"])

    # Trade-off table
    ref = cfg["ebn0_db"][len(cfg["ebn0_db"]) // 2]
    print(f"\nTrade-off table (AWGN, no Hamming) @ {ref} dB:")
    for row in plots.tradeoff_table(csv_path, ref):
        print(f"  {row['modulation']:>7}  "
              f"eff={row['spectral_efficiency']:.0f} bits/s/Hz  "
              f"BER={row['ber']:.2e}")

    # Consistency check
    print("\nConsistency check (Eb/N0 = 30 dB, Hamming on):")
    bits = text_to_bits(cfg["message"])
    for mod in cfg["modulations"]:
        rng = np.random.default_rng(cfg["master_seed"])
        rx, b = run_modulation_trial(bits, mod, "awgn", 30.0, True, rng)
        ok = bits_to_text(rx[:bits.size]) == cfg["message"]
        print(f"  {mod:>7}: BER={b:.1e}  recovered={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
