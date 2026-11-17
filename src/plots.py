import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import channel, metrics
from src.modulation import get_modulation


def _read_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _aggregate(rows, experiment, ch, use_hamming):
    # key: (label, ebn0) -> list of ber
    agg = defaultdict(list)
    for r in rows:
        if r["experiment"] != experiment or r["channel"] != ch:
            continue
        if use_hamming is not None and r["use_hamming"] != str(use_hamming):
            continue
        agg[(r["modulation_or_code"], float(r["ebn0_db"]))].append(float(r["ber"]))
    return agg


def plot_ber_curves(csv_path, out_dir, experiment, ch, use_hamming=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = _read_rows(csv_path)
    agg = _aggregate(rows, experiment, ch, use_hamming)
    labels = sorted({k[0] for k in agg})
    fig, ax = plt.subplots()
    for label in labels:
        pts = sorted((e, agg[(label, e)]) for (lb, e) in agg if lb == label)
        x = [e for e, _ in pts]
        mean = [np.mean(v) for _, v in pts]
        std = [np.std(v) for _, v in pts]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=label)
        if experiment == "modulation":
            xt = np.linspace(min(x), max(x), 100)
            ax.plot(xt, metrics.theoretical_ber(label, xt), "--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"{experiment} — {ch} — hamming={use_hamming}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    name = f"ber_{experiment}_{ch}_h{use_hamming}.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_constellations(out_dir, ebn0_db, rng_seed):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in ["BPSK", "QPSK", "16-QAM"]:
        mod = get_modulation(name)
        rng = np.random.default_rng(rng_seed)
        bits = rng.integers(0, 2, size=mod.bits_per_symbol * 2000, dtype=np.uint8)
        syms = mod.modulate(bits)
        noisy = channel.awgn(syms, ebn0_db, mod.bits_per_symbol, 1.0, rng)
        fig, ax = plt.subplots()
        ax.scatter(noisy.real, noisy.imag, s=4, alpha=0.3, label="após canal")
        ax.scatter(mod.constellation.real, mod.constellation.imag,
                   c="red", marker="x", s=80, label="ideal")
        ax.set_title(f"Constelação {name} @ {ebn0_db} dB")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.legend()
        path = os.path.join(out_dir, f"const_{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def tradeoff_table(csv_path, ref_ebn0_db):
    rows = _read_rows(csv_path)
    agg = defaultdict(list)
    for r in rows:
        if (r["experiment"] == "modulation" and r["channel"] == "awgn"
                and r["use_hamming"] == "False"
                and float(r["ebn0_db"]) == float(ref_ebn0_db)):
            agg[r["modulation_or_code"]].append(float(r["ber"]))
    table = []
    for mod, bers in sorted(agg.items()):
        bps = get_modulation(mod).bits_per_symbol
        table.append({
            "modulation": mod,
            "spectral_efficiency": metrics.spectral_efficiency(bps),
            "ber": float(np.mean(bers)),
        })
    return table
