# Simulador de Transmissão Digital — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python simulator of a digital transmission chain (ASCII → bits → line/channel coding → modulation → noisy channel → recovery) and run BER × Eb/N0 experiments.

**Architecture:** Single-responsibility modules under `src/`, two parallel pipelines (line coding in baseband; modulation in passband with optional Hamming(7,4)). An experiment orchestrator sweeps Eb/N0 with repetitions and seeds, writes raw CSVs; a plots module regenerates figures from CSVs. Config lives in `config.yaml`.

**Tech Stack:** Python 3, NumPy, Matplotlib, PyYAML, pytest.

## Global Constraints

- Language: Python 3 (NumPy, Matplotlib, PyYAML); tests with pytest.
- Modulations: BPSK (1 bit/symbol), QPSK (2), 16-QAM (4), all with Gray mapping, unit average symbol energy.
- Line codes: Manchester (`0→[0,1]`, `1→[1,0]`, 2 samples/bit), AMI (`0→0`, `1→±1` alternating).
- Channel coding: Hamming(7,4), corrects 1 bit/block; code rate R = 4/7.
- Channels: AWGN (complex Gaussian, calibrated by Eb/N0) and burst (AWGN base + random high-noise windows).
- X-axis of BER curves: Eb/N0 in dB. Y-axis: BER on log scale.
- Experiment defaults: ≥ 8 Eb/N0 values, ≥ 100,000 bits/point, ≥ 5 repetitions, seeds derived from a master seed via `np.random.SeedSequence`.
- Bits: 8 bits/char, MSB-first.
- Reproducibility: all params in `config.yaml`; raw results in `results/*.csv` (versioned); figures regenerable into `figures/`.
- RNG: always inject a `numpy.random.Generator` into channel functions — never use global `np.random`.

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `src` package and a working `pytest` command.

- [ ] **Step 1: Create `requirements.txt`**

```
numpy>=1.24
matplotlib>=3.7
PyYAML>=6.0
pytest>=7.0
```

- [ ] **Step 2: Create empty package markers**

Create `src/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v
```

- [ ] **Step 4: Install dependencies and verify pytest runs**

Run: `python3 -m pip install -r requirements.txt && python3 -m pytest`
Expected: pytest runs and reports "no tests ran" (exit cleanly, no import errors).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py pytest.ini
git commit -m "chore: project scaffolding and dependencies"
```

---

### Task 2: Message module (ASCII ↔ bits)

**Files:**
- Create: `src/message.py`
- Test: `tests/test_message.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `text_to_bits(s: str) -> np.ndarray` — 1-D uint8 array of 0/1, 8 bits/char, MSB-first.
  - `bits_to_text(bits: np.ndarray) -> str` — inverse of `text_to_bits`.
  - `bit_errors(tx: np.ndarray, rx: np.ndarray) -> tuple[int, np.ndarray]` — returns (count, indices where they differ).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.message import text_to_bits, bits_to_text, bit_errors

def test_roundtrip_ascii():
    msg = "Hello, Redes!"
    bits = text_to_bits(msg)
    assert bits.dtype == np.uint8
    assert bits.size == len(msg) * 8
    assert set(np.unique(bits)).issubset({0, 1})
    assert bits_to_text(bits) == msg

def test_known_encoding_A():
    # 'A' = 0x41 = 0b01000001, MSB-first
    bits = text_to_bits("A")
    assert list(bits) == [0, 1, 0, 0, 0, 0, 0, 1]

def test_bit_errors_counts_and_locates():
    tx = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    rx = np.array([0, 0, 0, 1, 0], dtype=np.uint8)
    n, idx = bit_errors(tx, rx)
    assert n == 2
    assert list(idx) == [1, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_message.py -v`
Expected: FAIL with "No module named 'src.message'" / "cannot import".

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np

def text_to_bits(s: str) -> np.ndarray:
    data = s.encode("ascii")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)

def bits_to_text(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")
    bytes_ = np.packbits(bits)
    return bytes_.tobytes().decode("ascii")

def bit_errors(tx: np.ndarray, rx: np.ndarray):
    tx = np.asarray(tx, dtype=np.uint8)
    rx = np.asarray(rx, dtype=np.uint8)
    n = min(tx.size, rx.size)
    diff = tx[:n] != rx[:n]
    idx = np.nonzero(diff)[0]
    return int(idx.size), idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_message.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/message.py tests/test_message.py
git commit -m "feat: ASCII <-> bits conversion and bit-error locator"
```

---

### Task 3: Line coding (Manchester, AMI)

**Files:**
- Create: `src/line_coding.py`
- Test: `tests/test_line_coding.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `manchester_encode(bits: np.ndarray) -> np.ndarray` — float array, 2 samples/bit, values in {0.0, 1.0}; `0→[0,1]`, `1→[1,0]`.
  - `manchester_decode(signal: np.ndarray) -> np.ndarray` — inverse, uint8 bits.
  - `ami_encode(bits: np.ndarray) -> np.ndarray` — float array, 1 sample/bit; `0→0.0`, `1→±1.0` alternating polarity, first `1→+1`.
  - `ami_decode(signal: np.ndarray) -> np.ndarray` — threshold decode (`|x|>0.5 → 1`), uint8 bits.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.line_coding import (
    manchester_encode, manchester_decode, ami_encode, ami_decode,
)

def test_manchester_mapping():
    sig = manchester_encode(np.array([0, 1], dtype=np.uint8))
    assert list(sig) == [0.0, 1.0, 1.0, 0.0]

def test_manchester_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    assert list(manchester_decode(manchester_encode(bits))) == list(bits)

def test_ami_alternates_polarity():
    sig = ami_encode(np.array([1, 0, 1, 1], dtype=np.uint8))
    assert list(sig) == [1.0, 0.0, -1.0, 1.0]

def test_ami_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    assert list(ami_decode(ami_encode(bits))) == list(bits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_line_coding.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np

def manchester_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    out = np.empty(bits.size * 2, dtype=float)
    out[0::2] = bits            # 1 -> [1,0], 0 -> [0,1]
    out[1::2] = 1 - bits
    return out

def manchester_decode(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    first_half = signal[0::2]
    return (first_half > 0.5).astype(np.uint8)

def ami_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    out = np.zeros(bits.size, dtype=float)
    polarity = 1.0
    for i, b in enumerate(bits):
        if b == 1:
            out[i] = polarity
            polarity = -polarity
    return out

def ami_decode(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    return (np.abs(signal) > 0.5).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_line_coding.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/line_coding.py tests/test_line_coding.py
git commit -m "feat: Manchester and AMI line coding"
```

---

### Task 4: Channel coding (Hamming 7,4)

**Files:**
- Create: `src/channel_coding.py`
- Test: `tests/test_channel_coding.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `hamming74_encode(bits: np.ndarray) -> tuple[np.ndarray, int]` — returns (coded uint8 bits, n_padding_bits added to make input a multiple of 4). Each 4-bit block → 7 coded bits.
  - `hamming74_decode(coded: np.ndarray, n_padding: int) -> tuple[np.ndarray, int]` — returns (decoded uint8 bits with padding removed, n_corrected single-bit errors).
  - `CODE_RATE = 4/7` module constant.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.channel_coding import hamming74_encode, hamming74_decode, CODE_RATE

def test_code_rate():
    assert abs(CODE_RATE - 4/7) < 1e-12

def test_roundtrip_no_error():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=37, dtype=np.uint8)  # not a multiple of 4
    coded, pad = hamming74_encode(bits)
    assert coded.size % 7 == 0
    decoded, n_corr = hamming74_decode(coded, pad)
    assert list(decoded) == list(bits)
    assert n_corr == 0

def test_corrects_single_bit_error_per_block():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    coded, pad = hamming74_encode(bits)
    corrupted = coded.copy()
    corrupted[2] ^= 1  # flip one bit in the single block
    decoded, n_corr = hamming74_decode(corrupted, pad)
    assert list(decoded) == list(bits)
    assert n_corr == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_channel_coding.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np

CODE_RATE = 4 / 7

# Generator matrix G (4x7): systematic [I4 | P]. Rows map data bits d0..d3.
_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

# Parity-check matrix H (3x7): [P^T | I3].
_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)

def hamming74_encode(bits: np.ndarray):
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.size) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits.reshape(-1, 4)
    coded = (blocks @ _G) % 2
    return coded.reshape(-1).astype(np.uint8), pad

def hamming74_decode(coded: np.ndarray, n_padding: int):
    coded = np.asarray(coded, dtype=np.uint8).reshape(-1, 7)
    n_corrected = 0
    for row in coded:
        syndrome = (_H @ row) % 2
        if syndrome.any():
            # syndrome matches a column of H -> that bit is in error
            for col in range(7):
                if np.array_equal(_H[:, col], syndrome):
                    row[col] ^= 1
                    n_corrected += 1
                    break
    data = coded[:, :4].reshape(-1)
    if n_padding:
        data = data[:-n_padding]
    return data.astype(np.uint8), n_corrected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_channel_coding.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/channel_coding.py tests/test_channel_coding.py
git commit -m "feat: Hamming(7,4) channel coding with single-error correction"
```

---

### Task 5: Modulation (BPSK, QPSK, 16-QAM)

**Files:**
- Create: `src/modulation.py`
- Test: `tests/test_modulation.py`

**Interfaces:**
- Consumes: nothing.
- Produces a `Modulation` protocol-like interface. Each modulation is an object with:
  - `name: str`, `bits_per_symbol: int`, `constellation: np.ndarray` (complex, unit average energy).
  - `modulate(bits: np.ndarray) -> np.ndarray` (complex symbols; pads bits to a multiple of `bits_per_symbol` with zeros — note: experiment code must use bit counts that are multiples so no padding occurs).
  - `demodulate(symbols: np.ndarray) -> np.ndarray` (uint8 bits, min-distance hard decision).
  - Factory: `get_modulation(name: str) -> Modulation` for names `"BPSK"`, `"QPSK"`, `"16-QAM"`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.modulation import get_modulation

def test_bits_per_symbol():
    assert get_modulation("BPSK").bits_per_symbol == 1
    assert get_modulation("QPSK").bits_per_symbol == 2
    assert get_modulation("16-QAM").bits_per_symbol == 4

def test_unit_average_energy():
    for name in ["BPSK", "QPSK", "16-QAM"]:
        c = get_modulation(name).constellation
        assert abs(np.mean(np.abs(c) ** 2) - 1.0) < 1e-9

def test_modulate_demodulate_noiseless_is_lossless():
    rng = np.random.default_rng(1)
    for name in ["BPSK", "QPSK", "16-QAM"]:
        mod = get_modulation(name)
        n = mod.bits_per_symbol * 1000
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        syms = mod.modulate(bits)
        assert syms.size == n // mod.bits_per_symbol
        out = mod.demodulate(syms)
        assert list(out) == list(bits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_modulation.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np

class _Modulation:
    def __init__(self, name, bits_per_symbol, symbol_map):
        self.name = name
        self.bits_per_symbol = bits_per_symbol
        # symbol_map: dict tuple(bits) -> complex, already unit-energy
        self._symbol_map = symbol_map
        self._keys = list(symbol_map.keys())
        self.constellation = np.array([symbol_map[k] for k in self._keys])

    def modulate(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        k = self.bits_per_symbol
        pad = (-bits.size) % k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        groups = bits.reshape(-1, k)
        syms = np.array([self._symbol_map[tuple(g)] for g in groups])
        return syms

    def demodulate(self, symbols):
        symbols = np.asarray(symbols)
        # min-distance decision
        dists = np.abs(symbols[:, None] - self.constellation[None, :])
        nearest = np.argmin(dists, axis=1)
        out = []
        for idx in nearest:
            out.extend(self._keys[idx])
        return np.array(out, dtype=np.uint8)

def _bpsk():
    return _Modulation("BPSK", 1, {(0,): -1 + 0j, (1,): 1 + 0j})

def _qpsk():
    s = 1 / np.sqrt(2)
    # Gray-coded
    return _Modulation("QPSK", 2, {
        (0, 0): complex(-s, -s), (0, 1): complex(-s, s),
        (1, 1): complex(s, s),  (1, 0): complex(s, -s),
    })

def _qam16():
    # Gray-coded 4-PAM levels per axis: bits (b0,b1)->level
    gray = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    norm = np.sqrt(10)  # avg energy of 16-QAM with levels {-3,-1,1,3} is 10
    smap = {}
    for i_bits, i_lvl in gray.items():
        for q_bits, q_lvl in gray.items():
            key = i_bits + q_bits  # 4 bits: I first, then Q
            smap[key] = complex(i_lvl, q_lvl) / norm
    return _Modulation("16-QAM", 4, smap)

_FACTORY = {"BPSK": _bpsk, "QPSK": _qpsk, "16-QAM": _qam16}

def get_modulation(name: str):
    return _FACTORY[name]()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_modulation.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/modulation.py tests/test_modulation.py
git commit -m "feat: BPSK/QPSK/16-QAM with Gray mapping, unit energy"
```

---

### Task 6: Channel (AWGN, burst)

**Files:**
- Create: `src/channel.py`
- Test: `tests/test_channel.py`

**Interfaces:**
- Consumes: nothing (operates on complex symbol arrays).
- Produces:
  - `awgn(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int, code_rate: float, rng: np.random.Generator) -> np.ndarray` — adds complex Gaussian noise. Noise variance per complex sample: `N0 = 1 / (ebn0_linear * bits_per_symbol * code_rate)` for unit-energy symbols; real & imag each get `N0/2`.
  - `burst(symbols, ebn0_db, bits_per_symbol, code_rate, rng, burst_prob=0.05, burst_len=20, burst_factor=10.0) -> np.ndarray` — AWGN base plus windows of amplified noise.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.channel import awgn, burst

def test_awgn_noise_variance_matches_target():
    rng = np.random.default_rng(42)
    syms = np.ones(200000, dtype=complex)  # unit energy
    ebn0_db = 6.0
    bps, R = 1, 1.0
    out = awgn(syms, ebn0_db, bps, R, rng)
    noise = out - syms
    ebn0_lin = 10 ** (ebn0_db / 10)
    expected_N0 = 1 / (ebn0_lin * bps * R)
    measured = np.var(noise)  # total complex variance ~ N0
    assert abs(measured - expected_N0) / expected_N0 < 0.05

def test_awgn_is_reproducible_with_seed():
    syms = np.ones(1000, dtype=complex)
    a = awgn(syms, 5.0, 2, 1.0, np.random.default_rng(7))
    b = awgn(syms, 5.0, 2, 1.0, np.random.default_rng(7))
    assert np.allclose(a, b)

def test_burst_adds_more_noise_than_awgn():
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    syms = np.ones(50000, dtype=complex)
    a = awgn(syms, 8.0, 2, 1.0, rng1)
    b = burst(syms, 8.0, 2, 1.0, rng2)
    assert np.var(b - syms) > np.var(a - syms)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_channel.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_channel.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/channel.py tests/test_channel.py
git commit -m "feat: AWGN and burst-noise channels calibrated by Eb/N0"
```

---

### Task 7: Metrics (BER, spectral efficiency, theoretical BER)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `src.message.bit_errors`.
- Produces:
  - `ber(tx: np.ndarray, rx: np.ndarray) -> float`.
  - `spectral_efficiency(bits_per_symbol: int) -> float` (nominal bits/s/Hz = bits_per_symbol).
  - `theoretical_ber(name: str, ebn0_db: np.ndarray) -> np.ndarray` — AWGN closed-form: BPSK/QPSK `Q(sqrt(2·Eb/N0))`; 16-QAM standard M-QAM approximation.
  - `qfunc(x) -> np.ndarray` helper.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np
from scipy.special import erfc  # if scipy unavailable, use math.erfc via np.vectorize
from src.message import bit_errors

def qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))

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
```

Note: add `scipy>=1.10` to `requirements.txt` in this task's commit (used only for `erfc`). If avoiding scipy is preferred, replace with `np.vectorize(math.erfc)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pip install scipy && python3 -m pytest tests/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py requirements.txt
git commit -m "feat: BER, spectral efficiency, theoretical AWGN BER curves"
```

---

### Task 8: Pipelines (line + modulation end-to-end)

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `line_coding`, `channel_coding`, `modulation`, `channel`, `metrics`.
- Produces:
  - `run_modulation_trial(bits, mod_name, channel_name, ebn0_db, use_hamming, rng, **channel_kwargs) -> tuple[np.ndarray, float]` — returns (rx_bits, ber). Chain: `[hamming encode] → modulate → channel → demodulate → [hamming decode] → ber`.
  - `run_line_trial(bits, code_name, channel_name, ebn0_db, rng, **channel_kwargs) -> tuple[np.ndarray, float]` — baseband: encode → add real AWGN/burst to samples → decode → ber. (Line codes carry 1 bit/sample effective after decode; treat samples as unit-energy real symbols with bits_per_symbol=1, code_rate=1.)
  - `CHANNELS = {"awgn": channel.awgn, "burst": channel.burst}`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from src.pipeline import run_modulation_trial, run_line_trial

def test_modulation_ideal_channel_is_lossless():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=4000, dtype=np.uint8)
    for mod in ["BPSK", "QPSK", "16-QAM"]:
        rx, b = run_modulation_trial(bits, mod, "awgn", 30.0, False, rng)
        assert b == 0.0  # 30 dB ~ error-free

def test_modulation_higher_ebn0_lower_ber():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=20000, dtype=np.uint8)
    low = run_modulation_trial(bits, "16-QAM", "awgn", 2.0, False,
                               np.random.default_rng(1))[1]
    high = run_modulation_trial(bits, "16-QAM", "awgn", 12.0, False,
                                np.random.default_rng(1))[1]
    assert low > high

def test_line_ideal_channel_is_lossless():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=2000, dtype=np.uint8)
    for code in ["manchester", "ami"]:
        rx, b = run_line_trial(bits, code, "awgn", 30.0, rng)
        assert b == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pipeline.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end modulation and line-coding pipelines"
```

---

### Task 9: Config loading and experiment orchestration

**Files:**
- Create: `config.yaml`
- Create: `src/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `pipeline`, `metrics`, PyYAML.
- Produces:
  - `load_config(path: str) -> dict`.
  - `run_sweep(config: dict, out_dir: str) -> str` — runs all configured experiments, writes one tidy CSV `results/ber_results.csv` with columns: `experiment,modulation_or_code,channel,use_hamming,ebn0_db,rep,ber,n_bits`. Returns CSV path.
  - Seeds derived via `np.random.SeedSequence(master_seed).spawn(...)`.

- [ ] **Step 1: Create `config.yaml`**

```yaml
master_seed: 20261116
n_bits: 100000
n_reps: 5
ebn0_db: [0, 2, 4, 6, 8, 10, 12, 14]
message: "Redes de Computadores - UNISINOS - Trabalho II"
modulations: ["BPSK", "QPSK", "16-QAM"]
line_codes: ["manchester", "ami"]
channels: ["awgn", "burst"]
use_hamming_options: [false, true]
burst:
  burst_prob: 0.05
  burst_len: 20
  burst_factor: 10.0
```

- [ ] **Step 2: Write the failing tests**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_experiment.py -v`
Expected: FAIL with import error.

- [ ] **Step 4: Write minimal implementation**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_experiment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/experiment.py tests/test_experiment.py
git commit -m "feat: config loading and BER sweep orchestration with seeded reps"
```

---

### Task 10: Plots (BER×SNR, constellations, trade-off table)

**Files:**
- Create: `src/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: CSV from `experiment`, `metrics`, `modulation`, Matplotlib (Agg backend).
- Produces:
  - `plot_ber_curves(csv_path, out_dir, experiment, channel, use_hamming=None) -> list[str]` — BER×Eb/N0 with mean + std error bars per modulation/code, log-y, theoretical overlay for modulation experiments. Returns list of saved PNG paths.
  - `plot_constellations(out_dir, ebn0_db, rng_seed) -> list[str]` — before/after channel scatter for BPSK/QPSK/16-QAM.
  - `tradeoff_table(csv_path, ref_ebn0_db) -> list[dict]` — rows of {modulation, spectral_efficiency, ber} at reference Eb/N0 (AWGN, no Hamming).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plots.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plots.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/plots.py tests/test_plots.py
git commit -m "feat: BER curves, constellations, and trade-off table plots"
```

---

### Task 11: Main entrypoint and consistency test

**Files:**
- Create: `main.py`
- Test: `tests/test_consistency.py`

**Interfaces:**
- Consumes: `experiment`, `plots`, `message`, `pipeline`.
- Produces:
  - `main.py` CLI: `python3 main.py` runs the full sweep from `config.yaml`, writes `results/ber_results.csv`, generates all figures into `figures/`, prints the trade-off table, and runs the consistency check (message reconstructed error-free at high Eb/N0).
  - `src.pipeline` already supports an end-to-end message trial; the consistency test uses `run_modulation_trial` with the real ASCII message.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_consistency.py -v`
Expected: initially may fail if `rx` length/padding handling is off; fix `run_modulation_trial` trimming if needed so it PASSES.

- [ ] **Step 3: Write `main.py`**

```python
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
```

- [ ] **Step 4: Run full test suite and the entrypoint**

Run: `python3 -m pytest && python3 main.py`
Expected: all tests PASS; `main.py` prints trade-off table and consistency = OK for all modulations; `figures/` populated; `results/ber_results.csv` created.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_consistency.py results/ber_results.csv
git commit -m "feat: main entrypoint, figure generation, consistency check"
```

---

### Task 12: README and documentation

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the finished codebase.
- Produces: README with architecture, install, usage, parameters, reproducibility, line-vs-channel-coding explanation, results discussion placeholders, and a dupla-responsibility placeholder.

- [ ] **Step 1: Write `README.md`**

Include these sections (fill real content; placeholders only where marked):
- **Título e objetivo** — resumo do trabalho.
- **Arquitetura do simulador** — diagrama ASCII dos dois pipelines e tabela de módulos (`src/*`).
- **Instalação** — `python3 -m pip install -r requirements.txt`.
- **Execução** — `python3 main.py`; onde saem CSV e figuras.
- **Parâmetros** — tabela explicando cada chave de `config.yaml`.
- **Reprodutibilidade** — seed-mestre, seeds por repetição, CSVs versionados.
- **Codificação de linha × codificação de canal** — o texto explicativo exigido pelo enunciado.
- **Resultados e discussão** — espaço para colar figuras e a discussão BER × eficiência espectral (placeholder: `<!-- preencher após rodar os experimentos -->`).
- **Divisão de responsabilidades da dupla** — placeholder: `<!-- Integrante A: ... / Integrante B: ... -->`.
- **Vídeo** — placeholder para o link (a ser adicionado depois).

- [ ] **Step 2: Verify the README renders and links are correct**

Run: `python3 -c "import pathlib; print('README.md' in [p.name for p in pathlib.Path('.').iterdir()])"`
Expected: prints `True`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with architecture, usage, reproducibility, and theory"
```

---

## Self-Review

**Spec coverage:**
- ASCII→bits → Task 2 ✓
- ≥2 line codes → Task 3 (Manchester, AMI) ✓
- line vs channel coding explanation → Task 12 README ✓
- ≥3 modulations incl. higher order → Task 5 (BPSK/QPSK/16-QAM) ✓
- AWGN + SNR control → Task 6 ✓
- 2nd channel (burst) → Task 6 ✓
- demod/decode/reconstruct/identify wrong bits → Tasks 2 (`bit_errors`), 8 ✓
- BER + spectral efficiency → Task 7 ✓
- ≥8 Eb/N0, ≥100k bits, ≥5 reps → Task 9 config + Task 1 defaults ✓
- BER×SNR curves with variability → Task 10 (error bars) ✓
- compare AWGN vs burst → Tasks 9/10/11 (both channels swept and plotted) ✓
- trade-off table → Task 10 ✓
- consistency test → Task 11 ✓
- reproducibility (config, seeds, raw results, scripts) → Tasks 1/9/11 ✓
- optional theoretical curves → Task 7/10 ✓
- optional constellations → Task 10 ✓
- Hamming(7,4) with/without comparison → Tasks 4/8/9/10 ✓

**Placeholder scan:** No "TBD/implement later" in code steps. README placeholders are intentional (dupla split, results discussion, video link) — these require human/experiment input, flagged explicitly.

**Type consistency:** Function names/signatures consistent across tasks (`run_modulation_trial`, `run_line_trial`, `theoretical_ber`, `get_modulation`, `hamming74_encode/decode`, `awgn/burst`, `ber/spectral_efficiency`). CSV columns in Task 9 match readers in Task 10. `code_rate` passed to channel matches `CODE_RATE` from Task 4. Note added in Task 7: scipy added to requirements.
