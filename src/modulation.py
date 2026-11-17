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
