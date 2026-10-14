"""
A torch (and ONNX-exportable) re-implementation of

    librosa.hybrid_cqt(y, sr=22050, hop_length=512, fmin=F#0, n_bins=288,
                       bins_per_octave=36, tuning=0)[18:270]

which is the input the ISMIR 2019 large-vocabulary ChordNet expects
(lv_chordia.extractors.cqt.CQTV2 followed by the slice in ChordNet.inference).

Design
------
librosa splits the 288 bins into
  * "pseudo" bins (filter length <= 512): STFT magnitude (n_fft=1024, hann)
    projected onto |FFT(filter)|, divided by sqrt(n_fft);
  * "full" bins: a true CQT, computed octave by octave with early
    downsampling.  The response is |<x, w_k>| / ||w_k||_1 * sqrt(len_k).

We keep the pseudo bins identical (same basis matrix, taken from librosa) and
compute the full bins with complex FIR filters applied by strided Conv1d on a
signal that is decimated by 2 per octave group.  This avoids FFT ops inside
the graph, so the exported ONNX file runs on any ONNX Runtime build.

The match against librosa is checked in verify_chordnet.py.
"""

import librosa
import librosa.core.constantq as _constantq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 22050
HOP = 512
N_BINS_TOTAL = 288
BPO = 36
FMIN = librosa.note_to_hz("F#0")
BIN_LO, BIN_HI = 18, 270  # slice used by ChordNet (C1 .. B7 + 2/3)
GROUP = 36  # bins per octave group


def _halfband(n_taps=63):
    # Kaiser windowed sinc, cutoff at a quarter of the input rate (the output Nyquist).
    n = np.arange(n_taps) - (n_taps - 1) / 2
    h = 0.5 * np.sinc(0.5 * n) * np.kaiser(n_taps, 8.0)
    return (h / h.sum()).astype(np.float32)


class HybridCQT(nn.Module):
    def __init__(self):
        super().__init__()
        freqs = librosa.cqt_frequencies(N_BINS_TOTAL, fmin=FMIN, bins_per_octave=BPO)
        alpha = librosa.filters._relative_bandwidth(freqs=freqs)
        lengths, _ = librosa.filters.wavelet_lengths(
            freqs=freqs, sr=SR, window="hann", filter_scale=1, alpha=alpha
        )
        pseudo = 2.0 ** np.ceil(np.log2(lengths)) < 2 * HOP
        first_pseudo = int(np.argmax(pseudo))  # 238 for this configuration
        assert pseudo[first_pseudo:].all() and not pseudo[:first_pseudo].any()

        # ---- pseudo-CQT part (bins first_pseudo .. BIN_HI-1) -------------------
        vqt_filter_fft = getattr(_constantq, "__vqt_filter_fft")
        fft_basis, n_fft, _ = vqt_filter_fft(
            SR, freqs[first_pseudo:], 1, 1, 0.01, hop_length=HOP, window="hann",
            alpha=alpha[first_pseudo:],
        )
        fft_basis = np.abs(fft_basis.toarray() if hasattr(fft_basis, "toarray") else fft_basis)
        fft_basis = fft_basis[: BIN_HI - first_pseudo] / np.sqrt(n_fft)
        self.n_fft = int(n_fft)
        win = librosa.filters.get_window("hann", self.n_fft, fftbins=True)
        # Only the DFT bins that the (sparse) basis actually uses are computed.
        used = np.nonzero(fft_basis.sum(axis=0) > 0)[0]
        fft_basis = fft_basis[:, used]
        k = used[:, None]
        n = np.arange(self.n_fft)[None, :]
        # Real DFT as two conv kernels (cos and -sin), window folded in.
        dft_re = (np.cos(2 * np.pi * k * n / self.n_fft) * win).astype(np.float32)
        dft_im = (-np.sin(2 * np.pi * k * n / self.n_fft) * win).astype(np.float32)
        self.register_buffer("dft", torch.from_numpy(np.concatenate([dft_re, dft_im])[:, None, :]))
        self.register_buffer("pseudo_basis", torch.from_numpy(fft_basis.astype(np.float32)))

        # ---- full CQT part (bins BIN_LO .. first_pseudo-1), octave groups -----
        self.register_buffer("halfband", torch.from_numpy(_halfband())[None, None, :])
        groups = []
        hi = first_pseudo
        g = 0
        while hi > BIN_LO:
            lo = max(BIN_LO, hi - GROUP)
            groups.append((g, lo, hi))
            hi = lo
            g += 1
        self.groups = []
        for g, lo, hi in groups:
            down = g + 1  # number of /2 decimations before this group
            sr_g = SR / 2 ** down
            hop_g = HOP // 2 ** down
            basis, lens = librosa.filters.wavelet(
                freqs=freqs[lo:hi], sr=sr_g, window="hann", filter_scale=1,
                norm=1, pad_fft=False, alpha=alpha[lo:hi],
            )
            width = basis.shape[1]
            if width % 2 == 0:  # make it odd so the filter is centred
                basis = np.pad(basis, ((0, 0), (0, 1)))
                width += 1
            # Conv1d computes correlation: y[t] = sum_n x[t+n] w[n]; we want
            # <x, conj(filter)> magnitude, and |.| is invariant to conjugation.
            scale = np.sqrt(lengths[lo:hi])[:, None]
            kern = np.concatenate([basis.real * scale, basis.imag * scale]).astype(np.float32)
            name = f"kern{g}"
            self.register_buffer(name, torch.from_numpy(kern[:, None, :]))
            self.groups.append((name, down, hop_g, width, hi - lo))
        self.n_groups = len(self.groups)

    @staticmethod
    def n_frames(n_samples):
        return 1 + n_samples // HOP

    def forward(self, y):
        """y: (N,) float32 mono audio at 22050 Hz -> (T, 252) magnitude CQT."""
        n = y.shape[0]
        t_frames = 1 + n // HOP
        x = y.reshape(1, 1, -1)

        outs = []
        # full CQT groups, from high to low frequency
        cur = x
        cur_down = 0
        half = (self.halfband.shape[-1] - 1) // 2
        for name, down, hop_g, width, nb in self.groups:
            while cur_down < down:
                cur = F.conv1d(F.pad(cur, (half, half)), self.halfband, stride=2)
                cur_down += 1
            kern = getattr(self, name)
            pad = (width - 1) // 2
            # enough right padding so that every frame centre has a full window
            need = (t_frames - 1) * hop_g + width
            right = max(need - (cur.shape[-1] + pad), 0)
            z = F.conv1d(F.pad(cur, (pad, right)), kern, stride=hop_g)[0, :, :t_frames]
            re, im = z[:nb], z[nb:]
            outs.append(torch.sqrt(re * re + im * im))
        full = torch.cat(outs[::-1], dim=0)  # low -> high

        # pseudo CQT (STFT magnitude projection)
        pad = self.n_fft // 2
        need = (t_frames - 1) * HOP + self.n_fft
        right = max(need - (n + pad), 0)
        spec = F.conv1d(F.pad(x, (pad, right)), self.dft, stride=HOP)[0, :, :t_frames]
        nb = self.pseudo_basis.shape[1]
        mag = torch.sqrt(spec[:nb] ** 2 + spec[nb:] ** 2)
        pseudo = self.pseudo_basis @ mag

        return torch.cat([full, pseudo], dim=0).transpose(0, 1)
