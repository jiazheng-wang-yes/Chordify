"""
Export the "Beat This!" beat and downbeat tracker (Foscarin, Schlueter, Widmer,
ISMIR 2024; MIT license) as two ONNX graphs:

  beat_mel.onnx   audio (N,) float32 mono 22050 Hz -> log-mel spectrogram (T, 128), 50 frames/s
  beat_this.onnx  spect (1, T, 128) -> beat (1, T), downbeat (1, T) logits

The app runs beat_this.onnx on 1500-frame chunks with a 6-frame border, exactly
like beat_this.inference.split_predict_aggregate, and peak-picks the logits
like beat_this.model.postprocessor (the "minimal" postprocessing).

Usage:
    python export_beat_this.py --beat-this /path/to/beat_this --checkpoint final0.ckpt --out-dir OUT
    python export_beat_this.py --beat-this /path/to/beat_this --checkpoint random --out-dir OUT   # smoke test
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnx_utils import compress_fp16

SR = 22050
N_FFT = 1024
HOP = 441
N_MELS = 128
F_MIN, F_MAX = 30.0, 11000.0


def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank():
    """Same as torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, norm=None, mel_scale='slaney')."""
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SR // 2, n_freqs)
    m_pts = np.linspace(_hz_to_mel_slaney(F_MIN), _hz_to_mel_slaney(F_MAX), N_MELS + 2)
    f_pts = _mel_to_hz_slaney(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    return fb.astype(np.float32)  # (513, 128)


class LogMel(nn.Module):
    """torchaudio MelSpectrogram(normalized='frame_length', power=1, center=True,
    pad_mode='reflect', periodic hann) followed by log1p(1000 * x), as in
    beat_this.preprocessing.LogMelSpect."""

    def __init__(self):
        super().__init__()
        n = np.arange(N_FFT)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic hann
        k = np.arange(N_FFT // 2 + 1)[:, None]
        scale = 1.0 / np.sqrt(N_FFT)  # torch.stft(normalized=True)
        re = np.cos(2 * np.pi * k * n[None] / N_FFT) * win * scale
        im = -np.sin(2 * np.pi * k * n[None] / N_FFT) * win * scale
        self.register_buffer("dft", torch.from_numpy(np.concatenate([re, im]).astype(np.float32)[:, None, :]))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))

    def forward(self, audio):
        x = audio.reshape(1, 1, -1)
        x = F.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
        spec = F.conv1d(x, self.dft, stride=HOP)[0]  # (1026, T)
        nb = N_FFT // 2 + 1
        mag = torch.sqrt(spec[:nb] ** 2 + spec[nb:] ** 2)
        mel = mag.transpose(0, 1) @ self.fb  # (T, 128)
        return torch.log1p(1000.0 * mel)


class Wrapped(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spect):
        out = self.model(spect)
        return out["beat"], out["downbeat"]


def load_model(beat_this_root, checkpoint):
    sys.path.insert(0, beat_this_root)
    from beat_this.model.beat_tracker import BeatThis
    if checkpoint.startswith("random"):
        dim = 128 if checkpoint == "random-small" else 512
        torch.manual_seed(0)
        return BeatThis(transformer_dim=dim).eval()
    from beat_this.inference import load_model as bt_load
    return bt_load(checkpoint, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beat-this", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mel = LogMel().eval()
    p = os.path.join(args.out_dir, "beat_mel.onnx")
    with torch.no_grad():
        torch.onnx.export(mel, (torch.randn(SR * 5),), p, input_names=["audio"], output_names=["spect"],
                          dynamic_axes={"audio": {0: "samples"}, "spect": {0: "frames"}},
                          opset_version=17, dynamo=False)
    print("wrote", p, os.path.getsize(p) / 1e6, "MB")

    model = Wrapped(load_model(args.beat_this, args.checkpoint)).eval()
    p = os.path.join(args.out_dir, "beat_this.onnx")
    tmp = p + ".tmp.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.randn(1, 1500, N_MELS),), tmp,
                          input_names=["spect"], output_names=["beat", "downbeat"],
                          dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}},
                          opset_version=17, dynamo=False)
    compress_fp16(tmp, p, keep_fp32=args.fp32)
    os.remove(tmp)
    print("wrote", p, os.path.getsize(p) / 1e6, "MB")


if __name__ == "__main__":
    main()
