"""
Check beat_mel.onnx / beat_this.onnx against the original beat_this code
(torchaudio log-mel + torch model), including chunk lengths other than 1500.

    python verify_beat_this.py --beat-this /path/to/beat_this --checkpoint final0.ckpt --dir OUT --audio song.wav
"""

import argparse
import sys

import librosa
import numpy as np
import onnxruntime as ort
import torch

from export_beat_this import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beat-this", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.beat_this)
    from beat_this.preprocessing import LogMelSpect

    y, _ = librosa.load(args.audio, sr=22050, duration=40)
    y = y.astype(np.float32)
    ref_mel = LogMelSpect()(torch.from_numpy(y)).numpy()
    mel = ort.InferenceSession(f"{args.dir}/beat_mel.onnx").run(None, {"audio": y})[0]
    print("mel shape", mel.shape, ref_mel.shape, "max|diff|", np.abs(mel - ref_mel).max(),
          "max", ref_mel.max())

    model = load_model(args.beat_this, args.checkpoint)
    sess = ort.InferenceSession(f"{args.dir}/beat_this.onnx")
    for t in [1500, 777]:
        x = ref_mel[None, :t].astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))
        b, d = sess.run(None, {"spect": x})
        print(f"T={t}: beat max|diff|={np.abs(b - ref['beat'].numpy()).max():.4f}, "
              f"downbeat max|diff|={np.abs(d - ref['downbeat'].numpy()).max():.4f}, "
              f"peak agree={np.mean((b > 0) == (ref['beat'].numpy() > 0)):.4f}")


if __name__ == "__main__":
    main()
