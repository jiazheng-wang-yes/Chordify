"""
Check chordnet.onnx against the original lv-chordia path
(librosa hybrid CQT -> 5 torch ChordNets -> mean of softmax).

    python verify_chordnet.py --lv-chordia /path/to/lv-chordia --onnx chordnet.onnx --audio song.wav
"""

import argparse
import sys
import time

import librosa
import numpy as np
import onnxruntime as ort
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lv-chordia", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    sys.path.insert(0, args.lv_chordia)
    from export_chordnet import load_ensemble

    y, _ = librosa.load(args.audio, sr=22050, offset=args.offset, duration=args.duration)
    y = y.astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    t0 = time.time()
    onnx_out = sess.run(None, {"audio": y})
    print(f"onnx: {time.time() - t0:.2f}s for {len(y) / 22050:.0f}s of audio")

    nets = load_ensemble(args.lv_chordia)
    for tuning in [0.0, None]:
        cqt = np.abs(librosa.hybrid_cqt(
            y, sr=22050, hop_length=512, fmin=librosa.note_to_hz("F#0"), n_bins=288,
            bins_per_octave=36, tuning=tuning)).T.astype(np.float32)
        with torch.no_grad():
            per_net = [net.inference(torch.from_numpy(cqt)) for net in nets]
        ref = [np.mean([p[i] for p in per_net], axis=0) for i in range(6)]
        label = "tuning=0" if tuning == 0.0 else "tuning=estimated"
        for name, a, b in zip(["triad", "bass", "7", "9", "11", "13"], onnx_out, ref):
            agree = np.mean(a.argmax(1) == b.argmax(1))
            print(f"{label:17s} {name:6s} max|diff|={np.abs(a - b).max():.4f} argmax agree={agree:.4f}")


if __name__ == "__main__":
    main()
