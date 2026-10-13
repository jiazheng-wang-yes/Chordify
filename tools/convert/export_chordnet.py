"""
Export the ISMIR 2019 large-vocabulary chord recognizer (Jiang, Chen, Li, Xia)
as one ONNX graph:

    audio (N,) float32, mono, 22050 Hz
      -> HybridCQT (bins C1..B7, 36 bins/octave)
      -> 5 x ChordNet (CNN + BiLSTM), softmax per head
      -> mean over the 5 networks
      -> triad (T,73), bass (T,13), seventh (T,4), ninth (T,4),
         eleventh (T,3), thirteenth (T,3)

The network code and weights come from lv-chordia (MIT), which repackages
https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.

Usage:
    python export_chordnet.py --lv-chordia /path/to/lv-chordia --out ../../Chordify/Resources/chordnet.onnx
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from cqt_torch import HybridCQT
from onnx_utils import compress_fp16


def load_ensemble(lv_root):
    sys.path.insert(0, lv_root)
    from lv_chordia.chordnet_ismir_naive import ChordNet

    paths = sorted(glob.glob(os.path.join(lv_root, "weights", "*.sdict")))
    assert len(paths) == 5, paths
    nets = []
    for p in paths:
        net = ChordNet(None, use_gpu=False)
        state = torch.load(p, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        net.eval()
        nets.append(net)
    return nets


class Ensemble(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.cqt = HybridCQT()
        self.nets = nn.ModuleList(nets)

    def forward(self, audio):
        spec = self.cqt(audio)  # (T, 252)
        x = spec.unsqueeze(0)
        sums = None
        for net in self.nets:
            outs = net(x)  # six logits tensors, each (T, C)
            probs = [F.softmax(o, dim=1) for o in outs]
            sums = probs if sums is None else [a + b for a, b in zip(sums, probs)]
        n = float(len(self.nets))
        return tuple(s / n for s in sums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lv-chordia", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="keep fp32 weights in the file")
    args = ap.parse_args()

    model = Ensemble(load_ensemble(args.lv_chordia)).eval()
    dummy = torch.randn(22050 * 8)
    names = ["triad", "bass", "seventh", "ninth", "eleventh", "thirteenth"]
    tmp = args.out + ".tmp.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), tmp,
            input_names=["audio"], output_names=names,
            dynamic_axes={"audio": {0: "samples"}, **{n: {0: "frames"} for n in names}},
            opset_version=17, dynamo=False,
        )
    compress_fp16(tmp, args.out, keep_fp32=args.fp32)
    os.remove(tmp)
    print("wrote", args.out, os.path.getsize(args.out) / 1e6, "MB")


if __name__ == "__main__":
    main()
