"""
Write reference data for the Swift unit tests (Tests/ChordifyCoreTests/Fixtures):

  chord_probs.bin   float32, the six chordnet outputs for the first 30 s of the
                    sample song, concatenated per frame (73+13+4+4+3+3 = 100 values)
  beat_logits.bin   float32, beat and downbeat logits (2 x frames) for the same audio
  fixtures.json     shapes, expected Viterbi paths, chunk starts, beat times,
                    and the expected segments of the whole pipeline
"""

import argparse
import json
import os

import librosa
import numpy as np
import onnxruntime as ort

import reference_pipeline as rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    y, _ = librosa.load(args.audio, sr=rp.SR, duration=args.seconds)
    y = y.astype(np.float32)
    probs = ort.InferenceSession(f"{args.models}/chordnet.onnx").run(None, {"audio": y})
    n_frames = len(probs[0])
    flat = np.concatenate(probs, axis=1).astype(np.float32)
    flat.tofile(os.path.join(args.out, "chord_probs.bin"))

    vocab = json.load(open(f"{args.models}/chord_vocab.json"))
    cands = rp.expand_vocab(vocab)
    names = [c[0] for c in cands]
    obs = rp.chord_obs(probs, cands)

    mel = ort.InferenceSession(f"{args.models}/beat_mel.onnx").run(None, {"audio": y})[0]
    bsess = ort.InferenceSession(f"{args.models}/beat_this.onnx")
    bl, dl = rp.beat_logits(mel, lambda x: bsess.run(None, {"spect": x}))
    np.stack([bl, dl]).astype(np.float32).tofile(os.path.join(args.out, "beat_logits.bin"))
    raw_beats, raw_downbeats = rp.postprocess_beats(bl, dl)
    beats = rp.fill_beat_gaps(raw_beats)
    downbeats = rp.clean_downbeats(beats, raw_downbeats)

    viterbi_cases = []
    for name, pen in [
        ("constant30", np.full(n_frames, 30.0)),
        ("beats", rp.beat_change_penalties(n_frames, beats, rp.OFF_BEAT_PENALTY, rp.ON_BEAT_PENALTY)),
    ]:
        path = rp.viterbi(obs, pen)
        viterbi_cases.append({
            "name": name,
            "penalty": [float(p) if np.isfinite(p) else 1e30 for p in pen],
            "segments": [[s, e, n] for s, e, n in rp.segments(path, names, rp.CHORD_FPS)],
        })

    data = {
        "seconds": args.seconds,
        "samples": int(len(y)),
        "chord_frames": int(n_frames),
        "head_sizes": [int(p.shape[1]) for p in probs],
        "n_candidates": len(cands),
        "candidate_names_head": names[:30],
        "obs_checks": [[int(t), int(c), float(obs[t, c])] for t, c in
                       [(0, 0), (10, 5), (200, 100), (n_frames - 1, len(cands) - 1), (500, 437)]],
        "viterbi": viterbi_cases,
        "beat_frames": int(len(bl)),
        "chunk_starts": {str(n): rp.split_chunks(n) for n in [100, 1488, 1489, 1501, 4000, 15000]},
        "raw_beats": raw_beats.tolist(),
        "raw_downbeats": raw_downbeats.tolist(),
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "gap_cases": [
            {"in": b, "out": rp.fill_beat_gaps(np.array(b)).tolist()}
            for b in [[0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 5.0, 5.5, 6.0],
                      [0.0, 0.5, 1.0, 1.7, 2.2, 2.7, 3.9, 4.4], [1.0, 2.0]]
        ],
        "downbeat_cases": [
            {"beats": [i * 0.5 for i in range(20)], "in": d,
             "out": rp.clean_downbeats(np.array([i * 0.5 for i in range(20)]), np.array(d)).tolist()}
            for d in [[0.0, 0.5, 2.0, 4.0, 6.0, 8.0], [0.1, 2.1, 3.0, 4.05, 6.0, 8.0], [1.0]]
        ],
        "mel_checks": [[int(t), int(f), float(mel[t, f])] for t, f in [(0, 0), (100, 64), (1000, 127)]],
    }
    with open(os.path.join(args.out, "fixtures.json"), "w") as f:
        json.dump(data, f)
    print("fixtures written to", args.out)


if __name__ == "__main__":
    main()
