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


def synth_beats(spec, seed):
    """spec: [(bars, beats_per_bar, bpm, level)]; level 2 = tracker counts every other beat."""
    rng = np.random.default_rng(seed)
    beats, downs, t = [], [], 0.3
    for nbars, bpb, bpm, level in spec:
        for _ in range(nbars):
            downs.append(t)
            for _ in range(bpb // level):
                beats.append(round(t, 6))
                t += 60 / bpm * level * (1 + rng.normal(0, 0.008))
    return beats, [round(d, 6) for d in downs], round(t, 6)


def rhythm_cases(args):
    cases = []
    specs = [
        [(16, 4, 120, 1), (8, 4, 120, 2), (12, 3, 90, 1), (10, 4, 90, 1)],
        [(20, 4, 100, 1), (20, 4, 130, 1)],
        [(30, 4, 96, 1)],
    ]
    inputs = [synth_beats(sp, i) for i, sp in enumerate(specs)]
    # the whole sample song (its beat tracker output switches metrical level)
    import librosa
    y, _ = librosa.load(args.audio, sr=rp.SR)
    y = y.astype(np.float32)
    mel = ort.InferenceSession(f"{args.models}/beat_mel.onnx").run(None, {"audio": y})[0]
    bsess = ort.InferenceSession(f"{args.models}/beat_this.onnx")
    bl, dl = rp.beat_logits(mel, lambda x: bsess.run(None, {"spect": x}))
    b, d = rp.postprocess_beats(bl, dl)
    b = rp.fill_beat_gaps(b)
    d = rp.clean_downbeats(b, d)
    inputs.append((b.tolist(), d.tolist(), len(y) / rp.SR))
    for beats, downs, duration in inputs:
        rb, rd = rp.regularize_rhythm(np.array(beats), np.array(downs))
        cases.append({
            "beats": list(map(float, beats)), "downbeats": list(map(float, downs)), "duration": float(duration),
            "out_beats": rb.tolist(), "out_downbeats": rd.tolist(),
            "sections": [[float(s), float(e), int(i), float(t), int(m)]
                         for s, e, i, t, m in rp.rhythm_sections(rb, rd, duration)],
        })
    return cases


def key_cases():
    shapes = {"M": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "M7": [0, 4, 7, 11]}
    prog = [(0, "M"), (9, "m"), (2, "m7"), (7, "7"), (0, "M7"), (5, "M"), (7, "7"), (0, "M")]
    cases = []
    for shifts in ([0, 0, 3, 3, 0, 0, 2, 2], [0, 0, 0, 0, 0, 0], [7, 7, 7, 9, 9, 9]):
        segs, beats, downs, t = [], [], [], 0.0
        for k in shifts:
            for root, q in prog:
                r = (root + k) % 12
                tones = sorted({(r + x) % 12 for x in shapes[q]})
                segs.append([t, t + 2.0, tones, r, r])
                downs.append(t)
                beats += [t + 0.5 * j for j in range(4)]
                t += 2.0
        regions = rp.key_regions([tuple(x) for x in segs], beats, downs, t)
        cases.append({"segments": segs, "beats": beats, "downbeats": downs, "duration": t,
                      "regions": [[float(s), float(e), int(k)] for s, e, k in regions]})
    return cases


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
    beat_pen = rp.beat_change_penalties(n_frames, beats, rp.OFF_BEAT_PENALTY, rp.ON_BEAT_PENALTY)
    for name, pen, boost in [
        ("constant30", np.full(n_frames, 30.0), 0.0),
        ("beats", beat_pen, 0.0),
        ("beats_boost", beat_pen, 2.0),
    ]:
        path = rp.viterbi(rp.chord_obs(probs, cands, boost) if boost else obs, pen)
        viterbi_cases.append({
            "name": name,
            "boost": boost,
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
    data["rhythm_cases"] = rhythm_cases(args)
    data["key_cases"] = key_cases()
    with open(os.path.join(args.out, "fixtures.json"), "w") as f:
        json.dump(data, f)
    print("fixtures written to", args.out)


if __name__ == "__main__":
    main()
