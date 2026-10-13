"""
Python reference of the analysis pipeline that the Swift app implements
(Sources/ChordifyCore).  It is used to
  * compare the decoder with lv-chordia's XHMMDecoder,
  * try decoding settings on real songs,
  * write the fixtures for the Swift unit tests.

    python reference_pipeline.py --models ../../Models --audio song.mp3
"""

import argparse
import json
import time

import numpy as np

SR = 22050
CHORD_FPS = SR / 512  # 43.07 frames/s
BEAT_FPS = 50.0
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------- vocabulary
def expand_vocab(vocab):
    """All (root, type) candidates plus N, as model index arrays, like XHMMDecoder."""
    cands = [("N", -1, -1, [0, 0, -1, -1, -1, -1])]  # triad 0 (=N), bass N (index 0 after +1)
    seen = set()
    for ti, t in enumerate(vocab["types"]):
        tri, bass, s7, s9, s11, s13 = t["model"]
        for root in range(12):
            tri_r = (tri - 1) % 12  # always 0 for root C, kept general
            tri_idx = ((tri - 1) // 12) * 12 + (tri_r + root) % 12 + 1
            bass_idx = (bass + root) % 12 + 1  # +1 because index 0 is "no bass"
            arr = (tri_idx, bass_idx, s7, s9, s11, s13)
            if arr in seen:
                continue
            seen.add(arr)
            cands.append((f"{NOTE_NAMES[root]}:{t['harte']}", root, ti, list(arr)))
    return cands


def chord_obs(probs, cands):
    """log-likelihood of every candidate at every frame (T, C)."""
    eps = 1e-12
    logs = [np.log(np.maximum(p, eps)) for p in probs]
    arr = np.array([c[3] for c in cands])
    obs = logs[0][:, arr[:, 0]] + logs[1][:, arr[:, 1]]
    for i in range(4):
        col = arr[:, 2 + i]
        ok = col >= 0
        obs[:, ok] += logs[2 + i][:, col[ok]]
    return obs


def viterbi(obs, change_penalty):
    """change_penalty[t] is the cost of switching chord between frame t-1 and t
    (np.inf = not allowed).  The path starts in state 0 (N), like XHMMDecoder."""
    n_frames, n_states = obs.shape
    back = np.zeros((n_frames, n_states), dtype=np.int32)
    score = np.full(n_states, -np.inf)
    score[0] = 0.0
    score = score + obs[0]
    for t in range(1, n_frames):
        best = int(np.argmax(score))
        switch = score[best] - change_penalty[t]
        stay = score > switch
        back[t] = np.where(stay, np.arange(n_states), best)
        score = np.where(stay, score, switch) + obs[t]
    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def segments(path, names, fps):
    out = []
    start = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            out.append((start / fps, t / fps, names[path[start]]))
            start = t
    return out


# ---------------------------------------------------------------- beats
def split_chunks(n, chunk=1500, border=6):
    starts = list(range(-border, n - border, chunk - 2 * border))
    if n > chunk - 2 * border:
        starts[-1] = n - (chunk - border)
    return starts


def beat_logits(spect, run_model, chunk=1500, border=6):
    n = len(spect)
    beat = np.full(n, -1000.0, np.float32)
    down = np.full(n, -1000.0, np.float32)
    starts = split_chunks(n, chunk, border)
    # keep_first: later chunks are written first, earlier ones overwrite them
    for s in reversed(starts):
        lo, hi = max(s, 0), min(s + chunk, n)
        x = spect[lo:hi]
        pad_l = max(0, -s)
        pad_r = max(0, min(border, s + chunk - n))
        x = np.pad(x, ((pad_l, pad_r), (0, 0)))
        b, d = run_model(x[None].astype(np.float32))
        b, d = b[0][border:-border], d[0][border:-border]
        a = s + border
        e = min(a + len(b), n)
        beat[a:e] = b[: e - a]
        down[a:e] = d[: e - a]
    return beat, down


def pick_peaks(logits):
    n = len(logits)
    padded = np.pad(logits, 3, constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    frames = np.nonzero((logits == win) & (logits > 0))[0]
    # deduplicate adjacent peaks (beat_this.model.postprocessor.deduplicate_peaks)
    out = []
    if len(frames) == 0:
        return np.array(out)
    p, c = float(frames[0]), 1
    for f in frames[1:]:
        if f - p <= 1:
            c += 1
            p += (f - p) / c
        else:
            out.append(p)
            p, c = float(f), 1
    out.append(p)
    return np.array(out)


def postprocess_beats(beat, down):
    bt = pick_peaks(beat) / BEAT_FPS
    dt = pick_peaks(down) / BEAT_FPS
    if len(bt):
        dt = np.unique([bt[np.argmin(np.abs(bt - d))] for d in dt])
    return bt, dt


def fill_beat_gaps(beats, tolerance=0.2):
    """Insert beats that the peak picker dropped: a gap close to k local beat
    periods (k >= 2) is split into k equal parts."""
    beats = list(beats)
    if len(beats) < 4:
        return np.array(beats)
    ibis = np.diff(beats)
    out = [beats[0]]
    for i, gap in enumerate(ibis):
        lo, hi = max(0, i - 4), min(len(ibis), i + 5)
        local = np.median(ibis[lo:hi])
        k = int(round(gap / local))
        if k >= 2 and abs(gap / k - local) < tolerance * local:
            out.extend(beats[i] + gap * j / k for j in range(1, k))
        out.append(beats[i + 1])
    return np.array(out)


def clean_downbeats(beats, downbeats):
    """Snap downbeats to beats and drop the ones that would make a bar shorter
    than half the typical bar (spurious downbeats, mostly in intros)."""
    if len(beats) == 0 or len(downbeats) == 0:
        return np.array(downbeats)
    idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    if len(idx) < 3:
        return beats[idx]
    typical = np.median(np.diff(idx))
    keep = [idx[0]]
    for j in idx[1:]:
        if j - keep[-1] < typical / 2:
            # keep whichever of the two starts the more regular grid
            if len(keep) == 1:
                keep[-1] = j
            continue
        keep.append(j)
    return beats[keep]


# ---------------------------------------------------------------- pipeline
OFF_BEAT_PENALTY = 50.0
ON_BEAT_PENALTY = 25.0


def run(models_dir, audio_path, change_penalty=OFF_BEAT_PENALTY, beat_penalty=ON_BEAT_PENALTY):
    import librosa
    import onnxruntime as ort

    y, _ = librosa.load(audio_path, sr=SR)
    y = y.astype(np.float32)
    t0 = time.time()
    chord = ort.InferenceSession(f"{models_dir}/chordnet.onnx")
    probs = chord.run(None, {"audio": y})
    t1 = time.time()
    mel = ort.InferenceSession(f"{models_dir}/beat_mel.onnx").run(None, {"audio": y})[0]
    bsess = ort.InferenceSession(f"{models_dir}/beat_this.onnx")
    bl, dl = beat_logits(mel, lambda x: bsess.run(None, {"spect": x}))
    beats, downbeats = postprocess_beats(bl, dl)
    beats = fill_beat_gaps(beats)
    downbeats = clean_downbeats(beats, downbeats)
    t2 = time.time()
    vocab = json.load(open(f"{models_dir}/chord_vocab.json"))
    cands = expand_vocab(vocab)
    obs = chord_obs(probs, cands)
    pen = np.full(len(obs), change_penalty)
    if beat_penalty is not None:
        pen = beat_change_penalties(len(obs), beats, change_penalty, beat_penalty)
    path = viterbi(obs, pen)
    segs = segments(path, [c[0] for c in cands], CHORD_FPS)
    print(f"chords {t1 - t0:.1f}s, beats {t2 - t1:.1f}s, decode {time.time() - t2:.1f}s")
    return segs, beats, downbeats, probs


def beat_change_penalties(n_frames, beats, off_beat, on_beat):
    pen = np.full(n_frames, off_beat)
    idx = np.round(np.asarray(beats) * CHORD_FPS).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    pen[idx] = on_beat
    return pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()
    segs, beats, downbeats, _ = run(args.models, args.audio)
    print(f"{len(beats)} beats, {len(downbeats)} downbeats, tempo ~{60 / np.median(np.diff(beats)):.1f} BPM")
    for s, e, n in segs:
        print(f"{s:8.2f} {e:8.2f} {n}")


if __name__ == "__main__":
    main()
