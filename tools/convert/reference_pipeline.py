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
    # deduplicate adjacent peaks (running mean of groups with gaps <= 1 frame)
    out = []
    for f in frames:
        if out and f - out[-1][1] <= 1:
            grp = out[-1]
            grp[2] += 1
            grp[0] += (f - grp[0]) / grp[2]
            grp[1] = f
        else:
            out.append([float(f), f, 1])
    return np.array([g[0] for g in out])


def postprocess_beats(beat, down):
    bt = pick_peaks(beat) / BEAT_FPS
    dt = pick_peaks(down) / BEAT_FPS
    if len(bt):
        dt = np.unique([bt[np.argmin(np.abs(bt - d))] for d in dt])
    return bt, dt


# ---------------------------------------------------------------- pipeline
def run(models_dir, audio_path, change_penalty=30.0, beat_penalty=None):
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
