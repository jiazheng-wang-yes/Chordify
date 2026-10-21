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


def chord_obs(probs, cands, extension_boost=0.0):
    """log-likelihood of every candidate at every frame (T, C).  `extension_boost`
    is added once for each 9th / 11th / 13th a candidate contains: the network
    was trained mostly on pop songs and is conservative with extensions."""
    eps = 1e-12
    logs = [np.log(np.maximum(p, eps)) for p in probs]
    arr = np.array([c[3] for c in cands])
    obs = logs[0][:, arr[:, 0]] + logs[1][:, arr[:, 1]]
    for i in range(4):
        col = arr[:, 2 + i]
        ok = col >= 0
        obs[:, ok] += logs[2 + i][:, col[ok]]
    if extension_boost:
        n_ext = ((arr[:, 3] > 0).astype(np.float32) + (arr[:, 4] > 0) + (arr[:, 5] > 0)).astype(np.float32)
        obs += np.float32(extension_boost) * n_ext[None, :]
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
EXTENSION_BOOST = 1.0


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
    obs = chord_obs(probs, cands, EXTENSION_BOOST)
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


# ---------------------------------------------------------------- song structure
# Mirrors Sources/ChordifyCore/SongStructure.swift.

def segment_series(x, sigma, beta):
    """Optimal partitioning of a 1-D series into constant pieces:
    minimise sum of squared deviations / sigma^2 + beta per piece.
    Returns [(start, end)] index ranges."""
    x = np.asarray(x, float)
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    best = np.full(n + 1, np.inf)
    best[0] = -beta
    prev = np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):
        for i in range(j):
            m = j - i
            s = cs[j] - cs[i]
            cost = (cs2[j] - cs2[i] - s * s / m) / sigma ** 2
            v = best[i] + cost + beta
            if v < best[j]:
                best[j] = v
                prev[j] = i
    out = []
    j = n
    while j > 0:
        out.append((int(prev[j]), j))
        j = prev[j]
    return out[::-1]


def unify_beat_levels(beats, downbeats, tol=0.07):
    """The beat tracker sometimes switches between metrical levels (for
    example eighth notes and dotted quarters).  Regions whose beat period is
    2x or 3x (or 1/2, 1/3) of a reference region are brought to the
    reference level; other differences are real tempo changes."""
    b = np.asarray(beats, float)
    if len(b) < 8:
        return b
    ibi = np.diff(b)
    regs = segment_series(np.log(ibi), 0.04, 15.0)
    per = [float(np.median(ibi[a:c])) for a, c in regs]
    dur = [float(ibi[a:c].sum()) for a, c in regs]
    mode = [None] * len(regs)
    fixed = [False] * len(regs)

    def match(q):
        for k in (2, 3):
            if abs(q / k - 1) < tol:
                return ("split", k)
            if abs(q * k - 1) < tol:
                return ("merge", k)
        return None

    order = sorted(range(len(regs)), key=lambda i: (-dur[i], i))
    fixed[order[0]] = True
    changed = True
    while changed:
        changed = False
        for j in order:
            if fixed[j]:
                continue
            refs = sorted((i for i in range(len(regs)) if fixed[i]), key=lambda i: (abs(i - j), -dur[i], i))
            for i in refs:
                if abs(per[j] / per[i] - 1) < tol:
                    fixed[j] = True
                    changed = True
                    break
                m = match(per[j] / per[i])
                if m:
                    mode[j] = m
                    per[j] = per[i]
                    fixed[j] = True
                    changed = True
                    break
        if not changed:
            rest = [j for j in order if not fixed[j]]
            if rest:
                fixed[rest[0]] = True
                changed = True
    dl = np.asarray(downbeats, float)
    out = []
    for (a, c), md in zip(regs, mode):
        if md is None:
            out += list(b[a:c])
        elif md[0] == "split":
            k = md[1]
            for j in range(a, c):
                out += [b[j] + (b[j + 1] - b[j]) * m / k for m in range(k)]
        else:
            k = md[1]
            dset = [i for i in range(a, c) if len(dl) and np.min(np.abs(dl - b[i])) < 0.05]
            phase = 0
            if dset:
                counts = [sum(1 for i in dset if (i - a) % k == ph) for ph in range(k)]
                phase = int(np.argmax(counts))
            out += [b[i] for i in range(a + phase, c, k)]
    out.append(b[-1])
    return np.array(sorted(set(np.round(out, 6))))


def split_merged_bars(beats, downbeats):
    """Downbeats on the (unified) beats; a bar of k*M beats, M being the
    dominant meter, is split into k bars (missed downbeats)."""
    if len(beats) == 0 or len(downbeats) == 0:
        return np.array(downbeats, float)
    idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    lens = [b - a for a, b in zip(idx, idx[1:] + [len(beats)])]
    m = int(np.argmax(np.bincount(lens, weights=np.array(lens, float))))
    out = []
    for a, ln in zip(idx, lens):
        if m >= 2 and ln >= 2 * m and ln % m == 0:
            out += list(range(a, a + ln, m))
        else:
            out.append(a)
    return beats[out]


def regularize_rhythm(beats, downbeats):
    b = unify_beat_levels(beats, downbeats)
    return b, split_merged_bars(b, downbeats)


def bar_starts(beats, downbeats):
    """Bar start times: the downbeats, plus the first beat when there is a pickup."""
    beats = list(beats)
    downs = list(downbeats)
    if not downs:
        return beats[::4]
    if beats and beats[0] < downs[0] - 0.01:
        return [beats[0]] + downs
    return downs


def song_end(beats, duration):
    if len(beats) < 2:
        return duration
    return min(duration, beats[-1] + float(np.median(np.diff(beats[-9:]))))


def rhythm_sections(beats, downbeats, duration, sigma=0.025, beta=20.0, merge=0.04, min_bars=2):
    """Sections of constant tempo and meter: [(start, end, first_bar, bpm, beats_per_bar)]."""
    starts = bar_starts(beats, downbeats)
    n = len(starts)
    if n == 0:
        return []
    end = song_end(beats, duration)
    ends = starts[1:] + [end]
    b = np.asarray(beats, float)
    counts = [max(1, int(((b >= s - 1e-6) & (b < e - 1e-6)).sum())) for s, e in zip(starts, ends)]
    bpm = [60.0 * k / max(e - s, 1e-3) for s, e, k in zip(starts, ends, counts)]
    met = list(counts)
    # a change of meter must last at least `min_bars` bars
    runs = []
    s = 0
    for i in range(1, n + 1):
        if i == n or met[i] != met[s]:
            runs.append([s, i, met[s]])
            s = i
    longs = [r for r in runs if r[1] - r[0] >= min_bars]
    for r in runs:
        if r[1] - r[0] < min_bars and longs:
            before = [x for x in longs if x[1] <= r[0]]
            r[2] = before[-1][2] if before else longs[0][2]
    for a, c, m in runs:
        for i in range(a, c):
            met[i] = m
    pieces = segment_series(np.log(bpm), sigma, beta)
    bounds = sorted({0, n} | {a for a, _ in pieces} | {i for i in range(1, n) if met[i] != met[i - 1]})
    out = []
    for a, c in zip(bounds[:-1], bounds[1:]):
        tempo = float(np.exp(np.median(np.log(bpm[a:c]))))
        m = met[a]
        if out and (c - a < min_bars or (out[-1][4] == m and abs(np.log(tempo / out[-1][3])) < merge)):
            pa = out[-1][2]
            tempo = float(np.exp(np.median(np.log(bpm[pa:c]))))
            out[-1] = [starts[pa], ends[c - 1], pa, tempo, out[-1][4]]
        else:
            out.append([starts[a], ends[c - 1], a, tempo, m])
    return [tuple(x) for x in out]


KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def key_regions(segments_pc, beats, downbeats, duration, switch=3.0, relative=1.5, smooth=2):
    """Local keys by Viterbi over bars.  `segments_pc` is a list of
    (start, end, tones, bass, root) with absolute pitch classes (root -1 = no chord).
    Keys are numbered tonic + 12 * minor.  Returns [(start, end, key)]."""
    starts = bar_starts(beats, downbeats)
    if not starts:
        starts = [0.0]
    end = song_end(beats, duration)
    ends = starts[1:] + [max(end, starts[-1] + 1e-3)]
    n = len(starts)
    hist = np.zeros((n, 12))
    for i, (a, c) in enumerate(zip(starts, ends)):
        for s, e, tones, bass, root in segments_pc:
            o = min(e, c) - max(s, a)
            if o <= 0 or root < 0:
                continue
            for t in tones:
                hist[i, t] += o
            hist[i, bass] += 0.5 * o
            hist[i, root] += 0.5 * o
    profiles = [np.roll(KK_MAJOR, k) for k in range(12)] + [np.roll(KK_MINOR, k) for k in range(12)]
    emis = np.zeros((n, 24))
    for i in range(n):
        lo, hi = max(0, i - smooth), min(n, i + smooth + 1)
        h = np.zeros(12)
        for j in range(lo, hi):
            h += hist[j] * (smooth + 1 - abs(j - i))
        if h.sum() > 0:
            emis[i] = [_corr(h, p) for p in profiles]

    def cost(a, b):
        if a == b:
            return 0.0
        ta, ma, tb, mb = a % 12, a >= 12, b % 12, b >= 12
        rel = (not ma and mb and tb == (ta + 9) % 12) or (ma and not mb and tb == (ta + 3) % 12)
        return switch + (relative if rel else 0.0)

    trans = np.array([[cost(a, b) for b in range(24)] for a in range(24)])
    score = emis[0].copy()
    back = np.zeros((n, 24), dtype=int)
    for i in range(1, n):
        cand = score[:, None] - trans
        back[i] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emis[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path = path[::-1]
    out = []
    s = 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[s]:
            out.append((0.0 if s == 0 else starts[s], duration if i == n else starts[i], path[s]))
            s = i
    return out
