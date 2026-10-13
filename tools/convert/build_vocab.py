"""
Build Models/chord_vocab.json: the chord dictionary used by the HMM decoder
in the app.

Each entry is a chord type with root C, written in Harte syntax, plus
  * "model": the decomposition used by the network heads
             [triad_type, bass_interval, 7th, 9th, 11th, 13th]
             (lv_chordia.complex_chord.Chord.to_numpy with root C; -1 = "don't care")
  * "tones": the pitch classes of the chord relative to the root (mir_eval)
  * "bass":  the bass interval relative to the root
  * "jazz" / "simple": the display suffix in lead-sheet style

The decoder transposes every entry to the 12 roots, exactly like
lv_chordia.extractors.xhmm_ismir.XHMMDecoder.

    python build_vocab.py --lv-chordia /path/to/lv-chordia --out Models/chord_vocab.json
"""

import argparse
import json
import sys

import mir_eval

# (harte suffix, jazz symbol, simple symbol)
# Unicode: ♭ flat, ♯ sharp, ø half-diminished, ° diminished, Δ maj7 triangle.
TYPES = [
    # triads
    ("maj", "", ""),
    ("min", "m", "m"),
    ("dim", "°", "°"),
    ("aug", "+", "+"),
    ("sus4", "sus4", "sus4"),
    ("sus2", "sus2", "sus2"),
    # sixths and added tones
    ("maj6", "6", ""),
    ("min6", "m6", "m"),
    ("maj6(9)", "6/9", ""),
    ("min6(9)", "m6/9", "m"),
    ("maj(9)", "add9", ""),
    ("min(9)", "m(add9)", "m"),
    # sevenths
    ("7", "7", "7"),
    ("maj7", "maj7", "maj7"),
    ("min7", "m7", "m7"),
    ("minmaj7", "m(maj7)", "m"),
    ("dim7", "°7", "°7"),
    ("hdim7", "m7♭5", "m7♭5"),
    ("aug(b7)", "7♯5", "7"),
    ("aug(7)", "maj7♯5", "maj7"),
    ("sus4(b7)", "7sus4", "7sus4"),
    # ninths
    ("9", "9", "7"),
    ("maj9", "maj9", "maj7"),
    ("min9", "m9", "m7"),
    ("minmaj7(9)", "m(maj9)", "m"),
    ("hdim7(9)", "m9♭5", "m7♭5"),
    ("7(b9)", "7♭9", "7"),
    ("7(#9)", "7♯9", "7"),
    ("sus4(b7,9)", "9sus4", "7sus4"),
    # elevenths
    ("11", "11", "7"),
    ("min11", "m11", "m7"),
    ("min7(11)", "m7(add11)", "m7"),
    ("7(#11)", "7♯11", "7"),
    ("9(#11)", "9♯11", "7"),
    ("maj7(#11)", "maj7♯11", "maj7"),
    ("maj9(#11)", "maj9♯11", "maj7"),
    ("7(b9,#11)", "7♭9♯11", "7"),
    # thirteenths and altered dominants
    ("13", "13", "7"),
    ("min13", "m13", "m7"),
    ("maj13", "maj13", "maj7"),
    ("7(13)", "7(13)", "7"),
    ("9(13)", "13", "7"),
    ("13(#11)", "13♯11", "7"),
    ("7(b13)", "7♭13", "7"),
    ("7(b9,b13)", "7♭9♭13", "7"),
    ("7(#9,b13)", "7alt", "7"),
    ("7(b9,13)", "13♭9", "7"),
    ("7(#9,#11)", "7♯9♯11", "7"),
    ("min7(b13)", "m7♭13", "m7"),
    ("sus4(b7,9,13)", "13sus4", "7sus4"),
    # inversions / slash chords (bass written after "/")
    ("maj/3", "", ""),
    ("maj/5", "", ""),
    ("maj/2", "", ""),
    ("maj/4", "", ""),
    ("maj/b7", "", ""),
    ("maj/7", "", ""),
    ("min/b3", "m", "m"),
    ("min/5", "m", "m"),
    ("min/b7", "m", "m"),
    ("7/3", "7", "7"),
    ("7/5", "7", "7"),
    ("7/b7", "7", "7"),
    ("maj7/3", "maj7", "maj7"),
    ("maj7/5", "maj7", "maj7"),
    ("maj7/7", "maj7", "maj7"),
    ("min7/b3", "m7", "m7"),
    ("min7/5", "m7", "m7"),
    ("min7/b7", "m7", "m7"),
    ("sus4(b7)/b7", "7sus4", "7sus4"),
    ("hdim7/b3", "m7♭5", "m7♭5"),
    ("dim7/b3", "°7", "°7"),
]

POP = {"maj", "min", "dim", "aug", "sus4", "sus2", "7", "maj7", "min7", "dim7", "hdim7",
       "maj6", "min6", "sus4(b7)", "9", "maj9", "min9", "11", "13",
       "maj/3", "maj/5", "maj/2", "maj/4", "maj/b7", "min/b3", "min/5", "min/b7"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lv-chordia", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.lv_chordia)
    from lv_chordia.complex_chord import Chord

    entries = []
    seen = {}
    for suffix, jazz, simple in TYPES:
        name = "C:" + suffix
        arr = [int(v) for v in Chord(name).to_numpy()]
        if -2 in arr:
            raise SystemExit(f"{name} is not representable by the network heads: {arr}")
        if tuple(arr) in seen:
            raise SystemExit(f"{name} duplicates {seen[tuple(arr)]}")
        seen[tuple(arr)] = name
        _, bitmap, bass = mir_eval.chord.encode(name, reduce_extended_chords=True)
        tones = [i for i in range(12) if bitmap[i]]
        if bass not in tones:
            tones = sorted(tones + [int(bass)])
        entries.append({
            "harte": suffix,
            "model": arr,  # root C: triad index = (type-1)*12+1, bass = interval
            "tones": tones,
            "bass": int(bass),
            "jazz": jazz,
            "simple": simple,
            "pop": suffix in POP,
        })
    with open(args.out, "w") as f:
        json.dump({"source": "Jiang et al., ISMIR 2019 chord structure decomposition",
                   "types": entries}, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(entries)} chord types to {args.out}")


if __name__ == "__main__":
    main()
