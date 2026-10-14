# Chordify 2023: the CS545 version

This folder keeps the original Chordify exactly as it was built for **CS545 at
the University of Illinois Urbana-Champaign** (2023) by Bozhang Chen,
Jiazheng Wang and Xixiang Liu. The project poster is in [`cs545.pdf`](cs545.pdf).
Nothing in this folder is used by the new app; it is kept for memory.

## What it did

The goal had two stages:

1. **Stage one:** recognise a single guitar chord from a short recording.
2. **Stage two:** find the sequence of chords in a complete guitar piece.

### Data

[GuitarSet](https://github.com/marl/GuitarSet): the mono-pickup recordings
(cleaner than the microphone ones, with fewer extra harmonics), accompaniment
tracks only, since solo tracks carry little harmonic information. Each piece
was cut into single-chord excerpts using the `.jams` annotations.

### Features

For each excerpt, a constant-Q transform (CQT) was computed and folded into a
12-bin chroma (pitch class) profile. The chromagram was zero-padded or
truncated to 100 frames and flattened. CQT-chroma was compared with
STFT-chroma; CQT-chroma was better in every setting.

### Stage one results (80/20 split, random seed 2048)

| Classifier | Root only (CQT) | Full chord (CQT) | Root only (STFT) | Full chord (STFT) |
|---|---|---|---|---|
| Kernel SVM | 87.73% | 78.70% | 80.09% | 67.13% |
| Multilayer perceptron | 88.66% | 81.94% | 80.79% | 75.23% |
| Gaussian naive Bayes | 79.86% | 38.65% | 72.45% | 38.43% |

### Stage two (unfinished at poster time)

Two stacked LSTM layers plus a linear layer, trained on padded CQT segments
cut at the annotated chord boundaries. At test time `librosa.onset.onset_detect`
proposed chord-change points. The predicted chords changed too fast, most
likely because onsets are not chord changes. The "future directions" on the
poster were: cluster the chromagram into similar regions before classifying,
try more instruments (the Beatles collection), and try HMMs and transformers.

The new Chordify (see the top-level README) ended up following those
directions: a CNN + BiLSTM over CQT frames, an HMM (Viterbi) to keep chord
changes from flickering, a transformer beat tracker so that changes snap to
beats, and training data with many instruments (done by the authors of the
models we now use, not by us).

### The web app

`app.py` is a small Flask server:

* upload a `.wav` (saved as `uploads/audio.wav`);
* tap tempo, or estimate it with `librosa.beat.beat_track`;
* cut the song into bars of `beats_per_bar` beats;
* compute a 100-frame CQT chromagram for each bar and classify it with the
  pickled kernel SVM (`model.pkl`, scikit-learn), one chord per bar.

To run it again (Python 3.9-era packages, scikit-learn version matching the pickle):

```bash
cd legacy/2023-cs545-svm
pip install flask librosa numpy joblib scikit-learn
mkdir -p uploads && cp sample_audio.wav uploads/audio.wav
python app.py   # http://127.0.0.1:5000
```

`sample_audio.wav` is the song that was sitting in `uploads/` when the project
was last touched. It is also used by the new app's tests, so every version of
Chordify has listened to the same song.
