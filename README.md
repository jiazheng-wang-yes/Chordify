# Chordify

A small, native macOS app that listens to a song and writes out its chords,
down to jazz voicings like `B♭13♯11`, `E7alt`, `Am9`, `F♯m7♭5` and slash chords.
You play the song and follow the chords bar by bar, then export everything as a
MIDI file that lines up with the original recording.

Everything runs on your Mac (Apple silicon, macOS 14 or later). Nothing is uploaded.

## Features

- **Drop a song, get the chart.** MP3, M4A/AAC, ALAC, WAV, AIFF, FLAC, CAF.
  A 4:18 song takes 16.6 s on GitHub's 3-core virtual M1 (release build);
  a 10-core M1 Pro should be faster. Results are cached, so re-opening a song
  is instant.
- **Audio and chords together.** The big chord display, the "next" chord, and a
  piano with the chord tones (bass marked) follow the playhead. The bar grid
  highlights the current beat and scrolls with the music. Click any beat or
  any point on the waveform to jump there.
- **Jazz vocabulary.** 71 chord types: triads, sus, 6 and 6/9, all seventh
  qualities, 9/11/13 extensions, altered dominants (`7♭9`, `7♯9`, `7♯11`,
  `7♭13`, `7alt`, `13♭9`, `13♯11`, ...), `m(maj7)`, `maj7♯5`, `m7♭5`, `°7`, and
  inversions/slash chords. Switch to **Simple** to see the same song as
  triads and sevenths. The **Extensions** setting (Fewer / Normal / More)
  sets how readily 9ths, 11ths and 13ths are reported; changing it is instant.
- **MIDI export.** A type-1 MIDI file with a chord track (voice-led
  voicings), a bass track, chord names as markers, and a tempo map built from
  the detected beats, with a time signature per detected bar. Drop it into
  Logic, Ableton, etc. next to the original audio and it stays in sync.
- **Practice tools.** Slow down to 50% without changing pitch, transpose
  chords (and optionally the audio) by semitones, jump chord by chord.
- **Key, tempo, meter** shown for each song. Detuned recordings (for example
  old records) are measured and corrected before chord recognition.
- **Chord chart export** as text: bars with chords, plus timed chords in seconds.

### Keyboard

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Back / forward 5 seconds |
| ⌘← / ⌘→ | Previous / next chord |
| `-` / `=` / `0` | Slower / faster / normal speed |
| ⌘↑ / ⌘↓ | Transpose up / down |
| ⌘J | Jazz / Simple chord symbols |
| ⌘O | Open a song |
| ⌘E / ⇧⌘E | Export MIDI / chord chart |

## Install

### Build it yourself (recommended)

You need the Xcode command line tools (`xcode-select --install`).

```bash
git clone https://github.com/jiazheng-wang-yes/Chordify.git
cd Chordify
scripts/build_app.sh --install     # builds build/Chordify.app and copies it to /Applications
```

### Download a build

Every push builds the app on GitHub Actions ("macOS app" workflow). Download
the `Chordify-macOS-arm64` artifact from the latest run, unzip it, and move
`Chordify.app` to Applications. The build is ad-hoc signed, not notarised, so
the first time macOS will refuse to open it. Either run
`xattr -dr com.apple.quarantine /Applications/Chordify.app`, or open it once
and allow it in System Settings → Privacy & Security.

## How it works

```
audio file ──AVFoundation──► mono 22.05 kHz ──► tuning estimate (spectral peaks)
                                    │
             ┌──────────────────────┴─────────────────────┐
             ▼                                            ▼
  constant-Q transform (in the ONNX graph)        log-mel spectrogram (in the ONNX graph)
  5 × CNN + BiLSTM chord networks                 Beat This! transformer, 30 s chunks
  heads: triad, bass, 7th, 9th, 11th, 13th        beat + downbeat activations
             │                                            │
             ▼                                            ▼
  chord likelihoods for 853 (root, type) pairs     peak picking, missing-beat filling,
             │                                     spurious-downbeat removal
             └──────────────► Viterbi decoding ◄──────────┘
                     (chord changes cost less on beats)
                                    │
                                    ▼
            chord segments, bars, key  ──►  UI, MIDI, chart
```

**Chords.** The chord model is the large-vocabulary recognizer of
Junyan Jiang, Ke Chen, Wei Li and Gus Xia, *Large-Vocabulary Chord
Transcription via Chord Structure Decomposition*, ISMIR 2019
([code](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition),
packaged as [lv-chordia](https://github.com/openmirlab/lv-chordia), MIT). It does
not classify among a fixed list of chord names. Instead it predicts the
*structure* of the chord separately: triad and root, bass note, 7th, 9th, 11th
and 13th. The decoder then scores any chord that can be built from those
parts. This is what makes extensions and alterations recognisable. It is the
strongest openly available model I found that predicts 9th/11th/13th
extensions. As reported by lv-chordia, the ensemble scores about 81%
(Billboard), 78% (RWC Pop) and 83% (Beatles) on the standard MIREX-style
vocabulary. Jazz extensions are harder than that, and fast reharmonised
passages will contain mistakes.

The constant-Q front end was rewritten so that it can live inside the ONNX
graph. It matches `librosa.hybrid_cqt` to within 0.4% overall error, and the
exported 5-network ensemble agrees with the original PyTorch + librosa
pipeline on more than 99.9% of frames (`tools/convert/verify_chordnet.py`).

**Beats and bars.** Francesco Foscarin, Jan Schlüter and Gerhard Widmer,
*Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024
([code](https://github.com/CPJKU/beat_this), MIT), `final0` checkpoint. It
predicts beats and downbeats, so the grid shows real bars (including
3/4, 6/8 and changing meters).

**Decoding.** The Viterbi search is the same recursion as lv-chordia's
`XHMMDecoder`; the Python version in `tools/convert/reference_pipeline.py` gives
identical labels. Two things were added. First, a chord change costs less on
a detected beat, which keeps changes on the beat without forbidding
anticipations. Second, there is a small bonus for each 9th, 11th and 13th (the
Extensions setting). The network was trained mostly on pop songs, and without
the bonus it reports a jazz voicing like Dm9 or G13♭9 as Dm7 or G7. On a
synthetic ii–V–I test, the default bonus raised exact matches from 6/16 to 9/16
chords and added no extensions to a pop song.

**Key.** Krumhansl–Kessler key profiles correlated with the duration-weighted
chord tones. It is used to choose ♭ or ♯ spellings.

**Size.** Models: chord ensemble 5.7 MB, beat tracker 43 MB (weights stored
as float16, run in float32), CQT/mel front ends 4.5 MB. ONNX Runtime is
linked statically. The app bundle is 83 MB (53 MB zipped).

### Command line

The app binary also works in the terminal:

```bash
/Applications/Chordify.app/Contents/MacOS/Chordify --analyze song.mp3 --midi song.mid
```

It prints the key, tempo, tuning and the chord chart bar by bar.

## Repository layout

```
Sources/ChordifyCore/     analysis engine (Swift): audio, models, decoding, MIDI
Sources/Chordify/         the SwiftUI app
Tests/ChordifyCoreTests/  unit tests + end-to-end tests against Python fixtures
Models/                   ONNX models and the chord vocabulary (generated)
tools/convert/            PyTorch → ONNX export, verification, fixtures
scripts/build_app.sh      builds Chordify.app
legacy/2023-cs545-svm/    the original 2023 course project (SVM on CQT-chroma)
```

### Rebuilding the models

`Models/` is generated by `tools/convert/make_models.sh`. It clones lv-chordia
and Beat This! at pinned commits, downloads the Beat This! checkpoint,
exports the ONNX files, checks them against the original implementations, and
writes the Swift test fixtures. The "Convert models" GitHub workflow runs it
and commits the result.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r tools/convert/requirements.txt
tools/convert/make_models.sh
```

## History

Chordify started in 2023 as a CS545 course project at UIUC by Bozhang Chen,
Jiazheng Wang and Xixiang Liu: a Flask web page, tap tempo, and a kernel SVM
on 100-frame CQT chromagrams, one chord per bar, trained on GuitarSet. The
poster's "future directions" were HMMs, transformers and more instruments;
this version is built on exactly those ideas. The original code, the trained
SVM, and the poster are kept in [`legacy/2023-cs545-svm`](legacy/2023-cs545-svm).

## Licenses

Third-party parts: chord model weights, MIT (lv-chordia / Music X Lab); beat
model weights, MIT (CPJKU); ONNX Runtime, MIT. This repository has no license
file of its own yet.
