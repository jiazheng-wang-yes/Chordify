#!/usr/bin/env bash
# Rebuild everything in Models/ from the upstream projects.
#
#   tools/convert/make_models.sh [output-dir]
#
# Needs Python 3.10+ with the packages in tools/convert/requirements.txt and
# network access to github.com and cloud.cp.jku.at (Beat This! checkpoints).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/Models}" && cd "${1:-$ROOT/Models}" && pwd)"
WORK="${WORK:-$(mktemp -d)}"

LV_CHORDIA_REPO=https://github.com/openmirlab/lv-chordia.git
LV_CHORDIA_COMMIT=d8ab9ac99bce081bb79065d717f96e2c3b2099c8
BEAT_THIS_REPO=https://github.com/CPJKU/beat_this.git
BEAT_THIS_COMMIT=b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c
BEAT_THIS_CKPT=${BEAT_THIS_CKPT:-final0}
BEAT_THIS_URL="https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/${BEAT_THIS_CKPT}.ckpt"
SAMPLE="$ROOT/legacy/2023-cs545-svm/sample_audio.wav"

fetch() {  # repo commit dir
  if [ ! -d "$3/.git" ]; then git clone --quiet "$1" "$3"; fi
  git -C "$3" checkout --quiet "$2"
}
fetch "$LV_CHORDIA_REPO" "$LV_CHORDIA_COMMIT" "$WORK/lv-chordia"
fetch "$BEAT_THIS_REPO" "$BEAT_THIS_COMMIT" "$WORK/beat_this"
if [ ! -f "$WORK/$BEAT_THIS_CKPT.ckpt" ]; then
  curl -fL --retry 3 -o "$WORK/$BEAT_THIS_CKPT.ckpt" "$BEAT_THIS_URL"
fi
sha256sum "$WORK/$BEAT_THIS_CKPT.ckpt" 2>/dev/null || shasum -a 256 "$WORK/$BEAT_THIS_CKPT.ckpt"

cd "$HERE"
python export_chordnet.py --lv-chordia "$WORK/lv-chordia" --out "$OUT/chordnet.onnx"
python export_beat_this.py --beat-this "$WORK/beat_this" --checkpoint "$WORK/$BEAT_THIS_CKPT.ckpt" --out-dir "$OUT"
python build_vocab.py --lv-chordia "$WORK/lv-chordia" --out "$OUT/chord_vocab.json"

echo "== verification against the original implementations"
python verify_chordnet.py --lv-chordia "$WORK/lv-chordia" --onnx "$OUT/chordnet.onnx" --audio "$SAMPLE" --offset 20 --duration 60
python verify_beat_this.py --beat-this "$WORK/beat_this" --checkpoint "$WORK/$BEAT_THIS_CKPT.ckpt" --dir "$OUT" --audio "$SAMPLE"

echo "== fixtures for the Swift tests"
python make_fixtures.py --models "$OUT" --audio "$SAMPLE" --out "$ROOT/Tests/ChordifyCoreTests/Fixtures"
ls -la "$OUT"
