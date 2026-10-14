#!/usr/bin/env bash
# Build Chordify.app (Apple silicon) into build/.
#
#   scripts/build_app.sh            # build/Chordify.app
#   scripts/build_app.sh --install  # also copy it to /Applications
#
# Needs the Xcode command line tools (xcode-select --install) on macOS 14+.
set -euo pipefail
cd "$(dirname "$0")/.."

for f in chordnet.onnx beat_mel.onnx beat_this.onnx chord_vocab.json; do
  [ -f "Models/$f" ] || { echo "Models/$f is missing (see tools/convert/make_models.sh)"; exit 1; }
done

swift build -c release --product Chordify
BIN="$(swift build -c release --show-bin-path)"

APP=build/Chordify.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/Models"
cp "$BIN/Chordify" "$APP/Contents/MacOS/Chordify"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp Models/chordnet.onnx Models/beat_mel.onnx Models/beat_this.onnx Models/chord_vocab.json "$APP/Contents/Resources/Models/"
iconutil -c icns Resources/AppIcon.iconset -o "$APP/Contents/Resources/AppIcon.icns"
# SwiftPM resource bundles of dependencies, if any
find "$BIN" -maxdepth 1 -name "*.bundle" -exec cp -R {} "$APP/Contents/Resources/" \;

# ad-hoc signature so the app runs on Apple silicon
codesign --force --deep --sign - "$APP"

(cd build && rm -f Chordify.zip && ditto -c -k --keepParent Chordify.app Chordify.zip)
du -sh "$APP" build/Chordify.zip

if [ "${1:-}" = "--install" ]; then
  rm -rf /Applications/Chordify.app
  cp -R "$APP" /Applications/
  echo "Installed to /Applications/Chordify.app"
fi
