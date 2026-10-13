import XCTest
@testable import ChordifyCore

/// Runs the real models through ONNX Runtime and compares with the Python
/// reference (fixtures computed with librosa/soxr decoding and onnxruntime).
func argmax(_ a: ArraySlice<Float>) -> Int {
    var best = a.startIndex
    for i in a.indices where a[i] > a[best] { best = i }
    return best - a.startIndex
}

final class EndToEndTests: XCTestCase {
    var fx: Fixtures!

    override func setUpWithError() throws {
        fx = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: Paths.fixtures.appendingPathComponent("fixtures.json")))
    }

    func sampleAudio(seconds: Double) throws -> [Float] {
        let all = try AudioLoader.loadMono(url: Paths.sample, sampleRate: 22050)
        return Array(all.prefix(Int(seconds * 22050)))
    }

    func testChordNetworkMatchesReference() throws {
        let analyzer = try Analyzer(modelsDirectory: Paths.models)
        let audio = try sampleAudio(seconds: fx.seconds)
        XCTAssertEqual(audio.count, fx.samples)
        let t0 = Date()
        let p = try analyzer.chordProbabilities(audio, fps: 22050.0 / 512.0)
        print("chord network: \(Date().timeIntervalSince(t0)) s for \(fx.seconds) s of audio")
        XCTAssertEqual(p.frames, fx.chord_frames)

        // Decoding (AVAudioConverter vs soxr) differs slightly, so compare decisions.
        let ref = try loadFloats("chord_probs.bin")
        let width = ChordProbabilities.headSizes.reduce(0, +)
        var agree = 0
        for t in 0..<p.frames {
            if argmax(p.heads[0][(t * 73)..<((t + 1) * 73)]) == argmax(ref[(t * width)..<(t * width + 73)]) {
                agree += 1
            }
        }
        let rate = Double(agree) / Double(p.frames)
        print("triad argmax agreement: \(rate)")
        XCTAssertGreaterThan(rate, 0.97)
    }

    func testBeatsMatchReference() throws {
        let analyzer = try Analyzer(modelsDirectory: Paths.models)
        let audio = try sampleAudio(seconds: fx.seconds)
        let t0 = Date()
        let (beats, downbeats) = try analyzer.beatTimes(audio)
        print("beat network: \(Date().timeIntervalSince(t0)) s; \(beats.count) beats, \(downbeats.count) downbeats")
        var matched = 0
        for b in fx.beats where beats.contains(where: { abs($0 - b) < 0.03 }) { matched += 1 }
        XCTAssertGreaterThan(Double(matched) / Double(fx.beats.count), 0.95)
        XCTAssertEqual(Double(downbeats.count), Double(fx.downbeats.count), accuracy: 1)
    }

    func testWholeSong() throws {
        let analyzer = try Analyzer(modelsDirectory: Paths.models)
        let t0 = Date()
        let (a, probs) = try analyzer.analyze(url: Paths.sample)
        let elapsed = Date().timeIntervalSince(t0)
        print(String(format: "whole song (%.0f s): %.1f s, tuning %+.1f cents, key %@, tempo %.1f, %d chords",
                     a.duration, elapsed, a.tuningCents, a.key?.name() ?? "-", a.tempo ?? 0, a.segments.count))
        XCTAssertGreaterThan(a.segments.count, 40)
        XCTAssertNotNil(a.key)
        XCTAssertEqual(a.tuningCents, 7.5, accuracy: 4)
        XCTAssertGreaterThan(a.beats.count, 300)
        let v = analyzer.vocabulary
        let text = a.segments.prefix(30).map { v.symbol(for: $0, style: .jazz, transpose: 0, preferFlats: a.key?.prefersFlats).text }
        print(text.joined(separator: " "))
        let plain = analyzer.redecode(a, probabilities: probs, extensionBoost: 0)
        XCTAssertGreaterThan(plain.segments.count, 40)
        let midi = MIDIExport.data(for: a, vocabulary: v)
        XCTAssertGreaterThan(midi.count, 1000)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("chordify-test.mid")
        try midi.write(to: out)
        print("MIDI written to \(out.path)")
    }
}
