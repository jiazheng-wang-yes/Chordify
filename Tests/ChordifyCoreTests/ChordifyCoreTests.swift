import XCTest
@testable import ChordifyCore

/// Reference data written by tools/convert/make_fixtures.py.
struct Fixtures: Decodable {
    struct ViterbiCase: Decodable {
        let name: String
        let boost: Double
        let penalty: [Double]
        let segments: [[Segment]]
    }
    enum Segment: Decodable {
        case number(Double), text(String)
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let d = try? c.decode(Double.self) { self = .number(d) } else { self = .text(try c.decode(String.self)) }
        }
        var number: Double { if case .number(let d) = self { return d }; return .nan }
        var text: String { if case .text(let s) = self { return s }; return "" }
    }
    struct GapCase: Decodable { let `in`: [Double]; let out: [Double] }
    struct DownbeatCase: Decodable { let beats: [Double]; let `in`: [Double]; let out: [Double] }

    let seconds: Double
    let samples: Int
    let chord_frames: Int
    let head_sizes: [Int]
    let n_candidates: Int
    let candidate_names_head: [String]
    let obs_checks: [[Double]]
    let viterbi: [ViterbiCase]
    let beat_frames: Int
    let chunk_starts: [String: [Int]]
    let raw_beats: [Double]
    let raw_downbeats: [Double]
    let beats: [Double]
    let downbeats: [Double]
    let gap_cases: [GapCase]
    let downbeat_cases: [DownbeatCase]
    let mel_checks: [[Double]]
    let rhythm_cases: [RhythmCase]
    let key_cases: [KeyCase]

    struct RhythmCase: Decodable {
        let beats: [Double]
        let downbeats: [Double]
        let duration: Double
        let out_beats: [Double]
        let out_downbeats: [Double]
        let sections: [[Double]]
    }
    struct KeyCase: Decodable {
        let segments: [[Segment2]]
        let beats: [Double]
        let downbeats: [Double]
        let duration: Double
        let regions: [[Double]]
    }
    /// [start, end, [tones], bass, root]
    enum Segment2: Decodable {
        case number(Double), list([Int])
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let d = try? c.decode(Double.self) { self = .number(d) } else { self = .list(try c.decode([Int].self)) }
        }
        var number: Double { if case .number(let d) = self { return d }; return .nan }
        var list: [Int] { if case .list(let l) = self { return l }; return [] }
    }
}

enum Paths {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let fixtures = repo.appendingPathComponent("Tests/ChordifyCoreTests/Fixtures")
    static let models = repo.appendingPathComponent("Models")
    static let sample = repo.appendingPathComponent("legacy/2023-cs545-svm/sample_audio.wav")
}

func loadFloats(_ name: String) throws -> [Float] {
    let data = try Data(contentsOf: Paths.fixtures.appendingPathComponent(name))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

final class ChordifyCoreTests: XCTestCase {
    var fx: Fixtures!
    var vocab: ChordVocabulary!

    override func setUpWithError() throws {
        fx = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: Paths.fixtures.appendingPathComponent("fixtures.json")))
        vocab = try ChordVocabulary.load(from: Paths.models.appendingPathComponent("chord_vocab.json"))
    }

    func probabilities() throws -> ChordProbabilities {
        let flat = try loadFloats("chord_probs.bin")
        let sizes = ChordProbabilities.headSizes
        XCTAssertEqual(sizes, fx.head_sizes)
        let width = sizes.reduce(0, +)
        XCTAssertEqual(flat.count, fx.chord_frames * width)
        var heads = sizes.map { _ in [Float]() }
        for t in 0..<fx.chord_frames {
            var o = t * width
            for h in 0..<6 {
                heads[h].append(contentsOf: flat[o..<(o + sizes[h])])
                o += sizes[h]
            }
        }
        return ChordProbabilities(frames: fx.chord_frames, heads: heads, fps: 22050.0 / 512.0)
    }

    func testCandidatesMatchReference() {
        let c = vocab.candidates(.jazz)
        XCTAssertEqual(c.count, fx.n_candidates)
        XCTAssertEqual(c.prefix(fx.candidate_names_head.count).map { vocab.harteName($0) }, fx.candidate_names_head)
    }

    func testObservationsMatchReference() throws {
        let p = try probabilities()
        let c = vocab.candidates(.jazz)
        let obs = ChordDecoder.observations(p, candidates: c)
        for check in fx.obs_checks {
            let t = Int(check[0]), k = Int(check[1])
            XCTAssertEqual(Double(obs[t * c.count + k]), check[2], accuracy: 1e-3 * max(1, abs(check[2])))
        }
    }

    func testViterbiMatchesReference() throws {
        let p = try probabilities()
        let c = vocab.candidates(.jazz)
        for vc in fx.viterbi {
            let obs = ChordDecoder.observations(p, candidates: c, extensionBoost: Float(vc.boost))
            let pen = vc.penalty.map { Float($0) }
            let path = ChordDecoder.viterbi(observations: obs, states: c.count, changePenalty: pen)
            let segs = ChordDecoder.segments(path: path, candidates: c, fps: p.fps)
            XCTAssertEqual(segs.count, vc.segments.count, vc.name)
            for (s, ref) in zip(segs, vc.segments) {
                XCTAssertEqual(s.start, ref[0].number, accuracy: 1e-6, vc.name)
                XCTAssertEqual(s.end, ref[1].number, accuracy: 1e-6, vc.name)
                let name = s.isNoChord ? "N" : vocab.harteName(ChordCandidate(root: s.root, typeIndex: s.typeIndex, heads: []))
                XCTAssertEqual(name, ref[2].text, vc.name)
            }
        }
    }

    func testChunkStarts() {
        for (n, starts) in fx.chunk_starts {
            XCTAssertEqual(BeatTracking.chunkStarts(frames: Int(n)!), starts, "n=\(n)")
        }
    }

    func testBeatPostprocessingMatchesReference() throws {
        let flat = try loadFloats("beat_logits.bin")
        let n = fx.beat_frames
        let raw = BeatTracking.postprocess(beat: Array(flat[0..<n]), downbeat: Array(flat[n..<(2 * n)]))
        XCTAssertEqual(raw.beats.count, fx.raw_beats.count)
        for (a, b) in zip(raw.beats, fx.raw_beats) { XCTAssertEqual(a, b, accuracy: 1e-9) }
        XCTAssertEqual(raw.downbeats, fx.raw_downbeats)
        let filled = BeatTracking.fillGaps(raw.beats)
        XCTAssertEqual(filled.count, fx.beats.count)
        for (a, b) in zip(filled, fx.beats) { XCTAssertEqual(a, b, accuracy: 1e-9) }
        let clean = BeatTracking.cleanDownbeats(beats: filled, downbeats: raw.downbeats)
        XCTAssertEqual(clean.count, fx.downbeats.count)
        for (a, b) in zip(clean, fx.downbeats) { XCTAssertEqual(a, b, accuracy: 1e-9) }

        for g in fx.gap_cases {
            let out = BeatTracking.fillGaps(g.in)
            XCTAssertEqual(out.count, g.out.count, "\(g.in)")
            for (a, b) in zip(out, g.out) { XCTAssertEqual(a, b, accuracy: 1e-9) }
        }
        for d in fx.downbeat_cases {
            XCTAssertEqual(BeatTracking.cleanDownbeats(beats: d.beats, downbeats: d.in), d.out, "\(d.in)")
        }
    }

    func testSymbolsAndSpelling() {
        let i7b9 = vocab.types.firstIndex { $0.harte == "7(b9)" }!
        let s = ChordSegment(start: 0, end: 1, root: 10, typeIndex: i7b9)  // Bb7b9
        XCTAssertEqual(vocab.symbol(for: s, style: .jazz, transpose: 0, preferFlats: true).text, "B\u{266D}7\u{266D}9")
        XCTAssertEqual(vocab.symbol(for: s, style: .simple, transpose: 0, preferFlats: true).text, "B\u{266D}7")
        XCTAssertEqual(vocab.symbol(for: s, style: .jazz, transpose: 2, preferFlats: nil).text, "C7\u{266D}9")
        let slash = vocab.types.firstIndex { $0.harte == "maj7/3" }!
        let c = ChordSegment(start: 0, end: 1, root: 0, typeIndex: slash)
        XCTAssertEqual(vocab.symbol(for: c, style: .jazz, transpose: 0, preferFlats: nil).text, "Cmaj7/E")
        XCTAssertEqual(PitchSpelling.ascii("B\u{266D}7\u{266F}11"), "Bb7#11")
    }

    func testKeyEstimate() {
        func seg(_ root: Int, _ harte: String, _ t: Double) -> ChordSegment {
            ChordSegment(start: t, end: t + 2, root: root, typeIndex: vocab.types.firstIndex { $0.harte == harte }!)
        }
        // ii-V-I in Bb
        let segs = [seg(0, "min7", 0), seg(5, "7", 2), seg(10, "maj7", 4), seg(10, "maj7", 6)]
        let k = MusicalKey.estimate(segments: segs, vocabulary: vocab)
        XCTAssertEqual(k, MusicalKey(tonic: 10, minor: false))
        XCTAssertEqual(k?.prefersFlats, true)
    }

    func testMIDIFileStructure() {
        let maj7 = vocab.types.firstIndex { $0.harte == "maj7" }!
        let dom = vocab.types.firstIndex { $0.harte == "7" }!
        let beats = (0..<16).map { 0.25 + Double($0) * 0.5 }
        let a = SongAnalysis(duration: 8.5, segments: [
            ChordSegment(start: 0, end: 0.25, root: -1, typeIndex: -1),
            ChordSegment(start: 0.25, end: 2.25, root: 0, typeIndex: maj7),
            ChordSegment(start: 2.25, end: 8.5, root: 7, typeIndex: dom),
        ], beats: beats, downbeats: stride(from: 0, to: 16, by: 4).map { beats[$0] }, key: nil, tuningCents: 0)
        let data = MIDIExport.data(for: a, vocabulary: vocab)
        XCTAssertEqual(Array(data.prefix(4)), Array("MThd".utf8))
        XCTAssertEqual(data[9], 1)  // format 1
        XCTAssertEqual(data[11], 3)  // three tracks
        // the second chord starts on a downbeat: lead-in of one beat, then bar 1 of 4/4
        let map = MIDIExport.TempoMap(analysis: a)
        XCTAssertEqual(map.tick(0.25), 480)
        XCTAssertEqual(map.tick(2.25), 480 + 4 * 480)
        // lead-in beat (0 -> 0.25 s) and then 120 BPM
        XCTAssertEqual(map.tempoChanges.map(\.1), [250_000, 500_000])
        XCTAssertEqual(map.meterChanges.map(\.1), [1, 4])
        XCTAssertEqual(MIDIExport.varLen(0x0FFF_FFFF), Data([0xFF, 0xFF, 0xFF, 0x7F]))
        XCTAssertEqual(MIDIExport.varLen(0x80), Data([0x81, 0x00]))
    }

    func testVoicingIsCloseAndComplete() {
        let v = Voicing.upper(tones: [0, 4, 7, 11], bass: 0, previous: nil)
        XCTAssertEqual(Set(v.map { $0 % 12 }), Set([4, 7, 11]))  // root left to the bass
        XCTAssertLessThanOrEqual(v.max()! - v.min()!, 12)
        let next = Voicing.upper(tones: [2, 5, 9, 0], bass: 2, previous: v)  // Dm7
        XCTAssertLessThanOrEqual(abs(next.reduce(0, +) / next.count - v.reduce(0, +) / v.count), 4)
    }

    func testBarGrid() {
        let beats = (0..<13).map { Double($0) * 0.5 }
        let maj = vocab.types.firstIndex { $0.harte == "maj" }!
        let a = SongAnalysis(duration: 6.5, segments: [
            ChordSegment(start: 0, end: 2.6, root: 0, typeIndex: maj),
            ChordSegment(start: 2.6, end: 6.5, root: 5, typeIndex: maj),
        ], beats: beats, downbeats: [0.5, 2.5, 4.5], key: nil, tuningCents: 0)
        let g = BarGrid(analysis: a)
        XCTAssertEqual(g.bars.map { $0.cells.count }, [1, 4, 4, 4])
        XCTAssertEqual(g.beatsPerBar, 4)
        // the F chord starts between beats 2.5 and 3.0: shown on the beat at 3.0
        let cells = g.bars[2].cells
        XCTAssertEqual(cells.map(\.showsChord), [true, true, false, false])
        XCTAssertEqual(g.position(at: 3.1)?.bar, 2)
        XCTAssertEqual(g.position(at: 3.1)?.cell, 1)
    }

    // MARK: - song structure

    func testRhythmRegularizationAndSectionsMatchReference() {
        for (n, c) in fx.rhythm_cases.enumerated() {
            let r = SongStructure.regularize(beats: c.beats, downbeats: c.downbeats)
            XCTAssertEqual(r.beats.count, c.out_beats.count, "case \(n) beats")
            for (a, b) in zip(r.beats, c.out_beats) { XCTAssertEqual(a, b, accuracy: 1e-6, "case \(n)") }
            XCTAssertEqual(r.downbeats.count, c.out_downbeats.count, "case \(n) downbeats")
            for (a, b) in zip(r.downbeats, c.out_downbeats) { XCTAssertEqual(a, b, accuracy: 1e-6, "case \(n)") }
            let sections = SongStructure.rhythmSections(beats: c.out_beats, downbeats: c.out_downbeats, duration: c.duration)
            XCTAssertEqual(sections.count, c.sections.count, "case \(n) sections")
            for (s, ref) in zip(sections, c.sections) {
                XCTAssertEqual(s.start, ref[0], accuracy: 1e-6)
                XCTAssertEqual(s.end, ref[1], accuracy: 1e-6)
                XCTAssertEqual(s.firstBar, Int(ref[2]))
                XCTAssertEqual(s.bpm, ref[3], accuracy: 1e-6 * ref[3])
                XCTAssertEqual(s.beatsPerBar, Int(ref[4]))
            }
        }
        // the synthetic case: 4/4 at 120 (with a beat-level switch), 3/4 at 90, 4/4 at 90
        let s0 = SongStructure.rhythmSections(beats: fx.rhythm_cases[0].out_beats, downbeats: fx.rhythm_cases[0].out_downbeats,
                                              duration: fx.rhythm_cases[0].duration)
        XCTAssertEqual(s0.map(\.beatsPerBar), [4, 3, 4])
        XCTAssertEqual(s0.map { Int($0.bpm.rounded()) }, [120, 90, 90])
    }

    func testKeyRegionsMatchReference() {
        for (n, c) in fx.key_cases.enumerated() {
            let spans = c.segments.map {
                SongStructure.ChordSpan(start: $0[0].number, end: $0[1].number, tones: $0[2].list,
                                        bass: Int($0[3].number), root: Int($0[4].number))
            }
            let regions = SongStructure.keyRegions(chords: spans, beats: c.beats, downbeats: c.downbeats, duration: c.duration)
            XCTAssertEqual(regions.count, c.regions.count, "case \(n)")
            for (r, ref) in zip(regions, c.regions) {
                XCTAssertEqual(r.start, ref[0], accuracy: 1e-9)
                XCTAssertEqual(r.end, ref[1], accuracy: 1e-9)
                XCTAssertEqual(r.key.tonic + (r.key.minor ? 12 : 0), Int(ref[2]))
            }
        }
        // C -> Eb -> C -> D
        XCTAssertEqual(fx.key_cases[0].regions.map { Int($0[2]) }, [0, 3, 0, 2])
    }

    func testNashvilleNumbers() {
        func seg(_ root: Int, _ harte: String) -> ChordSegment {
            ChordSegment(start: 0, end: 1, root: root, typeIndex: vocab.types.firstIndex { $0.harte == harte }!)
        }
        let c = MusicalKey(tonic: 0, minor: false)
        func num(_ s: ChordSegment, _ k: MusicalKey, transpose: Int = 0, relative: Bool = false) -> String {
            vocab.symbol(for: s, style: .nashville, transpose: transpose, key: k, minorFromRelativeMajor: relative).text
        }
        XCTAssertEqual(num(seg(0, "maj"), c), "1")
        XCTAssertEqual(num(seg(7, "7"), c), "5(7)")
        XCTAssertEqual(num(seg(9, "min7"), c), "6m7")
        XCTAssertEqual(num(seg(5, "maj7"), c), "4maj7")
        XCTAssertEqual(num(seg(10, "maj"), c), "\u{266D}7")
        XCTAssertEqual(num(seg(0, "maj/3"), c), "1/3")
        XCTAssertEqual(num(seg(2, "9"), c), "2(7)")  // simple suffix of a 9 chord is 7
        // numbers do not change when transposing
        XCTAssertEqual(num(seg(7, "7"), c, transpose: 3), "5(7)")
        // minor key: from the minor tonic, or from the relative major
        let am = MusicalKey(tonic: 9, minor: true)
        XCTAssertEqual(num(seg(9, "min"), am), "1m")
        XCTAssertEqual(num(seg(5, "maj"), am), "\u{266D}6")
        XCTAssertEqual(num(seg(4, "7"), am), "5(7)")
        XCTAssertEqual(num(seg(9, "min"), am, relative: true), "6m")
        XCTAssertEqual(num(seg(5, "maj"), am, relative: true), "4")
        // spelling follows the local key in the other styles
        XCTAssertEqual(vocab.symbol(for: seg(10, "maj"), style: .simple, transpose: 0, key: MusicalKey(tonic: 3, minor: false)).text,
                       "B\u{266D}")
        XCTAssertEqual(vocab.symbol(for: seg(10, "maj"), style: .simple, transpose: 0, key: MusicalKey(tonic: 11, minor: false)).text,
                       "A\u{266F}")
        XCTAssertEqual(MusicalKey(tonic: 3, minor: false).signature, -3)
        XCTAssertEqual(MusicalKey(tonic: 4, minor: true).signature, 1)
    }

    func testChartMarksKeyChanges() {
        let c = fx.key_cases[0]
        let segs = c.segments.map { s -> ChordSegment in
            let root = Int(s[4].number)
            let tones = Set(s[2].list.map { ($0 - root + 12) % 12 })
            let type = vocab.types.firstIndex { Set($0.tones) == tones && $0.bass == 0 }!
            return ChordSegment(start: s[0].number, end: s[1].number, root: root, typeIndex: type)
        }
        var a = SongAnalysis(duration: c.duration, segments: segs, beats: c.beats, downbeats: c.downbeats,
                             key: nil, tuningCents: 0)
        a.updateKeys(vocabulary: vocab)
        a.sections = SongStructure.rhythmSections(beats: c.beats, downbeats: c.downbeats, duration: c.duration)
        XCTAssertEqual(a.keyRegions.count, 4)
        XCTAssertEqual(a.sections.count, 1)
        let grid = BarGrid(analysis: a)
        XCTAssertEqual(grid.bars.compactMap(\.keyChange).map(\.tonic), [3, 0, 2])
        let text = ChordChart.text(analysis: a, vocabulary: vocab, style: .nashville, transpose: 0)
        XCTAssertTrue(text.contains("[Key: Eb major at 0:32]"), text)
        XCTAssertTrue(text.contains("| 1 | 6m | 2m7 | 5(7) |"), text)
        let midi = MIDIExport.data(for: a, vocabulary: vocab)
        // four key signature events (FF 59 02)
        let sig = Data([0xFF, 0x59, 0x02])
        var count = 0, i = midi.startIndex
        while let r = midi[i...].firstRange(of: sig) { count += 1; i = r.upperBound }
        XCTAssertEqual(count, 4)
    }

    func testLinkDownloaderHelpers() {
        XCTAssertEqual(LinkDownloader.normalise("youtube.com/watch?v=abc")?.absoluteString, "https://youtube.com/watch?v=abc")
        XCTAssertNil(LinkDownloader.normalise("file:///etc/passwd"))
        XCTAssertNil(LinkDownloader.normalise("not a link"))
        let args = LinkDownloader.arguments(url: URL(string: "https://example.com/-x")!, output: URL(fileURLWithPath: "/tmp/a.%(ext)s"), ffmpeg: nil)
        XCTAssertEqual(Array(args.suffix(2)), ["--", "https://example.com/-x"])
        XCTAssertFalse(args.contains("-x"))
        var out = LinkDownloader.ToolOutput()
        out.parse("CHORDIFY_PROGRESS  42.5%")
        XCTAssertEqual(out.percent ?? 0, 0.425, accuracy: 1e-9)
        out.parse("CHORDIFY_FILE /tmp/x y.m4a")
        out.parse("CHORDIFY_TITLE Autumn Leaves")
        XCTAssertEqual(out.file, "/tmp/x y.m4a")
        XCTAssertEqual(out.title, "Autumn Leaves")
    }
}
