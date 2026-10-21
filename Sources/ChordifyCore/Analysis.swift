import CryptoKit
import Foundation

/// Everything the app needs to show and export a song.
public struct SongAnalysis: Codable, Sendable {
    public static let formatVersion = 2

    public var version = SongAnalysis.formatVersion
    public var duration: Double
    public var segments: [ChordSegment]
    public var beats: [Double]
    public var downbeats: [Double]
    public var key: MusicalKey?
    /// Tuning of the recording relative to A = 440 Hz, in cents.
    public var tuningCents: Double
    /// Peak-normalised loudness envelope (RMS), for drawing the waveform.
    public var envelope: [Float] = []
    /// Extension boost used to decode `segments`.
    public var extensionBoost: Float = Analyzer.defaultExtensionBoost
    /// Local keys (modulations); `key` is the key of the longest region.
    public var keyRegions: [KeyRegion] = []
    /// Stretches of constant tempo and meter.
    public var sections: [RhythmSection] = []

    /// Local key at time t (falls back to the global key).
    public func key(at t: Double) -> MusicalKey? {
        keyRegions.last(where: { $0.start <= t + 1e-6 })?.key ?? keyRegions.first?.key ?? key
    }

    public func section(at t: Double) -> RhythmSection? {
        sections.last(where: { $0.start <= t + 1e-6 }) ?? sections.first
    }

    /// Recompute keys from the current segments.
    mutating func updateKeys(vocabulary v: ChordVocabulary) {
        keyRegions = SongStructure.keyRegions(segments: segments, vocabulary: v, beats: beats,
                                              downbeats: downbeats, duration: duration)
        var length = [MusicalKey: Double]()
        for r in keyRegions { length[r.key, default: 0] += r.end - r.start }
        key = length.max { $0.value < $1.value }?.key ?? MusicalKey.estimate(segments: segments, vocabulary: v)
    }

    public var tempo: Double? {
        guard beats.count > 2 else { return nil }
        let ibis = zip(beats.dropFirst(), beats).map { $0 - $1 }
        return 60 / BeatTracking.median(ibis)
    }

    /// Index of the segment playing at time t.
    public func segmentIndex(at t: Double) -> Int? {
        var lo = 0, hi = segments.count - 1
        guard hi >= 0 else { return nil }
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if segments[mid].start <= t { lo = mid } else { hi = mid - 1 }
        }
        return lo
    }
}

/// Bars built from the beat and downbeat times.
public struct BarGrid: Sendable {
    public struct Cell: Hashable, Sendable {
        public var time: Double
        public var end: Double
        public var segment: Int?
        /// The chord differs from the previous cell (or this is the first cell of a line).
        public var showsChord: Bool
    }
    public struct Bar: Hashable, Sendable, Identifiable {
        public var id: Int
        public var cells: [Cell]
        /// The key changes at this bar (not set for the first key).
        public var keyChange: MusicalKey?
        /// A new tempo / meter section starts at this bar (not set for the first one).
        public var sectionChange: RhythmSection?
        public var start: Double { cells.first?.time ?? 0 }
        public var end: Double { cells.last?.end ?? 0 }
    }

    public var bars: [Bar]
    public var beatsPerBar: Int

    public init(analysis a: SongAnalysis) {
        var beats = a.beats
        if beats.count < 4 {  // no usable beats: fall back to a 120 BPM grid
            beats = Array(stride(from: 0.0, to: max(a.duration, 0.5), by: 0.5))
        }
        let downbeatSet = Set(a.downbeats.map { BeatTracking.nearestIndex(beats, $0) })
        var groups = [[Int]]()
        var current = [Int]()
        for i in beats.indices {
            if downbeatSet.contains(i), !current.isEmpty {
                groups.append(current)
                current = []
            }
            current.append(i)
        }
        if !current.isEmpty { groups.append(current) }
        if downbeatSet.isEmpty {  // no downbeats: bars of four beats
            groups = stride(from: 0, to: beats.count, by: 4).map { Array($0..<min($0 + 4, beats.count)) }
        }
        let counts = groups.map(\.count)
        let typical = counts.isEmpty ? 4 : BeatTracking.median(counts.map(Double.init))
        beatsPerBar = max(1, Int(typical.rounded()))

        var bars = [Bar]()
        var previous: Int?? = .none
        for (bi, g) in groups.enumerated() {
            var cells = [Cell]()
            for (ci, i) in g.enumerated() {
                let t = beats[i]
                let end = i + 1 < beats.count ? beats[i + 1] : min(a.duration, t + (beats.count > 1 ? beats[1] - beats[0] : 0.5))
                // the chord sounding just after the beat
                let seg = a.segmentIndex(at: t + 0.03)
                let shows = ci == 0 || previous != .some(seg)
                cells.append(Cell(time: t, end: end, segment: seg, showsChord: shows))
                previous = .some(seg)
            }
            bars.append(Bar(id: bi, cells: cells))
        }
        // mark key and tempo/meter changes on the bar where they start
        func barIndex(at t: Double) -> Int? {
            bars.firstIndex { abs($0.start - t) < 0.05 } ?? bars.lastIndex { $0.start <= t }
        }
        for r in a.keyRegions.dropFirst() {
            if let i = barIndex(at: r.start) { bars[i].keyChange = r.key }
        }
        for s in a.sections.dropFirst() {
            if let i = barIndex(at: s.start) { bars[i].sectionChange = s }
        }
        self.bars = bars
    }

    /// (bar, cell) for time t.
    public func position(at t: Double) -> (bar: Int, cell: Int)? {
        guard !bars.isEmpty else { return nil }
        var lo = 0, hi = bars.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if bars[mid].start <= t { lo = mid } else { hi = mid - 1 }
        }
        let cells = bars[lo].cells
        var c = 0
        while c + 1 < cells.count && cells[c + 1].time <= t { c += 1 }
        return (lo, c)
    }
}

/// Runs the whole analysis: decode audio, chords, beats, decoding, key.
public final class Analyzer: @unchecked Sendable {
    public static let sampleRate = 22050.0
    public static let chordHop = 512.0
    public static let offBeatPenalty: Float = 50
    public static let onBeatPenalty: Float = 25
    public static let defaultExtensionBoost: Float = 1

    public enum Stage: String, Sendable {
        case decoding = "Reading audio"
        case tuning = "Measuring tuning"
        case listening = "Listening for chords and beats"
        case finishing = "Writing the chart"
    }

    public let vocabulary: ChordVocabulary
    public let candidates: [ChordCandidate]
    private let modelsDir: URL
    private lazy var chordModel = Result { try OnnxModel(url: modelsDir.appendingPathComponent("chordnet.onnx")) }
    private lazy var melModel = Result { try OnnxModel(url: modelsDir.appendingPathComponent("beat_mel.onnx")) }
    private lazy var beatModel = Result { try OnnxModel(url: modelsDir.appendingPathComponent("beat_this.onnx")) }
    private let lock = NSLock()

    public init(modelsDirectory: URL? = nil) throws {
        modelsDir = try modelsDirectory ?? ModelLocator.directory()
        vocabulary = try ChordVocabulary.load(from: modelsDir.appendingPathComponent("chord_vocab.json"))
        candidates = vocabulary.candidates(.jazz)
    }

    public func analyze(url: URL, extensionBoost: Float = Analyzer.defaultExtensionBoost,
                        progress: @escaping @Sendable (Stage) -> Void = { _ in }) throws
        -> (analysis: SongAnalysis, probabilities: ChordProbabilities)
    {
        lock.lock()
        defer { lock.unlock() }
        progress(.decoding)
        let audio = try AudioLoader.loadMono(url: url, sampleRate: Self.sampleRate)
        guard audio.count > Int(Self.sampleRate) / 2 else { throw ChordifyError.audio("The file is too short.") }
        let duration = Double(audio.count) / Self.sampleRate

        // Detuned recordings are resampled so that the chord network sees
        // A = 440 Hz; the frame rate is corrected accordingly.
        progress(.tuning)
        let cents = Tuning.estimateCents(audio, sampleRate: Self.sampleRate)
        var chordAudio = audio
        var chordFPS = Self.sampleRate / Self.chordHop
        if abs(cents) >= 5 {
            let ratio = pow(2, cents / 1200)
            chordAudio = try AudioLoader.resample(audio, from: Self.sampleRate, to: Self.sampleRate * ratio)
            chordFPS = Self.sampleRate * ratio / Self.chordHop
        }

        // The chord and beat networks run at the same time.
        progress(.listening)
        _ = (try chordModel.get(), try melModel.get(), try beatModel.get())  // load before going parallel
        let box = ParallelResults()
        DispatchQueue.concurrentPerform(iterations: 2) { i in
            if i == 0 {
                box.chords = Result { try self.chordProbabilities(chordAudio, fps: chordFPS) }
            } else {
                box.beats = Result { try self.beatTimes(audio) }
            }
        }
        let probs = try box.chords!.get()
        let (beats, downbeats) = try box.beats!.get()

        progress(.finishing)
        let segments = decode(probs, beats: beats, extensionBoost: extensionBoost)
        var a = SongAnalysis(duration: duration, segments: segments, beats: beats, downbeats: downbeats,
                             key: nil, tuningCents: cents, envelope: Self.envelope(audio), extensionBoost: extensionBoost)
        a.updateKeys(vocabulary: vocabulary)
        a.sections = SongStructure.rhythmSections(beats: beats, downbeats: downbeats, duration: duration)
        return (a, probs)
    }

    /// Re-decode an analysis with another extension boost (fast: no network).
    public func redecode(_ a: SongAnalysis, probabilities: ChordProbabilities, extensionBoost: Float) -> SongAnalysis {
        var b = a
        b.segments = decode(probabilities, beats: a.beats, extensionBoost: extensionBoost)
        b.updateKeys(vocabulary: vocabulary)
        b.extensionBoost = extensionBoost
        return b
    }

    static func envelope(_ x: [Float], points: Int = 1200) -> [Float] {
        let step = max(1, x.count / points)
        var env = [Float]()
        env.reserveCapacity(points)
        var i = 0
        while i < x.count {
            let end = min(x.count, i + step)
            var s: Float = 0
            for j in i..<end { s += x[j] * x[j] }
            env.append(sqrt(s / Float(end - i)))
            i = end
        }
        let peak = env.max() ?? 1
        return peak > 0 ? env.map { $0 / peak } : env
    }

    public func decode(_ probs: ChordProbabilities, beats: [Double],
                       extensionBoost: Float = Analyzer.defaultExtensionBoost) -> [ChordSegment] {
        let obs = ChordDecoder.observations(probs, candidates: candidates, extensionBoost: extensionBoost)
        let pen = ChordDecoder.beatPenalties(frames: probs.frames, fps: probs.fps, beats: beats,
                                             offBeat: Self.offBeatPenalty, onBeat: Self.onBeatPenalty)
        let path = ChordDecoder.viterbi(observations: obs, states: candidates.count, changePenalty: pen)
        return ChordDecoder.segments(path: path, candidates: candidates, fps: probs.fps)
    }

    public func chordProbabilities(_ audio: [Float], fps: Double) throws -> ChordProbabilities {
        let model = try chordModel.get()
        let names = ["triad", "bass", "seventh", "ninth", "eleventh", "thirteenth"]
        let out = try model.run(["audio": .init(data: audio, shape: [audio.count])], outputs: names)
        let heads = names.map { out[$0]!.data }
        let frames = out["triad"]!.shape[0]
        return ChordProbabilities(frames: frames, heads: heads, fps: fps)
    }

    public func beatTimes(_ audio: [Float]) throws -> (beats: [Double], downbeats: [Double]) {
        let mel = try melModel.get().run(["audio": .init(data: audio, shape: [audio.count])], outputs: ["spect"])["spect"]!
        let frames = mel.shape[0]
        let tracker = try self.beatModel.get()
        let (b, d) = try BeatTracking.logits(spect: mel.data, frames: frames) { x, len in
            let o = try tracker.run(["spect": .init(data: x, shape: [1, len, 128])], outputs: ["beat", "downbeat"])
            return (o["beat"]!.data, o["downbeat"]!.data)
        }
        let raw = BeatTracking.postprocess(beat: b, downbeat: d)
        let beats = BeatTracking.fillGaps(raw.beats)
        let downbeats = BeatTracking.cleanDownbeats(beats: beats, downbeats: raw.downbeats)
        return SongStructure.regularize(beats: beats, downbeats: downbeats)
    }
}

private final class ParallelResults: @unchecked Sendable {
    var chords: Result<ChordProbabilities, Error>?
    var beats: Result<(beats: [Double], downbeats: [Double]), Error>?
}

/// On-disk cache of analyses, keyed by file content.
public enum AnalysisCache {
    static var directory: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Chordify", isDirectory: true)
    }

    public static func key(for url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        // size + first and last MiB is enough to tell songs apart, and fast.
        let size = (try? handle.seekToEnd()) ?? 0
        hasher.update(data: withUnsafeBytes(of: size) { Data($0) })
        try? handle.seek(toOffset: 0)
        if let head = try? handle.read(upToCount: 1 << 20) { hasher.update(data: head) }
        if size > 1 << 21 {
            try? handle.seek(toOffset: size - (1 << 20))
            if let tail = try? handle.read(upToCount: 1 << 20) { hasher.update(data: tail) }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func load(_ key: String) -> SongAnalysis? {
        let url = directory.appendingPathComponent(key + ".json")
        guard let data = try? Data(contentsOf: url),
              let a = try? JSONDecoder().decode(SongAnalysis.self, from: data),
              a.version == SongAnalysis.formatVersion else { return nil }
        return a
    }

    public static func store(_ a: SongAnalysis, key: String) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(a) {
            try? data.write(to: directory.appendingPathComponent(key + ".json"))
        }
    }

    /// Network output, kept so that decoding settings can change without re-analysis.
    /// Layout: frames (Int64), fps (Float64), then the six heads as float32.
    public static func store(_ p: ChordProbabilities, key: String) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var data = Data()
        withUnsafeBytes(of: Int64(p.frames)) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: p.fps) { data.append(contentsOf: $0) }
        for h in p.heads { h.withUnsafeBytes { data.append(contentsOf: $0) } }
        try? data.write(to: directory.appendingPathComponent(key + ".probs"))
    }

    public static func loadProbabilities(_ key: String) -> ChordProbabilities? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(key + ".probs")), data.count >= 16 else { return nil }
        let frames = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int64.self) })
        let fps = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: Double.self) }
        let total = ChordProbabilities.headSizes.reduce(0, +) * frames
        guard frames > 0, data.count == 16 + total * 4 else { return nil }
        let floats: [Float] = data.withUnsafeBytes { raw in
            (0..<total).map { raw.loadUnaligned(fromByteOffset: 16 + $0 * 4, as: Float.self) }
        }
        var heads = [[Float]]()
        var o = 0
        for size in ChordProbabilities.headSizes {
            heads.append(Array(floats[o..<(o + size * frames)]))
            o += size * frames
        }
        return ChordProbabilities(frames: frames, heads: heads, fps: fps)
    }
}
