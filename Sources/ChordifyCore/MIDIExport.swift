import Foundation

/// Writes a Standard MIDI File (type 1) whose tempo map follows the detected
/// beats, so the file lines up with the original recording in any DAW:
///
///   track 0  tempo map, time signatures per detected bar, chord-name markers
///   track 1  "Chords": voice-led chord voicings (Acoustic Grand Piano)
///   track 2  "Bass": the bass note of each chord (Fingered Bass)
public enum MIDIExport {
    public static let ticksPerBeat = 480

    public struct Options: Sendable {
        public var transpose = 0
        public var style = ChordStyle.jazz
        public var includeBass = true
        public var minorFromRelativeMajor = false
        public init() {}
    }

    public static func data(for a: SongAnalysis, vocabulary v: ChordVocabulary, options: Options = .init()) -> Data {
        let map = TempoMap(analysis: a)

        // ---- track 0: tempo, meter, markers
        var conductor = [Event]()
        conductor.append(.meta(0, 0x03, Array("Chordify".utf8)))
        for (tick, usPerBeat) in map.tempoChanges {
            conductor.append(.meta(tick, 0x51, [UInt8((usPerBeat >> 16) & 0xFF), UInt8((usPerBeat >> 8) & 0xFF), UInt8(usPerBeat & 0xFF)]))
        }
        for (tick, num) in map.meterChanges {
            conductor.append(.meta(tick, 0x58, [UInt8(clamping: num), 2, 24, 8]))
        }
        // key signature at the start and at every modulation
        let regions = a.keyRegions.isEmpty ? a.key.map { [KeyRegion(start: 0, end: a.duration, key: $0)] } ?? [] : a.keyRegions
        for r in regions {
            let k = MusicalKey(tonic: (r.key.tonic + options.transpose + 120) % 12, minor: r.key.minor)
            conductor.append(.meta(map.tick(r.start), 0x59, [UInt8(bitPattern: Int8(k.signature)), k.minor ? 1 : 0]))
            if r.start > 0 {
                conductor.append(.meta(map.tick(r.start), 0x01, Array(PitchSpelling.ascii("Key: " + k.name()).utf8)))
            }
        }
        for s in a.segments where !s.isNoChord {
            let symbol = v.symbol(for: s, style: options.style, transpose: options.transpose, key: a.key(at: s.start),
                                  minorFromRelativeMajor: options.minorFromRelativeMajor)
            conductor.append(.meta(map.tick(s.start), 0x06, Array(PitchSpelling.ascii(symbol.text).utf8)))
        }

        // ---- tracks 1 and 2: notes
        var chords: [Event] = [.meta(0, 0x03, Array("Chords".utf8)), .program(0, channel: 0, program: 0)]
        var bass: [Event] = [.meta(0, 0x03, Array("Bass".utf8)), .program(0, channel: 1, program: 33)]
        var previous: [Int]? = nil
        for s in a.segments {
            guard let pcs = v.pitchClasses(for: s, transpose: options.transpose) else { continue }
            let on = map.tick(s.start), off = max(map.tick(s.end) - 1, on + 1)
            let voicing = Voicing.upper(tones: pcs.tones, bass: pcs.bass, previous: previous)
            previous = voicing
            for note in voicing {
                chords.append(.note(on, channel: 0, key: note, velocity: 76))
                chords.append(.note(off, channel: 0, key: note, velocity: 0))
            }
            let b = 36 + pcs.bass  // C2 .. B2
            bass.append(.note(on, channel: 1, key: b, velocity: 90))
            bass.append(.note(off, channel: 1, key: b, velocity: 0))
        }

        var tracks = [conductor, chords]
        if options.includeBass { tracks.append(bass) }
        var out = Data("MThd".utf8)
        out.append(be32(6))
        out.append(be16(1))
        out.append(be16(UInt16(tracks.count)))
        out.append(be16(UInt16(ticksPerBeat)))
        for t in tracks { out.append(encode(track: t)) }
        return out
    }

    // MARK: - tempo map

    /// Maps seconds to ticks so that every detected beat falls on a quarter
    /// note and every detected downbeat on a bar line.
    struct TempoMap {
        var beatTicks: [(time: Double, tick: Int)] = []
        var tempoChanges: [(Int, Int)] = []
        var meterChanges: [(Int, Int)] = []

        init(analysis a: SongAnalysis) {
            let grid = BarGrid(analysis: a)
            let beats = grid.bars.flatMap { $0.cells.map(\.time) }
            let q = MIDIExport.ticksPerBeat
            guard beats.count >= 2 else {
                // no beat information: constant 120 BPM
                beatTicks = [(0, 0), (0.5, q)]
                tempoChanges = [(0, 500_000)]
                meterChanges = [(0, 4)]
                return
            }
            // Time before the first detected beat becomes whole "lead-in" beats
            // at the first beat's tempo.
            let ibi = max(beats[1] - beats[0], 1e-3)
            let leadBeats = beats[0] < 0.02 ? 0 : Int((beats[0] / ibi - 1e-6).rounded(.up))
            let firstTick = leadBeats * q
            if leadBeats > 0 { beatTicks.append((0, 0)) }
            for (i, t) in beats.enumerated() { beatTicks.append((t, firstTick + i * q)) }

            for i in 0..<(beatTicks.count - 1) {
                let dt = beatTicks[i + 1].time - beatTicks[i].time
                let dk = beatTicks[i + 1].tick - beatTicks[i].tick
                guard dk > 0 else { continue }
                let us = max(1, min(0xFFFFFF, Int((dt / Double(dk) * Double(q) * 1e6).rounded())))
                if tempoChanges.last?.1 != us { tempoChanges.append((beatTicks[i].tick, us)) }
            }
            if tempoChanges.isEmpty { tempoChanges = [(0, 500_000)] }

            // One time signature per change of bar length, so that every
            // detected downbeat falls on a bar line.
            var barLengths = [Int]()
            let perBar = grid.beatsPerBar
            if leadBeats % perBar > 0 { barLengths.append(leadBeats % perBar) }
            barLengths += Array(repeating: perBar, count: leadBeats / perBar)
            barLengths += grid.bars.map { $0.cells.count }
            var tick = 0
            var last = -1
            for n in barLengths {
                if n != last { meterChanges.append((tick, n)) }
                last = n
                tick += n * q
            }
        }

        func tick(_ t: Double) -> Int {
            guard let firstPoint = beatTicks.first, let lastPoint = beatTicks.last else { return 0 }
            if t <= firstPoint.time { return firstPoint.tick }
            // binary search for the span containing t
            var lo = 0, hi = beatTicks.count - 1
            if t >= lastPoint.time {
                let n = beatTicks.count
                guard n >= 2 else { return lastPoint.tick }
                let a = beatTicks[n - 2], b = beatTicks[n - 1]
                let slope = Double(b.tick - a.tick) / max(b.time - a.time, 1e-6)
                return b.tick + Int(((t - b.time) * slope).rounded())
            }
            while lo + 1 < hi {
                let mid = (lo + hi) / 2
                if beatTicks[mid].time <= t { lo = mid } else { hi = mid }
            }
            let a = beatTicks[lo], b = beatTicks[hi]
            let frac = (t - a.time) / max(b.time - a.time, 1e-9)
            return a.tick + Int((frac * Double(b.tick - a.tick)).rounded())
        }
    }

    // MARK: - encoding

    enum Event {
        case meta(Int, UInt8, [UInt8])
        case note(Int, channel: UInt8, key: Int, velocity: UInt8)
        case program(Int, channel: UInt8, program: UInt8)

        var tick: Int {
            switch self {
            case .meta(let t, _, _), .note(let t, _, _, _), .program(let t, _, _): return t
            }
        }

        /// Note-offs before note-ons at the same tick, meta events first.
        var order: Int {
            switch self {
            case .meta: return 0
            case .program: return 1
            case .note(_, _, _, let v): return v == 0 ? 2 : 3
            }
        }
    }

    static func encode(track events: [Event]) -> Data {
        let sorted = events.enumerated().sorted {
            ($0.element.tick, $0.element.order, $0.offset) < ($1.element.tick, $1.element.order, $1.offset)
        }.map(\.element)
        var body = Data()
        var now = 0
        for e in sorted {
            body.append(varLen(max(0, e.tick - now)))
            now = max(now, e.tick)
            switch e {
            case .meta(_, let type, let bytes):
                body.append(contentsOf: [0xFF, type])
                body.append(varLen(bytes.count))
                body.append(contentsOf: bytes)
            case .note(_, let ch, let key, let vel):
                body.append(contentsOf: [(vel == 0 ? 0x80 : 0x90) | ch, UInt8(clamping: key), vel])
            case .program(_, let ch, let prog):
                body.append(contentsOf: [0xC0 | ch, prog])
            }
        }
        body.append(contentsOf: [0x00, 0xFF, 0x2F, 0x00])
        var out = Data("MTrk".utf8)
        out.append(be32(UInt32(body.count)))
        out.append(body)
        return out
    }

    static func varLen(_ value: Int) -> Data {
        var v = UInt32(clamping: value)
        var bytes = [UInt8(v & 0x7F)]
        v >>= 7
        while v > 0 {
            bytes.insert(UInt8(v & 0x7F) | 0x80, at: 0)
            v >>= 7
        }
        return Data(bytes)
    }

    static func be16(_ v: UInt16) -> Data { Data([UInt8(v >> 8), UInt8(v & 0xFF)]) }
    static func be32(_ v: UInt32) -> Data { Data([UInt8(v >> 24), UInt8((v >> 16) & 0xFF), UInt8((v >> 8) & 0xFF), UInt8(v & 0xFF)]) }
}

/// Simple voice leading for the chord track.
public enum Voicing {
    /// Upper-structure notes between E3 and about C6: every chord tone once
    /// (the bass note is left to the bass track unless it is the root), the
    /// inversion chosen to move as little as possible from the previous chord.
    public static func upper(tones: [Int], bass: Int, previous: [Int]?) -> [Int] {
        var pcs = Array(Set(tones)).sorted()
        if pcs.count > 3, pcs.contains(bass) { pcs.removeAll { $0 == bass } }
        guard !pcs.isEmpty else { return [] }
        let target = previous.map { Double($0.reduce(0, +)) / Double($0.count) } ?? 62
        var best: [Int] = []
        var bestCost = Double.infinity
        for rotation in 0..<pcs.count {
            let order = Array(pcs[rotation...] + pcs[..<rotation])
            for low in 52...63 where low % 12 == order[0] {
                var notes = [low]
                for pc in order.dropFirst() {
                    var n = notes.last! + 1
                    while n % 12 != pc { n += 1 }
                    notes.append(n)
                }
                let center = Double(notes.reduce(0, +)) / Double(notes.count)
                var cost = abs(center - target)
                if let prev = previous {  // prefer common tones staying put
                    cost -= 0.5 * Double(Set(prev).intersection(notes).count)
                }
                if notes.last! > 84 { cost += 10 }
                if cost < bestCost { bestCost = cost; best = notes }
            }
        }
        return best
    }
}
