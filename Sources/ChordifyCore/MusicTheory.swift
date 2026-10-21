import Foundation

public enum PitchSpelling {
    public static let harteRoots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    static let sharps = ["C", "C\u{266F}", "D", "D\u{266F}", "E", "F", "F\u{266F}", "G", "G\u{266F}", "A", "A\u{266F}", "B"]
    static let flats = ["C", "D\u{266D}", "D", "E\u{266D}", "E", "F", "G\u{266D}", "G", "A\u{266D}", "A", "B\u{266D}", "B"]
    /// Lead-sheet default when the key gives no preference.
    static let neutral = ["C", "D\u{266D}", "D", "E\u{266D}", "E", "F", "F\u{266F}", "G", "A\u{266D}", "A", "B\u{266D}", "B"]

    public static func name(_ pc: Int, preferFlats: Bool?) -> String {
        let p = ((pc % 12) + 12) % 12
        switch preferFlats {
        case .some(true): return flats[p]
        case .some(false): return sharps[p]
        case .none: return neutral[p]
        }
    }

    /// ASCII spelling for MIDI meta events and text export.
    public static func ascii(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{266D}", with: "b")
            .replacingOccurrences(of: "\u{266F}", with: "#")
            .replacingOccurrences(of: "\u{00B0}", with: "dim")
            .replacingOccurrences(of: "\u{00F8}", with: "m7b5")
    }
}

public struct MusicalKey: Codable, Hashable, Sendable {
    public var tonic: Int
    public var minor: Bool

    public init(tonic: Int, minor: Bool) {
        self.tonic = tonic
        self.minor = minor
    }

    /// Flat-side keys are spelled with flats, sharp-side keys with sharps,
    /// C major / A minor with the usual lead-sheet mix.
    public var prefersFlats: Bool? {
        let major = minor ? (tonic + 3) % 12 : tonic
        switch major {
        case 0: return nil
        case 5, 10, 3, 8, 1: return true  // F Bb Eb Ab Db
        case 6: return true  // Gb (more common than F#) in lead sheets
        default: return false  // G D A E B
        }
    }

    /// Number of sharps (positive) or flats (negative), for MIDI key signatures.
    public var signature: Int {
        let major = minor ? (tonic + 3) % 12 : tonic
        let sharps = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5][major]
        return major == 6 && prefersFlats == true ? -6 : sharps
    }

    public func name(transpose: Int = 0) -> String {
        let k = MusicalKey(tonic: (tonic + transpose + 120) % 12, minor: minor)
        return PitchSpelling.name(k.tonic, preferFlats: k.prefersFlats) + (minor ? " minor" : " major")
    }

    /// Krumhansl-Kessler key profiles correlated with a duration-weighted
    /// pitch-class histogram of the recognised chords.
    public static func estimate(segments: [ChordSegment], vocabulary: ChordVocabulary) -> MusicalKey? {
        var hist = [Double](repeating: 0, count: 12)
        for s in segments where !s.isNoChord {
            let t = vocabulary.types[s.typeIndex]
            let d = s.end - s.start
            for tone in t.tones { hist[(s.root + tone) % 12] += d }
            hist[(s.root + t.bass) % 12] += 0.5 * d
            hist[s.root] += 0.5 * d
        }
        guard hist.reduce(0, +) > 0 else { return nil }
        let major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        let minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        var best = (r: -2.0, key: MusicalKey(tonic: 0, minor: false))
        for tonic in 0..<12 {
            for (isMinor, profile) in [(false, major), (true, minor)] {
                let rotated = (0..<12).map { profile[($0 - tonic + 12) % 12] }
                let r = correlation(hist, rotated)
                if r > best.r { best = (r, MusicalKey(tonic: tonic, minor: isMinor)) }
            }
        }
        return best.key
    }

    static func correlation(_ a: [Double], _ b: [Double]) -> Double {
        let ma = a.reduce(0, +) / 12, mb = b.reduce(0, +) / 12
        var num = 0.0, da = 0.0, db = 0.0
        for i in 0..<12 {
            num += (a[i] - ma) * (b[i] - mb)
            da += (a[i] - ma) * (a[i] - ma)
            db += (b[i] - mb) * (b[i] - mb)
        }
        return num / max(sqrt(da * db), 1e-12)
    }
}

public enum ChordStyle: String, CaseIterable, Codable, Sendable {
    /// Full symbol with extensions, alterations and slash bass.
    case jazz
    /// Triads and sevenths only, no slash bass.
    case simple
    /// Nashville numbers: scale degrees of the local key (1, 4, 5, 6m, ♭7, 5/7 ...).
    case nashville
}

public enum Nashville {
    /// Degree names for the 12 semitones above the "1".
    public static let degrees = ["1", "\u{266D}2", "2", "\u{266D}3", "3", "4", "\u{266F}4", "5", "\u{266D}6", "6", "\u{266D}7", "7"]

    /// The pitch class that is "1": the tonic, or for minor keys optionally the
    /// relative major (then the minor tonic is 6m).
    public static func one(of key: MusicalKey, minorFromRelativeMajor: Bool) -> Int {
        key.minor && minorFromRelativeMajor ? (key.tonic + 3) % 12 : key.tonic
    }

    public static func degree(_ pc: Int, one: Int) -> String {
        degrees[((pc - one) % 12 + 12) % 12]
    }
}

public struct ChordSymbol: Hashable, Sendable {
    public var root: String
    public var suffix: String
    public var bass: String?
    /// Nashville number: in plain text a suffix starting with a digit is put in
    /// parentheses ("5(7)", not "57").
    public var isNumber = false

    public init(root: String, suffix: String, bass: String?, isNumber: Bool = false) {
        self.root = root
        self.suffix = suffix
        self.bass = bass
        self.isNumber = isNumber
    }

    public var text: String {
        let sfx = isNumber && (suffix.first?.isNumber ?? false) ? "(" + suffix + ")" : suffix
        return root + sfx + (bass.map { "/" + $0 } ?? "")
    }
    public static let noChord = ChordSymbol(root: "N.C.", suffix: "", bass: nil)
}

extension ChordVocabulary {
    /// Symbol in the context of the local key (untransposed): the key sets the
    /// spelling (flats or sharps) and, for Nashville numbers, the "1".
    public func symbol(for s: ChordSegment, style: ChordStyle, transpose: Int, key: MusicalKey?,
                       minorFromRelativeMajor: Bool = false) -> ChordSymbol {
        let k = key.map { MusicalKey(tonic: ($0.tonic + transpose + 120) % 12, minor: $0.minor) }
        if style == .nashville, let k, !s.isNoChord {
            let t = types[s.typeIndex]
            let one = Nashville.one(of: k, minorFromRelativeMajor: minorFromRelativeMajor)
            let root = (s.root + transpose + 120) % 12
            let bass = t.bass == 0 ? nil : Nashville.degree((root + t.bass) % 12, one: one)
            return ChordSymbol(root: Nashville.degree(root, one: one), suffix: t.simple, bass: bass, isNumber: true)
        }
        return symbol(for: s, style: style == .nashville ? .simple : style, transpose: transpose,
                      preferFlats: k?.prefersFlats ?? nil)
    }

    public func symbol(for s: ChordSegment, style: ChordStyle, transpose: Int, preferFlats: Bool?) -> ChordSymbol {
        if s.isNoChord { return .noChord }
        let t = types[s.typeIndex]
        let root = (s.root + transpose + 120) % 12
        let rootName = PitchSpelling.name(root, preferFlats: preferFlats)
        switch style {
        case .simple, .nashville:
            return ChordSymbol(root: rootName, suffix: t.simple, bass: nil)
        case .jazz:
            let bassPC = (root + t.bass) % 12
            let bass = t.bass == 0 ? nil : PitchSpelling.name(bassPC, preferFlats: preferFlats)
            return ChordSymbol(root: rootName, suffix: t.jazz, bass: bass)
        }
    }

    /// Sounding pitch classes (absolute), after transposition.
    public func pitchClasses(for s: ChordSegment, transpose: Int = 0) -> (tones: [Int], bass: Int)? {
        if s.isNoChord { return nil }
        let t = types[s.typeIndex]
        let root = (s.root + transpose + 120) % 12
        return (t.tones.map { (root + $0) % 12 }, (root + t.bass) % 12)
    }
}
