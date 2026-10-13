import Foundation

/// One chord type of the decoding dictionary (root C), as written by
/// tools/convert/build_vocab.py.
public struct ChordType: Codable, Hashable, Sendable {
    /// Harte suffix, e.g. "7(b9,b13)" or "maj7/3".
    public let harte: String
    /// Network decomposition for root C: [triad, bass interval, 7th, 9th, 11th, 13th].
    /// Triad index is (type - 1) * 12 + 1 + root; -1 means "not constrained".
    public let model: [Int]
    /// Pitch classes relative to the root.
    public let tones: [Int]
    /// Bass interval relative to the root.
    public let bass: Int
    /// Lead-sheet suffix (without the slash bass).
    public let jazz: String
    /// Suffix reduced to triad / seventh level.
    public let simple: String
    /// Part of the small "pop" dictionary.
    public let pop: Bool
}

public struct ChordVocabulary: Codable, Sendable {
    public let types: [ChordType]

    public static func load(from url: URL) throws -> ChordVocabulary {
        try JSONDecoder().decode(ChordVocabulary.self, from: Data(contentsOf: url))
    }
}

/// A decoding candidate: a chord type at a given root (or "no chord").
public struct ChordCandidate: Hashable, Sendable {
    /// -1 for "N" (no chord).
    public let root: Int
    /// Index into `ChordVocabulary.types`, -1 for "N".
    public let typeIndex: Int
    /// Indices into the six network heads; -1 means the head is not used.
    public let heads: [Int]

    public var isNoChord: Bool { root < 0 }
}

public enum Vocabulary: String, CaseIterable, Codable, Sendable {
    /// Every type in chord_vocab.json (extensions, alterations, slash chords).
    case jazz
    /// The small dictionary used in MIREX-style evaluations.
    case pop
}

extension ChordVocabulary {
    /// Every (root, type) pair, transposed exactly like lv-chordia's XHMMDecoder,
    /// with "N" first. Duplicated head patterns are kept only once.
    public func candidates(_ which: Vocabulary = .jazz) -> [ChordCandidate] {
        var result = [ChordCandidate(root: -1, typeIndex: -1, heads: [0, 0, -1, -1, -1, -1])]
        var seen = Set<[Int]>()
        for (ti, type) in types.enumerated() where which == .jazz || type.pop {
            let m = type.model
            for root in 0..<12 {
                let triadIndex = ((m[0] - 1) / 12) * 12 + ((m[0] - 1) % 12 + root) % 12 + 1
                let bassIndex = (m[1] + root) % 12 + 1  // 0 is "no bass"
                let heads = [triadIndex, bassIndex, m[2], m[3], m[4], m[5]]
                if seen.insert(heads).inserted {
                    result.append(ChordCandidate(root: root, typeIndex: ti, heads: heads))
                }
            }
        }
        return result
    }

    /// Harte-style name, e.g. "Bb:7(b9)" (used by the tests and the .lab export).
    public func harteName(_ c: ChordCandidate) -> String {
        if c.isNoChord { return "N" }
        return "\(PitchSpelling.harteRoots[c.root]):\(types[c.typeIndex].harte)"
    }
}
