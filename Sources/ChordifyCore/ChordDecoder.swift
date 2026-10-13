import Foundation

/// Frame-wise output of the chord network: six probability heads
/// (triad 73, bass 13, 7th 4, 9th 4, 11th 3, 13th 3), row-major (frame, class).
public struct ChordProbabilities: Sendable {
    public static let headSizes = [73, 13, 4, 4, 3, 3]
    public let frames: Int
    public let heads: [[Float]]
    /// Frames per second of the network output (22050 / 512 at the nominal rate).
    public let fps: Double

    public init(frames: Int, heads: [[Float]], fps: Double) {
        precondition(heads.count == 6)
        for (h, size) in zip(heads, Self.headSizes) { precondition(h.count == frames * size) }
        self.frames = frames
        self.heads = heads
        self.fps = fps
    }
}

public struct ChordSegment: Codable, Hashable, Sendable {
    public var start: Double
    public var end: Double
    /// -1 for "N".
    public var root: Int
    /// Index into the vocabulary, -1 for "N".
    public var typeIndex: Int

    public var isNoChord: Bool { root < 0 }
}

public enum ChordDecoder {
    /// Log-likelihood of every candidate at every frame, laid out (frame, candidate).
    /// Same as XHMMDecoder.get_chord_tag_obs.
    public static func observations(_ p: ChordProbabilities, candidates: [ChordCandidate]) -> [Float] {
        let n = candidates.count
        let sizes = ChordProbabilities.headSizes
        var logs = [[Float]]()
        for h in 0..<6 {
            logs.append(p.heads[h].map { logf(max($0, 1e-12)) })
        }
        var obs = [Float](repeating: 0, count: p.frames * n)
        obs.withUnsafeMutableBufferPointer { out in
            for t in 0..<p.frames {
                let row = t * n
                for (c, cand) in candidates.enumerated() {
                    var s: Float = 0
                    for h in 0..<6 {
                        let k = cand.heads[h]
                        if k >= 0 { s += logs[h][t * sizes[h] + k] }
                    }
                    out[row + c] = s
                }
            }
        }
        return obs
    }

    /// Viterbi decoding with a per-frame cost for changing chord
    /// (`.infinity` forbids a change at that frame).  The path starts in
    /// state 0 ("N").  Same recursion as XHMMDecoder.decode.
    public static func viterbi(observations obs: [Float], states n: Int, changePenalty: [Float]) -> [Int] {
        let frames = obs.count / n
        guard frames > 0 else { return [] }
        var back = [Int32](repeating: 0, count: frames * n)
        // Scores are accumulated in double precision, like the numpy reference.
        var score = [Double](repeating: -.infinity, count: n)
        score[0] = 0
        for c in 0..<n { score[c] += Double(obs[c]) }
        var next = [Double](repeating: 0, count: n)
        for t in 1..<frames {
            var best = 0
            for c in 1..<n where score[c] > score[best] { best = c }
            let switchScore = score[best] - Double(changePenalty[t])
            let row = t * n
            for c in 0..<n {
                if score[c] > switchScore {
                    next[c] = score[c] + Double(obs[row + c])
                    back[row + c] = Int32(c)
                } else {
                    next[c] = switchScore + Double(obs[row + c])
                    back[row + c] = Int32(best)
                }
            }
            swap(&score, &next)
        }
        var path = [Int](repeating: 0, count: frames)
        var last = 0
        for c in 1..<n where score[c] > score[last] { last = c }
        path[frames - 1] = last
        if frames > 1 {
            for t in stride(from: frames - 1, to: 0, by: -1) {
                path[t - 1] = Int(back[t * n + path[t]])
            }
        }
        return path
    }

    /// Change penalties that favour chord changes on beats: `onBeat` at the
    /// frame nearest to each beat, `offBeat` everywhere else.
    public static func beatPenalties(frames: Int, fps: Double, beats: [Double], offBeat: Float, onBeat: Float) -> [Float] {
        var pen = [Float](repeating: offBeat, count: frames)
        for b in beats {
            let i = Int((b * fps).rounded())
            if i >= 0 && i < frames { pen[i] = onBeat }
        }
        return pen
    }

    public static func segments(path: [Int], candidates: [ChordCandidate], fps: Double) -> [ChordSegment] {
        var out = [ChordSegment]()
        var start = 0
        for t in 1...path.count where t == path.count || path[t] != path[start] {
            let c = candidates[path[start]]
            out.append(ChordSegment(start: Double(start) / fps, end: Double(t) / fps,
                                    root: c.root, typeIndex: c.typeIndex))
            start = t
        }
        return out
    }
}
