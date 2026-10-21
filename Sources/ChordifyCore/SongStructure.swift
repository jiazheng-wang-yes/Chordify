import Foundation

/// A stretch of the song with one tempo and meter.
public struct RhythmSection: Codable, Hashable, Sendable {
    public var start: Double
    public var end: Double
    /// Index of the first bar (as in `BarGrid`).
    public var firstBar: Int
    public var bpm: Double
    public var beatsPerBar: Int

    public init(start: Double, end: Double, firstBar: Int, bpm: Double, beatsPerBar: Int) {
        self.start = start
        self.end = end
        self.firstBar = firstBar
        self.bpm = bpm
        self.beatsPerBar = beatsPerBar
    }
}

/// A stretch of the song in one key.
public struct KeyRegion: Codable, Hashable, Sendable {
    public var start: Double
    public var end: Double
    public var key: MusicalKey

    public init(start: Double, end: Double, key: MusicalKey) {
        self.start = start
        self.end = end
        self.key = key
    }
}

/// Key changes, tempo changes and meter changes. Same algorithms as the
/// "song structure" part of tools/convert/reference_pipeline.py.
public enum SongStructure {
    // MARK: - piecewise-constant segmentation

    /// Optimal partitioning of a series into constant pieces: minimises the
    /// squared deviations / sigma^2 plus `beta` per piece. Returns index ranges.
    public static func segmentSeries(_ x: [Double], sigma: Double, beta: Double) -> [Range<Int>] {
        let n = x.count
        guard n > 0 else { return [] }
        var cs = [0.0], cs2 = [0.0]
        for v in x {
            cs.append(cs.last! + v)
            cs2.append(cs2.last! + v * v)
        }
        var best = [Double](repeating: .infinity, count: n + 1)
        best[0] = -beta
        var prev = [Int](repeating: 0, count: n + 1)
        let s2 = sigma * sigma
        for j in 1...n {
            for i in 0..<j {
                let m = Double(j - i)
                let s = cs[j] - cs[i]
                let cost = (cs2[j] - cs2[i] - s * s / m) / s2
                let v = best[i] + cost + beta
                if v < best[j] {
                    best[j] = v
                    prev[j] = i
                }
            }
        }
        var out = [Range<Int>]()
        var j = n
        while j > 0 {
            out.append(prev[j]..<j)
            j = prev[j]
        }
        return out.reversed()
    }

    // MARK: - rhythm

    /// Brings regions where the beat tracker switched metrical level (x2, x3,
    /// 1/2, 1/3 of a reference region's beat period) back to one level.
    public static func unifyBeatLevels(_ b: [Double], downbeats: [Double], tolerance tol: Double = 0.07) -> [Double] {
        guard b.count >= 8 else { return b }
        let ibi = zip(b.dropFirst(), b).map { $0 - $1 }
        let regs = segmentSeries(ibi.map { log($0) }, sigma: 0.04, beta: 15)
        var per = regs.map { BeatTracking.median(Array(ibi[$0])) }
        let dur = regs.map { r in ibi[r].reduce(0, +) }
        var mode = [(split: Bool, k: Int)?](repeating: nil, count: regs.count)
        var fixed = [Bool](repeating: false, count: regs.count)

        func match(_ q: Double) -> (split: Bool, k: Int)? {
            for k in [2, 3] {
                if abs(q / Double(k) - 1) < tol { return (true, k) }
                if abs(q * Double(k) - 1) < tol { return (false, k) }
            }
            return nil
        }

        let order = regs.indices.sorted { (-dur[$0], $0) < (-dur[$1], $1) }
        fixed[order[0]] = true
        var changed = true
        while changed {
            changed = false
            for j in order where !fixed[j] {
                let refs = regs.indices.filter { fixed[$0] }
                    .sorted { (abs($0 - j), -dur[$0], $0) < (abs($1 - j), -dur[$1], $1) }
                for i in refs {
                    if abs(per[j] / per[i] - 1) < tol {
                        fixed[j] = true
                        changed = true
                        break
                    }
                    if let m = match(per[j] / per[i]) {
                        mode[j] = m
                        per[j] = per[i]
                        fixed[j] = true
                        changed = true
                        break
                    }
                }
            }
            if !changed, let rest = order.first(where: { !fixed[$0] }) {
                fixed[rest] = true
                changed = true
            }
        }

        var out = [Double]()
        for (r, md) in zip(regs, mode) {
            let a = r.lowerBound, c = r.upperBound
            guard let md else {
                out += b[a..<c]
                continue
            }
            if md.split {
                for j in a..<c {
                    for m in 0..<md.k { out.append(b[j] + (b[j + 1] - b[j]) * Double(m) / Double(md.k)) }
                }
            } else {
                let k = md.k
                let downs = (a..<c).filter { i in downbeats.contains { abs($0 - b[i]) < 0.05 } }
                var phase = 0
                if !downs.isEmpty {
                    let counts = (0..<k).map { ph in downs.filter { ($0 - a) % k == ph }.count }
                    phase = counts.firstIndex(of: counts.max()!)!
                }
                out += stride(from: a + phase, to: c, by: k).map { b[$0] }
            }
        }
        out.append(b[b.count - 1])
        let rounded = out.map { ($0 * 1e6).rounded(.toNearestOrEven) / 1e6 }
        return Array(Set(rounded)).sorted()
    }

    /// Splits bars of k * M beats (M = dominant meter) into k bars: the beat
    /// tracker missed a downbeat.
    public static func splitMergedBars(beats: [Double], downbeats: [Double]) -> [Double] {
        guard !beats.isEmpty, !downbeats.isEmpty else { return downbeats }
        let idx = Array(Set(downbeats.map { BeatTracking.nearestIndex(beats, $0) })).sorted()
        let lens = zip(idx.dropFirst() + [beats.count], idx).map { $0 - $1 }
        var weight = [Double](repeating: 0, count: (lens.max() ?? 0) + 1)
        for l in lens { weight[l] += Double(l) }
        let m = weight.firstIndex(of: weight.max()!)!
        var out = [Int]()
        for (a, len) in zip(idx, lens) {
            if m >= 2 && len >= 2 * m && len % m == 0 {
                out += stride(from: a, to: a + len, by: m)
            } else {
                out.append(a)
            }
        }
        return out.map { beats[$0] }
    }

    public static func regularize(beats: [Double], downbeats: [Double]) -> (beats: [Double], downbeats: [Double]) {
        let b = unifyBeatLevels(beats, downbeats: downbeats)
        return (b, splitMergedBars(beats: b, downbeats: downbeats))
    }

    /// Bar start times: the downbeats, plus the first beat when there is a pickup.
    public static func barStarts(beats: [Double], downbeats: [Double]) -> [Double] {
        guard let first = downbeats.first else { return stride(from: 0, to: beats.count, by: 4).map { beats[$0] } }
        if let b0 = beats.first, b0 < first - 0.01 { return [b0] + downbeats }
        return downbeats
    }

    static func songEnd(beats: [Double], duration: Double) -> Double {
        guard beats.count >= 2 else { return duration }
        let tail = Array(beats.suffix(9))
        let ibi = zip(tail.dropFirst(), tail).map { $0 - $1 }
        return min(duration, beats[beats.count - 1] + BeatTracking.median(ibi))
    }

    public static func rhythmSections(beats: [Double], downbeats: [Double], duration: Double,
                                      sigma: Double = 0.025, beta: Double = 20, merge: Double = 0.04,
                                      minBars: Int = 2) -> [RhythmSection] {
        let starts = barStarts(beats: beats, downbeats: downbeats)
        let n = starts.count
        guard n > 0 else { return [] }
        let ends = Array(starts.dropFirst()) + [songEnd(beats: beats, duration: duration)]
        let counts = zip(starts, ends).map { s, e in
            max(1, beats.filter { $0 >= s - 1e-6 && $0 < e - 1e-6 }.count)
        }
        let bpm = (0..<n).map { 60 * Double(counts[$0]) / max(ends[$0] - starts[$0], 1e-3) }

        // a change of meter has to last at least `minBars` bars
        var met = counts
        var runs = [(a: Int, c: Int, m: Int)]()
        var s = 0
        for i in 1...n where i == n || met[i] != met[s] {
            runs.append((s, i, met[s]))
            s = i
        }
        let longs = runs.filter { $0.c - $0.a >= minBars }
        for r in runs.indices where runs[r].c - runs[r].a < minBars && !longs.isEmpty {
            let before = longs.filter { $0.c <= runs[r].a }
            runs[r].m = before.last?.m ?? longs[0].m
        }
        for r in runs { for i in r.a..<r.c { met[i] = r.m } }

        let pieces = segmentSeries(bpm.map { log($0) }, sigma: sigma, beta: beta)
        var bounds = Set([0, n])
        for p in pieces { bounds.insert(p.lowerBound) }
        for i in 1..<max(n, 1) where met[i] != met[i - 1] { bounds.insert(i) }
        let sorted = bounds.sorted()
        func medianTempo(_ a: Int, _ c: Int) -> Double { exp(BeatTracking.median(bpm[a..<c].map { log($0) })) }

        var out = [RhythmSection]()
        for (a, c) in zip(sorted, sorted.dropFirst()) {
            let tempo = medianTempo(a, c)
            let m = met[a]
            if let last = out.last,
               c - a < minBars || (last.beatsPerBar == m && abs(log(tempo / last.bpm)) < merge) {
                out[out.count - 1] = RhythmSection(start: starts[last.firstBar], end: ends[c - 1], firstBar: last.firstBar,
                                                   bpm: medianTempo(last.firstBar, c), beatsPerBar: last.beatsPerBar)
            } else {
                out.append(RhythmSection(start: starts[a], end: ends[c - 1], firstBar: a, bpm: tempo, beatsPerBar: m))
            }
        }
        return out
    }

    // MARK: - keys

    static let majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    static let minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    /// One chord as absolute pitch classes; root -1 means no chord.
    public struct ChordSpan: Sendable {
        public var start: Double, end: Double, tones: [Int], bass: Int, root: Int
        public init(start: Double, end: Double, tones: [Int], bass: Int, root: Int) {
            self.start = start; self.end = end; self.tones = tones; self.bass = bass; self.root = root
        }
    }

    /// Local keys by Viterbi decoding over bars: Krumhansl-Kessler profile
    /// correlation of the chord tones around each bar, and a cost for every key
    /// change (higher between relative major and minor).
    public static func keyRegions(chords: [ChordSpan], beats: [Double], downbeats: [Double], duration: Double,
                                  switchCost: Double = 3, relativeCost: Double = 1.5, smooth: Int = 2) -> [KeyRegion] {
        var starts = barStarts(beats: beats, downbeats: downbeats)
        if starts.isEmpty { starts = [0] }
        let end = songEnd(beats: beats, duration: duration)
        let ends = Array(starts.dropFirst()) + [max(end, starts[starts.count - 1] + 1e-3)]
        let n = starts.count
        var hist = [[Double]](repeating: [Double](repeating: 0, count: 12), count: n)
        for i in 0..<n {
            for c in chords {
                let o = min(c.end, ends[i]) - max(c.start, starts[i])
                if o <= 0 || c.root < 0 { continue }
                for t in c.tones { hist[i][t] += o }
                hist[i][c.bass] += 0.5 * o
                hist[i][c.root] += 0.5 * o
            }
        }
        let profiles = (0..<24).map { k -> [Double] in
            let p = k < 12 ? majorProfile : minorProfile
            return (0..<12).map { p[($0 - k % 12 + 12) % 12] }
        }
        var emis = [[Double]](repeating: [Double](repeating: 0, count: 24), count: n)
        for i in 0..<n {
            var h = [Double](repeating: 0, count: 12)
            for j in max(0, i - smooth)..<min(n, i + smooth + 1) {
                let w = Double(smooth + 1 - abs(j - i))
                for p in 0..<12 { h[p] += hist[j][p] * w }
            }
            if h.reduce(0, +) > 0 { emis[i] = profiles.map { MusicalKey.correlation(h, $0) } }
        }
        func cost(_ a: Int, _ b: Int) -> Double {
            if a == b { return 0 }
            let ta = a % 12, tb = b % 12, ma = a >= 12, mb = b >= 12
            let relative = (!ma && mb && tb == (ta + 9) % 12) || (ma && !mb && tb == (ta + 3) % 12)
            return switchCost + (relative ? relativeCost : 0)
        }
        var score = emis[0]
        var back = [[Int]](repeating: [Int](repeating: 0, count: 24), count: n)
        for i in 1..<max(n, 1) {
            var next = [Double](repeating: 0, count: 24)
            for b in 0..<24 {
                var bestA = 0, bestV = -Double.infinity
                for a in 0..<24 {
                    let v = score[a] - cost(a, b)
                    if v > bestV { bestV = v; bestA = a }
                }
                back[i][b] = bestA
                next[b] = bestV + emis[i][b]
            }
            score = next
        }
        var path = [Int](repeating: 0, count: n)
        path[n - 1] = score.firstIndex(of: score.max()!)!
        if n > 1 { for i in stride(from: n - 1, to: 0, by: -1) { path[i - 1] = back[i][path[i]] } }
        var out = [KeyRegion]()
        var s = 0
        for i in 1...n where i == n || path[i] != path[s] {
            out.append(KeyRegion(start: s == 0 ? 0 : starts[s], end: i == n ? duration : starts[i],
                                 key: MusicalKey(tonic: path[s] % 12, minor: path[s] >= 12)))
            s = i
        }
        return out
    }

    /// Key regions for decoded chord segments.
    public static func keyRegions(segments: [ChordSegment], vocabulary v: ChordVocabulary, beats: [Double],
                                  downbeats: [Double], duration: Double) -> [KeyRegion] {
        let spans = segments.map { s -> ChordSpan in
            guard let pcs = v.pitchClasses(for: s) else { return ChordSpan(start: s.start, end: s.end, tones: [], bass: 0, root: -1) }
            return ChordSpan(start: s.start, end: s.end, tones: Array(Set(pcs.tones)).sorted(), bass: pcs.bass, root: s.root)
        }
        return keyRegions(chords: spans, beats: beats, downbeats: downbeats, duration: duration)
    }
}
