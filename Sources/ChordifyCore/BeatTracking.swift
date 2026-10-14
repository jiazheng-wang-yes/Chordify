import Foundation

/// Post-processing of the Beat This! frame logits (50 frames/s), following
/// beat_this.inference / beat_this.model.postprocessor ("minimal"), plus two
/// clean-up steps (gap filling, spurious downbeats) shared with
/// tools/convert/reference_pipeline.py.
public enum BeatTracking {
    public static let fps = 50.0
    public static let chunkSize = 1500
    public static let border = 6

    /// Chunk start frames, like beat_this.inference.split_piece(avoid_short_end=True).
    public static func chunkStarts(frames n: Int, chunk: Int = chunkSize, border: Int = border) -> [Int] {
        var starts = [Int]()
        var s = -border
        while s < n - border {
            starts.append(s)
            s += chunk - 2 * border
        }
        if starts.isEmpty { starts = [-border] }
        if n > chunk - 2 * border {
            starts[starts.count - 1] = n - (chunk - border)
        }
        return starts
    }

    /// Runs `model` on overlapping chunks of a (frames x 128) spectrogram and
    /// stitches the logits back together ("keep_first" overlap mode).
    public static func logits(spect: [Float], frames n: Int, bins: Int = 128,
                              model: ([Float], Int) throws -> (beat: [Float], downbeat: [Float])) rethrows
        -> (beat: [Float], downbeat: [Float])
    {
        var beat = [Float](repeating: -1000, count: n)
        var down = [Float](repeating: -1000, count: n)
        let starts = chunkStarts(frames: n)
        for s in starts.reversed() {
            let lo = max(s, 0), hi = min(s + chunkSize, n)
            let padL = max(0, -s)
            let padR = max(0, min(border, s + chunkSize - n))
            let len = padL + (hi - lo) + padR
            var x = [Float](repeating: 0, count: len * bins)
            if hi > lo {
                let src = spect[(lo * bins)..<(hi * bins)]
                x.replaceSubrange((padL * bins)..<((padL + hi - lo) * bins), with: src)
            }
            let (b, d) = try model(x, len)
            let a = s + border
            let usable = len - 2 * border
            guard usable > 0 else { continue }
            let e = min(a + usable, n)
            for i in 0..<(e - a) where a + i >= 0 {
                beat[a + i] = b[border + i]
                down[a + i] = d[border + i]
            }
        }
        return (beat, down)
    }

    /// Local maxima within +/- 3 frames with logit > 0, adjacent peaks merged
    /// (beat_this.model.postprocessor.deduplicate_peaks). Returns frame positions.
    public static func pickPeaks(_ logits: [Float]) -> [Double] {
        let n = logits.count
        var frames = [Int]()
        for t in 0..<n where logits[t] > 0 {
            var isMax = true
            for k in max(0, t - 3)...min(n - 1, t + 3) where logits[k] > logits[t] {
                isMax = false
                break
            }
            if isMax { frames.append(t) }
        }
        var out = [Double]()
        guard let first = frames.first else { return out }
        var p = Double(first), count = 1
        for f in frames.dropFirst() {
            if Double(f) - p <= 1 {
                count += 1
                p += (Double(f) - p) / Double(count)
            } else {
                out.append(p)
                p = Double(f)
                count = 1
            }
        }
        out.append(p)
        return out
    }

    /// Beat and downbeat times in seconds, downbeats snapped to the nearest beat.
    public static func postprocess(beat: [Float], downbeat: [Float]) -> (beats: [Double], downbeats: [Double]) {
        let bt = pickPeaks(beat).map { $0 / fps }
        var dt = pickPeaks(downbeat).map { $0 / fps }
        if !bt.isEmpty {
            dt = dt.map { d in bt.min(by: { abs($0 - d) < abs($1 - d) })! }
            dt = Array(Set(dt)).sorted()
        }
        return (bt, dt)
    }

    /// Insert beats the peak picker dropped: a gap close to k local beat
    /// periods (k >= 2) is split into k equal parts.
    public static func fillGaps(_ beats: [Double], tolerance: Double = 0.2) -> [Double] {
        guard beats.count >= 4 else { return beats }
        let ibis = zip(beats.dropFirst(), beats).map { $0 - $1 }
        var out = [beats[0]]
        for (i, gap) in ibis.enumerated() {
            let lo = max(0, i - 4), hi = min(ibis.count, i + 5)
            let local = median(Array(ibis[lo..<hi]))
            let k = Int((gap / local).rounded(.toNearestOrEven))
            if k >= 2 && abs(gap / Double(k) - local) < tolerance * local {
                for j in 1..<k { out.append(beats[i] + gap * Double(j) / Double(k)) }
            }
            out.append(beats[i + 1])
        }
        return out
    }

    /// Snap downbeats to beats and drop those that would make a bar shorter
    /// than half the typical bar.
    public static func cleanDownbeats(beats: [Double], downbeats: [Double]) -> [Double] {
        guard !beats.isEmpty, !downbeats.isEmpty else { return downbeats }
        let idx = Array(Set(downbeats.map { d in nearestIndex(beats, d) })).sorted()
        if idx.count < 3 { return idx.map { beats[$0] } }
        let typical = median(zip(idx.dropFirst(), idx).map { Double($0 - $1) })
        var keep = [idx[0]]
        for j in idx.dropFirst() {
            if Double(j - keep[keep.count - 1]) < typical / 2 {
                if keep.count == 1 { keep[0] = j }
                continue
            }
            keep.append(j)
        }
        return keep.map { beats[$0] }
    }

    static func nearestIndex(_ xs: [Double], _ v: Double) -> Int {
        var best = 0
        for i in 1..<xs.count where abs(xs[i] - v) < abs(xs[best] - v) { best = i }
        return best
    }

    /// numpy.median semantics (mean of the two middle values for even counts).
    static func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        let m = s.count / 2
        return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
    }
}
