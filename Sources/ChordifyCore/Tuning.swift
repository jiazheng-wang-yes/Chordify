import Accelerate
import Foundation

/// Estimates how far a recording is tuned from A = 440 Hz, in cents
/// (similar in spirit to librosa.estimate_tuning: spectral peaks, parabolic
/// interpolation, histogram of deviations from the equal-tempered grid).
public enum Tuning {
    public static func estimateCents(_ x: [Float], sampleRate sr: Double, maxFrames: Int = 300) -> Double {
        let log2n = 12
        let n = 1 << log2n
        guard x.count > n * 4, let setup = vDSP_create_fftsetup(vDSP_Length(log2n), FFTRadix(kFFTRadix2)) else { return 0 }
        defer { vDSP_destroy_fftsetup(setup) }

        var window = [Float](repeating: 0, count: n)
        vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_NORM))
        let hop = max(n / 2, (x.count - n) / maxFrames)
        var hist = [Double](repeating: 0, count: 100)  // 1-cent bins, -50 ..< 50
        var frame = [Float](repeating: 0, count: n)
        var re = [Float](repeating: 0, count: n / 2)
        var im = [Float](repeating: 0, count: n / 2)
        var mag = [Float](repeating: 0, count: n / 2)
        let kLo = Int(80.0 / sr * Double(n)), kHi = min(n / 2 - 2, Int(2000.0 / sr * Double(n)))

        var start = 0
        while start + n <= x.count {
            x.withUnsafeBufferPointer { xp in
                vDSP_vmul(xp.baseAddress! + start, 1, window, 1, &frame, 1, vDSP_Length(n))
            }
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBufferPointer { fp in
                        fp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n / 2) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(n / 2))
                        }
                    }
                    vDSP_fft_zrip(setup, &split, 1, vDSP_Length(log2n), FFTDirection(FFT_FORWARD))
                    vDSP_zvabs(&split, 1, &mag, 1, vDSP_Length(n / 2))
                }
            }
            mag[0] = 0
            let peak = mag[kLo...kHi].max() ?? 0
            if peak > 1e-3 {
                let threshold = 0.1 * peak
                for k in (kLo + 1)..<kHi where mag[k] > threshold && mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] {
                    // parabolic interpolation on log magnitude
                    let a = log(Double(mag[k - 1]) + 1e-9), b = log(Double(mag[k]) + 1e-9), c = log(Double(mag[k + 1]) + 1e-9)
                    let denom = a - 2 * b + c
                    let delta = denom == 0 ? 0 : 0.5 * (a - c) / denom
                    let f = (Double(k) + delta) * sr / Double(n)
                    let cents = 1200 * log2(f / 440)
                    var dev = cents - 100 * (cents / 100).rounded()
                    if dev >= 50 { dev -= 100 }
                    let bin = min(99, max(0, Int((dev + 50).rounded(.down))))
                    hist[bin] += Double(mag[k] / peak)
                }
            }
            start += hop
        }
        guard hist.reduce(0, +) > 0 else { return 0 }
        // circular smoothing over +/- 5 cents, then the mode
        var best = 0, bestValue = -1.0
        for i in 0..<100 {
            var s = 0.0
            for d in -5...5 { s += hist[(i + d + 100) % 100] * (6 - Double(abs(d))) }
            if s > bestValue { bestValue = s; best = i }
        }
        return Double(best) - 50 + 0.5
    }
}
