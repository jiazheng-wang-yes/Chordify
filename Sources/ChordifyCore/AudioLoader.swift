import AVFoundation
import Foundation

public enum AudioLoader {
    /// Decodes any format Core Audio can read (mp3, m4a/aac, alac, wav, aiff,
    /// flac, caf, ...), mixes to mono and resamples to `sampleRate`.
    public static func loadMono(url: URL, sampleRate: Double) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw ChordifyError.audio("Could not open \(url.lastPathComponent): \(error.localizedDescription)")
        }
        let format = file.processingFormat
        let total = AVAudioFrameCount(file.length)
        guard total > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: total) else {
            throw ChordifyError.audio("\(url.lastPathComponent) contains no audio.")
        }
        try file.read(into: buffer)
        let n = Int(buffer.frameLength)
        let channels = Int(format.channelCount)
        guard let data = buffer.floatChannelData else {
            throw ChordifyError.audio("Unsupported sample format in \(url.lastPathComponent).")
        }
        var mono = [Float](repeating: 0, count: n)
        let scale = 1 / Float(channels)
        for ch in 0..<channels {
            let p = data[ch]
            for i in 0..<n { mono[i] += p[i] * scale }
        }
        if abs(format.sampleRate - sampleRate) < 1e-6 { return mono }
        return try resample(mono, from: format.sampleRate, to: sampleRate)
    }

    /// High-quality sample-rate conversion with AVAudioConverter.
    public static func resample(_ x: [Float], from src: Double, to dst: Double) throws -> [Float] {
        guard let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: src, channels: 1, interleaved: false),
              let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: dst, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFmt, to: outFmt),
              let input = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(max(x.count, 1)))
        else { throw ChordifyError.audio("Could not create a sample-rate converter.") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        input.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }

        let expected = Int((Double(x.count) * dst / src).rounded())
        var out = [Float]()
        out.reserveCapacity(expected + 1024)
        let chunk: AVAudioFrameCount = 1 << 16
        var fed = false
        while true {
            guard let buf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: chunk) else { break }
            var err: NSError?
            let status = converter.convert(to: buf, error: &err) { _, outStatus in
                if fed {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return input
            }
            if let err { throw ChordifyError.audio("Resampling failed: \(err.localizedDescription)") }
            let got = Int(buf.frameLength)
            if got > 0 {
                out.append(contentsOf: UnsafeBufferPointer(start: buf.floatChannelData![0], count: got))
            }
            if status == .endOfStream || status == .error || (got == 0 && fed) { break }
        }
        // Keep the nominal length so frame counts match the Python reference.
        if out.count > expected { out.removeLast(out.count - expected) }
        while out.count < expected { out.append(0) }
        return out
    }
}
