import AVFoundation
import Foundation

/// Audio playback with variable speed (pitch preserved) and exact seeking.
final class Player {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let timePitch = AVAudioUnitTimePitch()
    private var file: AVAudioFile?
    private var sampleRate = 44100.0
    private var startOffset = 0.0
    private var segmentToken = 0

    private(set) var duration = 0.0
    private(set) var isPlaying = false
    var onFinish: (() -> Void)?

    init() {
        engine.attach(node)
        engine.attach(timePitch)
    }

    func load(url: URL) throws {
        stop()
        let f = try AVAudioFile(forReading: url)
        file = f
        sampleRate = f.processingFormat.sampleRate
        duration = Double(f.length) / sampleRate
        engine.disconnectNodeOutput(node)
        engine.disconnectNodeOutput(timePitch)
        engine.connect(node, to: timePitch, format: f.processingFormat)
        engine.connect(timePitch, to: engine.mainMixerNode, format: f.processingFormat)
        engine.prepare()
        startOffset = 0
    }

    var rate: Float {
        get { timePitch.rate }
        set { timePitch.rate = max(0.25, min(2, newValue)) }
    }

    /// Transposition of the playback in semitones.
    var pitchShift: Int {
        get { Int((timePitch.pitch / 100).rounded()) }
        set { timePitch.pitch = Float(newValue * 100) }
    }

    var currentTime: Double {
        guard isPlaying, let nodeTime = node.lastRenderTime, let playerTime = node.playerTime(forNodeTime: nodeTime) else {
            return startOffset
        }
        return min(duration, startOffset + Double(playerTime.sampleTime) / playerTime.sampleRate)
    }

    func play() {
        guard let file, !isPlaying else { return }
        if startOffset >= duration - 0.05 { startOffset = 0 }
        do {
            if !engine.isRunning { try engine.start() }
        } catch {
            return
        }
        schedule(file: file, from: startOffset)
        node.play()
        isPlaying = true
    }

    func pause() {
        guard isPlaying else { return }
        startOffset = currentTime
        segmentToken += 1
        node.stop()
        isPlaying = false
    }

    func stop() {
        segmentToken += 1
        node.stop()
        isPlaying = false
        startOffset = 0
    }

    func seek(to t: Double) {
        let wasPlaying = isPlaying
        if wasPlaying {
            segmentToken += 1
            node.stop()
            isPlaying = false
        }
        startOffset = max(0, min(duration, t))
        if wasPlaying { play() }
    }

    private func schedule(file: AVAudioFile, from t: Double) {
        let start = AVAudioFramePosition(t * sampleRate)
        let frames = AVAudioFrameCount(max(0, file.length - start))
        guard frames > 0 else { return }
        segmentToken += 1
        let token = segmentToken
        node.scheduleSegment(file, startingFrame: start, frameCount: frames, at: nil,
                             completionCallbackType: .dataPlayedBack) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.segmentToken == token, self.isPlaying else { return }
                self.isPlaying = false
                self.startOffset = self.duration
                self.onFinish?()
            }
        }
    }
}
