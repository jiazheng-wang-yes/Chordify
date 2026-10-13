import AppKit
import ChordifyCore
import Foundation
import Observation
import UniformTypeIdentifiers

@MainActor
@Observable
final class AppModel {
    enum Phase {
        case empty
        case analyzing(Analyzer.Stage)
        case ready
        case failed(String)
    }

    var phase: Phase = .empty
    var fileURL: URL?
    var title = ""
    var analysis: SongAnalysis?
    var grid: BarGrid?

    // playback
    var isPlaying = false
    var currentTime = 0.0
    /// (bar, cell) under the playhead; only changes once per beat.
    var position: (bar: Int, cell: Int)?
    var currentSegment: Int?
    var speed: Double = 1 { didSet { player.rate = Float(speed) } }

    // display
    enum Extensions: Float, CaseIterable, Identifiable {
        case asTrained = 0, normal = 1, rich = 2
        var id: Float { rawValue }
        var title: String {
            switch self {
            case .asTrained: return "Fewer (as trained)"
            case .normal: return "Normal"
            case .rich: return "More (rich harmony)"
            }
        }
    }
    var extensions: Extensions = .normal { didSet { if extensions != oldValue { redecode() } } }
    var style: ChordStyle = .jazz
    var transpose = 0 { didSet { applyPitch() } }
    var shiftAudio = true { didSet { applyPitch() } }

    private(set) var vocabulary: ChordVocabulary?
    private var analyzer: Analyzer?
    private var probabilities: ChordProbabilities?
    private var cacheKey: String?
    private let player = Player()
    private var timer: Timer?
    private var analysisTask: Task<Void, Never>?

    init() {
        player.onFinish = { [weak self] in
            self?.isPlaying = false
            self?.stopTimer()
        }
    }

    var duration: Double { analysis?.duration ?? player.duration }

    var preferFlats: Bool? {
        guard let k = analysis?.key else { return nil }
        return MusicalKey(tonic: (k.tonic + transpose + 120) % 12, minor: k.minor).prefersFlats
    }

    func symbol(_ segment: Int?) -> ChordSymbol {
        guard let i = segment, let a = analysis, let v = vocabulary, i < a.segments.count else { return .noChord }
        return v.symbol(for: a.segments[i], style: style, transpose: transpose, preferFlats: preferFlats)
    }

    func pitchClasses(_ segment: Int?) -> (tones: [Int], bass: Int)? {
        guard let i = segment, let a = analysis, let v = vocabulary, i < a.segments.count else { return nil }
        return v.pitchClasses(for: a.segments[i], transpose: transpose)
    }

    /// The next segment with a different chord after `segment`.
    func nextSegment(after segment: Int?) -> Int? {
        guard let a = analysis else { return nil }
        let i = (segment ?? -1) + 1
        return i < a.segments.count ? i : nil
    }

    // MARK: - opening

    func open(_ url: URL) {
        analysisTask?.cancel()
        player.stop()
        stopTimer()
        isPlaying = false
        currentTime = 0
        position = nil
        currentSegment = nil
        transpose = 0
        fileURL = url
        title = url.deletingPathExtension().lastPathComponent
        analysis = nil
        grid = nil
        do {
            try player.load(url: url)
        } catch {
            phase = .failed("Could not open \(url.lastPathComponent).\n\(error.localizedDescription)")
            return
        }
        NSDocumentController.shared.noteNewRecentDocumentURL(url)
        phase = .analyzing(.decoding)
        analysisTask = Task { [weak self] in
            await self?.runAnalysis(url)
        }
    }

    private func runAnalysis(_ url: URL) async {
        do {
            if analyzer == nil {
                let a = try await Task.detached(priority: .userInitiated) { try Analyzer() }.value
                analyzer = a
                vocabulary = a.vocabulary
            }
            guard let analyzer else { return }
            let key = AnalysisCache.key(for: url)
            var result = key.flatMap(AnalysisCache.load)
            var probs = key.flatMap(AnalysisCache.loadProbabilities)
            if result == nil || probs == nil {
                let boost = extensions.rawValue
                let out = try await Task.detached(priority: .userInitiated) {
                    try analyzer.analyze(url: url, extensionBoost: boost) { stage in
                        Task { @MainActor [weak self] in
                            if case .analyzing = self?.phase { self?.phase = .analyzing(stage) }
                        }
                    }
                }.value
                result = out.analysis
                probs = out.probabilities
                if let key {
                    AnalysisCache.store(out.analysis, key: key)
                    AnalysisCache.store(out.probabilities, key: key)
                }
            }
            guard !Task.isCancelled, fileURL == url, var result else { return }
            probabilities = probs
            cacheKey = key
            if let probs, result.extensionBoost != extensions.rawValue {
                result = analyzer.redecode(result, probabilities: probs, extensionBoost: extensions.rawValue)
            }
            analysis = result
            grid = BarGrid(analysis: result)
            phase = .ready
            updatePosition()
        } catch {
            guard fileURL == url else { return }
            phase = .failed(error.localizedDescription)
        }
    }

    /// Decode again with the current extension setting (no network run).
    private func redecode() {
        guard let a = analysis, let p = probabilities, let analyzer else { return }
        let updated = analyzer.redecode(a, probabilities: p, extensionBoost: extensions.rawValue)
        analysis = updated
        grid = BarGrid(analysis: updated)
        if let cacheKey { AnalysisCache.store(updated, key: cacheKey) }
        updatePosition()
    }

    func showOpenPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a song"
        if panel.runModal() == .OK, let url = panel.url { open(url) }
    }

    // MARK: - transport

    func togglePlay() {
        guard case .ready = phase else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
            stopTimer()
            currentTime = player.currentTime
        } else {
            player.play()
            isPlaying = player.isPlaying
            if isPlaying { startTimer() }
        }
    }

    func seek(to t: Double) {
        player.seek(to: t)
        currentTime = player.currentTime
        updatePosition()
    }

    func skip(_ seconds: Double) { seek(to: max(0, min(duration, currentTime + seconds))) }

    /// Jump to the previous / next chord change.
    func jumpChord(_ direction: Int) {
        guard let a = analysis else { return }
        let i = a.segmentIndex(at: currentTime + 0.01) ?? 0
        if direction < 0 {
            let s = a.segments[i]
            // within the first half second of a chord, go to the one before
            let target = currentTime - s.start < 0.5 && i > 0 ? a.segments[i - 1].start : s.start
            seek(to: target)
        } else if i + 1 < a.segments.count {
            seek(to: a.segments[i + 1].start)
        }
    }

    private func applyPitch() {
        player.pitchShift = shiftAudio ? transpose : 0
    }

    private func startTimer() {
        stopTimer()
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        currentTime = player.currentTime
        updatePosition()
    }

    private func updatePosition() {
        let p = grid?.position(at: currentTime + 0.02)
        if p?.bar != position?.bar || p?.cell != position?.cell { position = p }
        let s = analysis?.segmentIndex(at: currentTime + 0.02)
        if s != currentSegment { currentSegment = s }
    }

    // MARK: - export

    func exportMIDI() {
        guard let a = analysis, let v = vocabulary else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "mid") ?? .midi]
        panel.nameFieldStringValue = title + (transpose != 0 ? " (\(transpose > 0 ? "+" : "")\(transpose))" : "") + ".mid"
        panel.message = "Chords and bass on a tempo map that follows the recording"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var options = MIDIExport.Options()
        options.transpose = transpose
        options.style = style
        do {
            try MIDIExport.data(for: a, vocabulary: v, options: options).write(to: url)
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    func exportChart() {
        guard let a = analysis, let g = grid else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = title + " chords.txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var lines = [title]
        if let k = a.key { lines.append("Key: \(PitchSpelling.ascii(k.name(transpose: transpose)))") }
        if let bpm = a.tempo { lines.append(String(format: "Tempo: %.0f BPM", bpm)) }
        lines.append("")
        var row = [String]()
        for bar in g.bars {
            let cells = bar.cells.filter(\.showsChord).map { PitchSpelling.ascii(symbol($0.segment).text) }
            row.append(cells.joined(separator: " "))
            if row.count == 4 {
                lines.append("| " + row.joined(separator: " | ") + " |")
                row = []
            }
        }
        if !row.isEmpty { lines.append("| " + row.joined(separator: " | ") + " |") }
        lines.append("")
        lines.append("Timed chords (seconds):")
        for (i, s) in a.segments.enumerated() {
            lines.append(String(format: "%8.2f  %8.2f  ", s.start, s.end) + PitchSpelling.ascii(symbol(i).text))
        }
        try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}
