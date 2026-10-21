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
        case downloading(String, Double?)
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
    /// Local key and tempo/meter section under the playhead.
    var currentKey: MusicalKey?
    var currentSection: RhythmSection?
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
    /// Nashville numbers: count minor keys from the relative major (minor tonic = 6m).
    var minorFromRelativeMajor = false

    // links (beta)
    var showLinkSheet = false
    var linkText = ""
    private let downloader = LinkDownloader()
    private var downloadTask: Task<Void, Never>?

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

    func symbol(_ segment: Int?) -> ChordSymbol {
        guard let i = segment, let a = analysis, let v = vocabulary, i < a.segments.count else { return .noChord }
        let s = a.segments[i]
        return v.symbol(for: s, style: style, transpose: transpose, key: a.key(at: s.start),
                        minorFromRelativeMajor: minorFromRelativeMajor)
    }

    /// "E♭ major" for a key, after transposition.
    func keyName(_ k: MusicalKey?) -> String {
        k.map { $0.name(transpose: transpose) } ?? "-"
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

    func open(_ url: URL, title displayTitle: String? = nil) {
        downloadTask?.cancel()
        analysisTask?.cancel()
        player.stop()
        stopTimer()
        isPlaying = false
        currentTime = 0
        position = nil
        currentSegment = nil
        transpose = 0
        fileURL = url
        title = displayTitle ?? url.deletingPathExtension().lastPathComponent
        currentKey = nil
        currentSection = nil
        analysis = nil
        grid = nil
        do {
            try player.load(url: url)
        } catch {
            phase = .failed("Could not open \(url.lastPathComponent).\n\(error.localizedDescription)")
            return
        }
        if displayTitle == nil { NSDocumentController.shared.noteNewRecentDocumentURL(url) }
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

    // MARK: - links (beta)

    /// Downloads the audio behind a web link, then analyses it like a file.
    func openLink(_ text: String) {
        guard let url = LinkDownloader.normalise(text) else {
            phase = .failed("That does not look like a web link:\n\(text)")
            return
        }
        showLinkSheet = false
        analysisTask?.cancel()
        downloadTask?.cancel()
        player.stop()
        stopTimer()
        isPlaying = false
        analysis = nil
        grid = nil
        title = url.host ?? url.absoluteString
        phase = .downloading("Looking up the link", nil)
        let fetcher = self.downloader
        downloadTask = Task { [weak self] in
            do {
                let result = try await fetcher.fetch(url, installIfNeeded: true) { p in
                    Task { @MainActor [weak self] in
                        guard let self, case .downloading = self.phase else { return }
                        switch p {
                        case .installingTool: self.phase = .downloading("Installing the yt-dlp download tool (once)", nil)
                        case .resolving: self.phase = .downloading("Looking up the link", nil)
                        case .downloading(let f): self.phase = .downloading("Downloading audio", f)
                        case .converting: self.phase = .downloading("Converting audio", nil)
                        }
                    }
                }
                guard !Task.isCancelled else { return }
                self?.open(result.file, title: result.title)
            } catch is CancellationError {
                self?.phase = .empty
            } catch {
                guard !Task.isCancelled else { return }
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancelDownload() {
        downloader.cancel()
        downloadTask?.cancel()
        phase = .empty
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
        let k = analysis?.key(at: currentTime + 0.02)
        if k != currentKey { currentKey = k }
        let sec = analysis?.section(at: currentTime + 0.02)
        if sec != currentSection { currentSection = sec }
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
        options.minorFromRelativeMajor = minorFromRelativeMajor
        do {
            try MIDIExport.data(for: a, vocabulary: v, options: options).write(to: url)
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    func exportChart() {
        guard let a = analysis, let v = vocabulary else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = title + (style == .nashville ? " numbers.txt" : " chords.txt")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let text = ChordChart.text(analysis: a, vocabulary: v, style: style, transpose: transpose,
                                   minorFromRelativeMajor: minorFromRelativeMajor, title: title)
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
}
