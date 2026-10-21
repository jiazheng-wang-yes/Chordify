import ChordifyCore
import Foundation

/// Terminal mode: prints the chord chart and timings, optionally writes MIDI.
enum CommandLineTool {
    static func run(_ args: [String]) -> Int32 {
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        let url: URL
        if let link = value("--analyze-url") {
            guard let web = LinkDownloader.normalise(link) else {
                FileHandle.standardError.write(Data("error: not a web link: \(link)\n".utf8))
                return 2
            }
            print("Fetching \(web.absoluteString)")
            let box = FetchBox()
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                do {
                    box.result = .success(try await LinkDownloader().fetch(web, installIfNeeded: true) { p in
                        switch p {
                        case .installingTool: print("  installing yt-dlp")
                        case .resolving: print("  resolving")
                        case .downloading(let f): if let f, Int(f * 100) % 25 == 0 { print(String(format: "  %.0f%%", f * 100)) }
                        case .converting: print("  converting")
                        }
                    })
                } catch {
                    box.result = .failure(error)
                }
                done.signal()
            }
            done.wait()
            switch box.result! {
            case .success(let r):
                print("Downloaded \"\(r.title)\" to \(r.file.path)")
                url = r.file
            case .failure(let e):
                FileHandle.standardError.write(Data("error: \(e.localizedDescription)\n".utf8))
                return 1
            }
        } else if let path = value("--analyze") {
            url = URL(fileURLWithPath: path)
        } else {
            return 2
        }
        do {
            let t0 = Date()
            let analyzer = try Analyzer()
            let clock = StageClock()
            let (a, _) = try analyzer.analyze(url: url) { stage in clock.enter(stage) }
            clock.enter(nil)
            print(String(format: "Analysed %.0f s of audio in %.1f s", a.duration, Date().timeIntervalSince(t0)))
            let v = analyzer.vocabulary
            let style: ChordStyle = args.contains("--nashville") ? .nashville : .jazz
            print("Key: \(a.key.map { PitchSpelling.ascii($0.name()) } ?? "-")" +
                  String(format: "   Tempo: %.1f BPM   Tuning: %+.1f cents", a.tempo ?? 0, a.tuningCents))
            for r in a.keyRegions {
                print(String(format: "  key   %7.2f - %7.2f s  ", r.start, r.end) + PitchSpelling.ascii(r.key.name()))
            }
            for s in a.sections {
                print(String(format: "  tempo %7.2f - %7.2f s  %.1f BPM, %ld beats per bar", s.start, s.end, s.bpm, s.beatsPerBar))
            }
            print(ChordChart.text(analysis: a, vocabulary: v, style: style, transpose: 0))
            if let out = value("--midi") {
                try MIDIExport.data(for: a, vocabulary: v).write(to: URL(fileURLWithPath: out))
                print("MIDI written to \(out)")
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            return 1
        }
    }
}

/// Prints how long each analysis stage took.
final class StageClock: @unchecked Sendable {
    private var last = Date()
    private var stage: Analyzer.Stage?

    func enter(_ next: Analyzer.Stage?) {
        let now = Date()
        if let s = stage {
            print("  " + s.rawValue.padding(toLength: 24, withPad: " ", startingAt: 0) + String(format: "%6.2f s", now.timeIntervalSince(last)))
        }
        stage = next
        last = now
    }
}

private final class FetchBox: @unchecked Sendable {
    var result: Swift.Result<LinkDownloader.Result, Error>?
}
