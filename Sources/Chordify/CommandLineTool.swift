import ChordifyCore
import Foundation

/// Terminal mode: prints the chord chart and timings, optionally writes MIDI.
enum CommandLineTool {
    static func run(_ args: [String]) -> Int32 {
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        guard let path = value("--analyze") else { return 2 }
        let url = URL(fileURLWithPath: path)
        do {
            let t0 = Date()
            let analyzer = try Analyzer()
            let clock = StageClock()
            let (a, _) = try analyzer.analyze(url: url) { stage in clock.enter(stage) }
            clock.enter(nil)
            print(String(format: "Analysed %.0f s of audio in %.1f s", a.duration, Date().timeIntervalSince(t0)))
            let v = analyzer.vocabulary
            let flats = a.key?.prefersFlats
            print("Key: \(a.key.map { PitchSpelling.ascii($0.name()) } ?? "-")" +
                  String(format: "   Tempo: %.1f BPM   Tuning: %+.1f cents", a.tempo ?? 0, a.tuningCents))
            let grid = BarGrid(analysis: a)
            var line = [String]()
            for bar in grid.bars {
                line.append(bar.cells.filter(\.showsChord).map {
                    guard let i = $0.segment else { return "N.C." }
                    return PitchSpelling.ascii(v.symbol(for: a.segments[i], style: .jazz, transpose: 0, preferFlats: flats).text)
                }.joined(separator: " "))
                if line.count == 4 {
                    print("| " + line.joined(separator: " | ") + " |")
                    line = []
                }
            }
            if !line.isEmpty { print("| " + line.joined(separator: " | ") + " |") }
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
