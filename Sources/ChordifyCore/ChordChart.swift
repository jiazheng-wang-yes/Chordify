import Foundation

/// Plain-text chord chart: four bars per line, with a heading line wherever
/// the key, tempo or meter changes.
public enum ChordChart {
    public static func text(analysis a: SongAnalysis, vocabulary v: ChordVocabulary, style: ChordStyle,
                            transpose: Int, minorFromRelativeMajor: Bool = false, title: String? = nil) -> String {
        let grid = BarGrid(analysis: a)
        var lines = [String]()
        if let title { lines.append(title) }
        if let k = a.keyRegions.first?.key ?? a.key { lines.append("Key: " + PitchSpelling.ascii(k.name(transpose: transpose))) }
        if let s = a.sections.first {
            lines.append(String(format: "Tempo: %.0f BPM, %ld beats per bar", s.bpm, s.beatsPerBar))
        } else if let bpm = a.tempo {
            lines.append(String(format: "Tempo: %.0f BPM", bpm))
        }
        if style == .nashville { lines.append("Nashville numbers" + (minorFromRelativeMajor ? " (minor keys counted from the relative major)" : "")) }
        lines.append("")

        var row = [String]()
        func flush() {
            if !row.isEmpty { lines.append("| " + row.joined(separator: " | ") + " |") }
            row = []
        }
        for bar in grid.bars {
            if bar.keyChange != nil || bar.sectionChange != nil {
                flush()
                var parts = [String]()
                if let k = bar.keyChange { parts.append("Key: " + PitchSpelling.ascii(k.name(transpose: transpose))) }
                if let s = bar.sectionChange { parts.append(String(format: "%.0f BPM, %ld beats per bar", s.bpm, s.beatsPerBar)) }
                lines.append("")
                lines.append("[" + parts.joined(separator: ", ") + String(format: " at %ld:%02ld]", Int(bar.start) / 60, Int(bar.start) % 60))
            }
            let names = bar.cells.filter(\.showsChord).map { cell -> String in
                guard let i = cell.segment else { return "N.C." }
                let s = a.segments[i]
                return PitchSpelling.ascii(v.symbol(for: s, style: style, transpose: transpose, key: a.key(at: s.start),
                                                    minorFromRelativeMajor: minorFromRelativeMajor).text)
            }
            row.append(names.joined(separator: " "))
            if row.count == 4 { flush() }
        }
        flush()
        lines.append("")
        lines.append("Timed chords (seconds):")
        for s in a.segments {
            let name = PitchSpelling.ascii(v.symbol(for: s, style: style, transpose: transpose, key: a.key(at: s.start),
                                                    minorFromRelativeMajor: minorFromRelativeMajor).text)
            lines.append(String(format: "%8.2f  %8.2f  ", s.start, s.end) + name)
        }
        return lines.joined(separator: "\n")
    }
}
