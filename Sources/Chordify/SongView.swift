import AppKit
import ChordifyCore
import SwiftUI

enum Theme {
    static let background = Color(nsColor: .windowBackgroundColor)
    static let accentGradient = LinearGradient(colors: [Color(red: 0.36, green: 0.42, blue: 0.98), Color(red: 0.78, green: 0.36, blue: 0.92)],
                                               startPoint: .topLeading, endPoint: .bottomTrailing)

    /// Chord colour by root, arranged around the circle of fifths so that
    /// related chords get related colours.
    static func color(root: Int?, minor: Bool = false) -> Color {
        guard let root, root >= 0 else { return Color.gray.opacity(0.35) }
        let hue = Double((root * 7) % 12) / 12
        return Color(hue: hue, saturation: minor ? 0.42 : 0.55, brightness: minor ? 0.78 : 0.9)
    }
}

extension AppModel {
    func color(_ segment: Int?) -> Color {
        guard let i = segment, let a = analysis, let v = vocabulary, i < a.segments.count else { return Theme.color(root: nil) }
        let s = a.segments[i]
        if s.isNoChord { return Theme.color(root: nil) }
        let tones = v.types[s.typeIndex].tones
        return Theme.color(root: (s.root + transpose + 120) % 12, minor: tones.contains(3) && !tones.contains(4))
    }
}

struct SongView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            NowPanel()
                .padding(.horizontal, 24)
                .padding(.top, 14)
                .padding(.bottom, 10)
            Divider().opacity(0.5)
            ChordGridView()
            Divider().opacity(0.5)
            SongTimeline()
                .frame(height: 64)
                .padding(.horizontal, 20)
                .padding(.top, 10)
            TransportBar()
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
        }
    }
}

// MARK: - now playing

struct NowPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let current = model.currentSegment
        let next = model.nextSegment(after: current)
        HStack(alignment: .center, spacing: 28) {
            VStack(alignment: .leading, spacing: 4) {
                ChordText(symbol: model.symbol(current), size: 64)
                    .foregroundStyle(.primary)
                    .contentTransition(.opacity)
                    .animation(.easeOut(duration: 0.12), value: current)
                HStack(spacing: 6) {
                    Text("next")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.tertiary)
                        .textCase(.uppercase)
                    ChordText(symbol: model.symbol(next), size: 22)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(minWidth: 240, alignment: .leading)

            PianoView(chord: model.pitchClasses(current), color: model.color(current))
                .frame(height: 92)
                .frame(maxWidth: 520)

            Spacer(minLength: 0)
            InfoColumn()
        }
    }
}

struct InfoColumn: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            if let k = model.analysis?.key {
                Pill(icon: "key", text: k.name(transpose: model.transpose))
            }
            if let bpm = model.analysis?.tempo {
                Pill(icon: "metronome", text: String(format: "%.0f BPM", bpm))
            }
            if let g = model.grid {
                Pill(icon: "music.quarternote.3", text: "\(g.beatsPerBar) beats / bar")
            }
        }
    }
}

struct Pill: View {
    var icon: String
    var text: String
    var body: some View {
        Label(text, systemImage: icon)
            .font(.callout)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.6), in: Capsule())
    }
}

/// Chord symbol with a superscript-style suffix and slash bass.
struct ChordText: View {
    var symbol: ChordSymbol
    var size: CGFloat

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            Text(symbol.root)
                .font(.system(size: size, weight: .semibold, design: .rounded))
            if !symbol.suffix.isEmpty {
                Text(symbol.suffix)
                    .font(.system(size: size * 0.6, weight: .medium, design: .rounded))
                    .baselineOffset(size * 0.28)
            }
            if let bass = symbol.bass {
                Text("/" + bass)
                    .font(.system(size: size * 0.7, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.4)
    }
}

// MARK: - piano

struct PianoView: View {
    var chord: (tones: [Int], bass: Int)?
    var color: Color

    private let lowest = 36  // C2
    private let octaves = 4

    var body: some View {
        Canvas { ctx, size in
            let whiteCount = octaves * 7
            let w = size.width / CGFloat(whiteCount)
            let h = size.height
            var lit = Set<Int>(), bassNote: Int?
            if let c = chord {
                bassNote = 36 + c.bass
                lit = Set(Voicing.upper(tones: c.tones, bass: c.bass, previous: nil))
            }
            let whiteSteps = [0, 2, 4, 5, 7, 9, 11]
            let blackSteps: [(Int, CGFloat)] = [(1, 1), (3, 2), (6, 4), (8, 5), (10, 6)]
            for o in 0..<octaves {
                for (i, step) in whiteSteps.enumerated() {
                    let note = lowest + o * 12 + step
                    let rect = CGRect(x: CGFloat(o * 7 + i) * w + 0.5, y: 0, width: w - 1, height: h)
                    let path = Path(roundedRect: rect, cornerRadius: 3)
                    let fill: Color = note == bassNote ? color.opacity(0.95) : lit.contains(note) ? color : Color.white.opacity(0.92)
                    ctx.fill(path, with: .color(fill))
                    ctx.stroke(path, with: .color(.black.opacity(0.18)), lineWidth: 0.5)
                    if note == bassNote {
                        ctx.fill(Path(ellipseIn: CGRect(x: rect.midX - 3, y: h - 12, width: 6, height: 6)), with: .color(.black.opacity(0.55)))
                    }
                }
                for (step, pos) in blackSteps {
                    let note = lowest + o * 12 + step
                    let rect = CGRect(x: (CGFloat(o * 7) + pos) * w - w * 0.3, y: 0, width: w * 0.6, height: h * 0.6)
                    let path = Path(roundedRect: rect, cornerRadius: 2)
                    let fill: Color = note == bassNote || lit.contains(note) ? color : Color(white: 0.12)
                    ctx.fill(path, with: .color(fill))
                    if note == bassNote {
                        ctx.fill(Path(ellipseIn: CGRect(x: rect.midX - 2.5, y: rect.maxY - 10, width: 5, height: 5)), with: .color(.white.opacity(0.8)))
                    }
                }
            }
        }
        .accessibilityLabel("Piano showing the chord tones")
    }
}

// MARK: - bar grid

struct ChordGridView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let grid = model.grid {
            GeometryReader { geo in
                let perRow = max(2, min(8, Int(geo.size.width / 190)))
                let rows = stride(from: 0, to: grid.bars.count, by: perRow).map { Array(grid.bars[$0..<min($0 + perRow, grid.bars.count)]) }
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(rows.indices, id: \.self) { r in
                                HStack(spacing: 8) {
                                    ForEach(rows[r]) { bar in
                                        BarView(bar: bar, active: model.position?.bar == bar.id ? model.position?.cell : nil)
                                    }
                                    ForEach(0..<(perRow - rows[r].count), id: \.self) { _ in Color.clear.frame(maxWidth: .infinity) }
                                }
                                .id(r)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 14)
                    }
                    .onChange(of: model.position?.bar) { _, bar in
                        guard let bar, model.isPlaying else { return }
                        withAnimation(.easeInOut(duration: 0.35)) { proxy.scrollTo(bar / perRow, anchor: UnitPoint(x: 0.5, y: 0.35)) }
                    }
                }
            }
        } else {
            Spacer()
        }
    }
}

struct BarView: View {
    @Environment(AppModel.self) private var model
    var bar: BarGrid.Bar
    var active: Int?

    var body: some View {
        HStack(spacing: 0) {
            ForEach(bar.cells.indices, id: \.self) { i in
                let cell = bar.cells[i]
                ZStack {
                    if active == i {
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .fill(model.color(cell.segment).opacity(0.85))
                            .padding(2)
                    }
                    if cell.showsChord {
                        ChordText(symbol: model.symbol(cell.segment), size: 17)
                            .padding(.horizontal, 3)
                    } else {
                        Circle().fill(.tertiary).frame(width: 3, height: 3)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 46)
                .contentShape(Rectangle())
                .onTapGesture { model.seek(to: cell.time) }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.primary.opacity(active != nil ? 0.09 : 0.04))
        )
        .overlay(alignment: .leading) {
            RoundedRectangle(cornerRadius: 1).fill(model.color(bar.cells.first?.segment)).frame(width: 3).padding(.vertical, 8)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - timeline

struct SongTimeline: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let duration = max(model.duration, 0.01)
            ZStack(alignment: .topLeading) {
                WaveformCanvas()
                Playhead(width: width, duration: duration)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                model.seek(to: max(0, min(duration, v.location.x / width * duration)))
            })
        }
    }
}

/// Loudness envelope and coloured chord blocks; redrawn only when the
/// analysis or the transposition changes.
struct WaveformCanvas: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let analysis = model.analysis
        let colors = (0..<(analysis?.segments.count ?? 0)).map { model.color($0) }
        Canvas { ctx, size in
            guard let a = analysis else { return }
            let duration = max(a.duration, 0.01)
            let h = size.height
            for (i, s) in a.segments.enumerated() {
                let x0 = s.start / duration * size.width, x1 = s.end / duration * size.width
                let r = CGRect(x: x0, y: h - 10, width: max(1, x1 - x0 - 1), height: 8)
                ctx.fill(Path(roundedRect: r, cornerRadius: 2), with: .color(colors[i]))
            }
            let env = a.envelope
            guard !env.isEmpty else { return }
            let barW = size.width / CGFloat(env.count)
            let mid = (h - 14) / 2
            var path = Path()
            for (j, v) in env.enumerated() {
                let amp = CGFloat(v) * mid
                path.addRect(CGRect(x: CGFloat(j) * barW, y: mid - amp, width: max(barW, 1), height: amp * 2 + 0.5))
            }
            ctx.fill(path, with: .color(.secondary.opacity(0.45)))
        }
    }
}

struct Playhead: View {
    @Environment(AppModel.self) private var model
    var width: CGFloat
    var duration: Double

    var body: some View {
        let x = model.currentTime / duration * width
        ZStack(alignment: .topLeading) {
            Rectangle().fill(Color.accentColor.opacity(0.10)).frame(width: max(0, x))
            Rectangle().fill(Color.accentColor).frame(width: 2).offset(x: x - 1)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - transport

struct TransportBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 16) {
            Button { model.jumpChord(-1) } label: { Image(systemName: "backward.end.fill") }
                .help("Previous chord (\u{2318}\u{2190})")
            Button { model.togglePlay() } label: {
                Image(systemName: model.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 34))
                    .foregroundStyle(Theme.accentGradient)
            }
            .help("Play / pause (space)")
            Button { model.jumpChord(1) } label: { Image(systemName: "forward.end.fill") }
                .help("Next chord (\u{2318}\u{2192})")

            Text("\(format(model.currentTime)) / \(format(model.duration))")
                .font(.system(.callout, design: .rounded).monospacedDigit())
                .foregroundStyle(.secondary)

            Spacer()

            HStack(spacing: 8) {
                Image(systemName: "tortoise").foregroundStyle(.secondary)
                Slider(value: Bindable(model).speed, in: 0.5...1.25, step: 0.05)
                    .frame(width: 140)
                Image(systemName: "hare").foregroundStyle(.secondary)
                Text("\(Int((model.speed * 100).rounded()))%")
                    .font(.callout.monospacedDigit())
                    .frame(width: 44, alignment: .trailing)
                    .onTapGesture(count: 2) { model.speed = 1 }
            }
            .help("Playback speed, pitch preserved (- / = / 0)")
        }
        .buttonStyle(.borderless)
        .font(.title3)
    }

    private func format(_ t: Double) -> String {
        let s = Int(t.rounded(.down))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
