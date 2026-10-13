import ChordifyCore
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var dropTargeted = false

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            switch model.phase {
            case .empty:
                EmptyStateView(targeted: dropTargeted)
            case .analyzing(let stage):
                AnalyzingView(title: model.title, stage: stage)
            case .failed(let message):
                FailedView(message: message)
            case .ready:
                SongView()
            }
        }
        .toolbar { toolbarItems }
        .navigationTitle(model.title.isEmpty ? "Chordify" : model.title)
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            guard let p = providers.first else { return false }
            _ = p.loadObject(ofClass: URL.self) { url, _ in
                if let url { Task { @MainActor in model.open(url) } }
            }
            return true
        }
    }
}

// MARK: - states

struct EmptyStateView: View {
    @Environment(AppModel.self) private var model
    var targeted: Bool

    var body: some View {
        VStack(spacing: 22) {
            ZStack {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8, 7]))
                    .foregroundStyle(targeted ? Color.accentColor : Color.secondary.opacity(0.35))
                    .background(
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .fill(targeted ? Color.accentColor.opacity(0.08) : Color.clear)
                    )
                VStack(spacing: 14) {
                    Image(systemName: "waveform.badge.magnifyingglass")
                        .font(.system(size: 54, weight: .light))
                        .foregroundStyle(Theme.accentGradient)
                    Text("Drop a song")
                        .font(.system(size: 28, weight: .semibold, design: .rounded))
                    Text("MP3, M4A, WAV, AIFF, FLAC, ALAC")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Button("Choose a File\u{2026}") { model.showOpenPanel() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .padding(.top, 6)
                }
            }
            .frame(width: 440, height: 300)
            Text("Chords, beats and bars are found on your Mac. Nothing is uploaded.")
                .font(.footnote)
                .foregroundStyle(.tertiary)
        }
        .animation(.easeOut(duration: 0.15), value: targeted)
    }
}

struct AnalyzingView: View {
    var title: String
    var stage: Analyzer.Stage
    private let stages: [Analyzer.Stage] = [.decoding, .tuning, .listening, .finishing]

    var body: some View {
        VStack(spacing: 26) {
            Text(title)
                .font(.system(size: 26, weight: .semibold, design: .rounded))
                .lineLimit(1)
            VStack(alignment: .leading, spacing: 12) {
                ForEach(stages, id: \.self) { s in
                    let index = stages.firstIndex(of: s)!
                    let current = stages.firstIndex(of: stage)!
                    HStack(spacing: 12) {
                        Group {
                            if index < current {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accentColor)
                            } else if index == current {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "circle").foregroundStyle(.tertiary)
                            }
                        }
                        .frame(width: 18)
                        Text(s.rawValue)
                            .foregroundStyle(index <= current ? .primary : .tertiary)
                    }
                }
            }
            .font(.body)
        }
        .padding(40)
    }
}

struct FailedView: View {
    @Environment(AppModel.self) private var model
    var message: String

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundStyle(.orange)
            Text(message)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            Button("Open Another Song\u{2026}") { model.showOpenPanel() }
        }
    }
}

// MARK: - toolbar

extension ContentView {
    @ToolbarContentBuilder
    var toolbarItems: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            if model.analysis != nil {
                Picker("Chord symbols", selection: Bindable(model).style) {
                    Text("Jazz").tag(ChordStyle.jazz)
                    Text("Simple").tag(ChordStyle.simple)
                }
                .pickerStyle(.segmented)
                .help("Jazz shows every extension and slash bass (\u{2318}J)")

                Menu {
                    Picker("Extensions", selection: Bindable(model).extensions) {
                        ForEach(AppModel.Extensions.allCases) { Text($0.title).tag($0) }
                    }
                    .pickerStyle(.inline)
                } label: {
                    Label("Extensions", systemImage: "slider.horizontal.3")
                }
                .help("How readily 9ths, 11ths and 13ths are reported")

                TransposeControl()

                Menu {
                    Button("MIDI File (chords + bass)\u{2026}") { model.exportMIDI() }
                    Button("Chord Chart (text)\u{2026}") { model.exportChart() }
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .help("Export MIDI (\u{2318}E)")
            }
            Button {
                model.showOpenPanel()
            } label: {
                Label("Open", systemImage: "folder")
            }
            .help("Open a song (\u{2318}O)")
        }
    }
}

struct TransposeControl: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 2) {
            Button { model.transpose = max(-11, model.transpose - 1) } label: { Image(systemName: "minus") }
            Text(model.transpose == 0 ? "Key" : (model.transpose > 0 ? "+\(model.transpose)" : "\(model.transpose)"))
                .monospacedDigit()
                .frame(minWidth: 30)
                .onTapGesture(count: 2) { model.transpose = 0 }
            Button { model.transpose = min(11, model.transpose + 1) } label: { Image(systemName: "plus") }
        }
        .buttonStyle(.borderless)
        .help("Transpose (\u{2318}\u{2191} / \u{2318}\u{2193}); double-click to reset")
    }
}
