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
            case .downloading(let status, let fraction):
                DownloadingView(status: status, fraction: fraction)
            case .analyzing(let stage):
                AnalyzingView(title: model.title, stage: stage)
            case .failed(let message):
                FailedView(message: message)
            case .ready:
                SongView()
            }
        }
        .tint(Theme.ink)
        .sheet(isPresented: Bindable(model).showLinkSheet) { LinkSheet() }
        .toolbar { toolbarItems }
        .navigationTitle(model.title.isEmpty ? "Chordify" : model.title)
        .onDrop(of: [.fileURL, .url], isTargeted: $dropTargeted) { providers in
            guard let p = providers.first else { return false }
            _ = p.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    // a dragged web link (from a browser) is analysed as a link
                    if url.isFileURL { model.open(url) } else { model.openLink(url.absoluteString) }
                }
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
                    .foregroundStyle(targeted ? Theme.ink : Color.secondary.opacity(0.35))
                    .background(
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .fill(targeted ? Theme.ink.opacity(0.08) : Color.clear)
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
                    HStack(spacing: 10) {
                        Button("Choose a File\u{2026}") { model.showOpenPanel() }
                            .buttonStyle(.borderedProminent)
                        Button("Open a Link\u{2026}") { model.showLinkSheet = true }
                            .buttonStyle(.bordered)
                    }
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

struct DownloadingView: View {
    @Environment(AppModel.self) private var model
    var status: String
    var fraction: Double?

    var body: some View {
        VStack(spacing: 18) {
            Text(model.title)
                .font(.system(size: 22, weight: .semibold, design: .rounded))
                .lineLimit(1)
            if let fraction {
                ProgressView(value: fraction).frame(width: 300)
            } else {
                ProgressView().controlSize(.regular)
            }
            Text(status + (fraction.map { String(format: " %.0f%%", $0 * 100) } ?? ""))
                .foregroundStyle(.secondary)
            Button("Cancel") { model.cancelDownload() }
        }
        .padding(40)
    }
}

/// Paste a link to a song or video (beta).
struct LinkSheet: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Text("Analyze a Link").font(.title2.weight(.semibold))
                Text("BETA")
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .overlay(Capsule().strokeBorder(.secondary))
                    .foregroundStyle(.secondary)
            }
            TextField("https://www.youtube.com/watch?v=\u{2026}", text: $model.linkText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 440)
                .focused($focused)
                .onSubmit { model.openLink(model.linkText) }
            Text("""
                YouTube, SoundCloud, Bandcamp, Vimeo and most pages with audio or video work, \
                as do direct links to audio files. Downloads use the open-source yt-dlp tool \
                (installed once, about 35 MB, if it is not already on your Mac). \
                Only analyse recordings you are allowed to download.
                """)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 440, alignment: .leading)
            HStack {
                Spacer()
                Button("Cancel") { model.showLinkSheet = false }
                    .keyboardShortcut(.cancelAction)
                Button("Analyze") { model.openLink(model.linkText) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(LinkDownloader.normalise(model.linkText) == nil)
            }
        }
        .padding(22)
        .onAppear {
            if model.linkText.isEmpty, let s = NSPasteboard.general.string(forType: .string),
               LinkDownloader.normalise(s) != nil, s.contains(".") {
                model.linkText = s.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            focused = true
        }
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
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.ink)
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
            HStack {
                Button("Open Another Song\u{2026}") { model.showOpenPanel() }
                Button("Open a Link\u{2026}") { model.showLinkSheet = true }
            }
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
                    Text("1\u{2009}4\u{2009}5").tag(ChordStyle.nashville)
                }
                .pickerStyle(.segmented)
                .help("Jazz: every extension and slash bass. Simple: triads and sevenths. 1 4 5: Nashville numbers in the local key (\u{2318}J cycles)")

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
                model.showLinkSheet = true
            } label: {
                Label("Open Link", systemImage: "link")
            }
            .help("Analyze a song or video from a link, beta (\u{2318}L)")
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
