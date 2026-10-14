import AppKit
import ChordifyCore
import SwiftUI

/// `Chordify --analyze song.mp3 [--midi out.mid]` runs the analysis in the
/// terminal; anything else starts the app.
@main
enum Entry {
    static func main() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--analyze"), i + 1 < args.count {
            exit(CommandLineTool.run(args))
        }
        ChordifyApp.main()
    }
}

struct ChordifyApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AppModel.shared

    var body: some Scene {
        Window("Chordify", id: "main") {
            ContentView()
                .environment(model)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open\u{2026}") { model.showOpenPanel() }
                    .keyboardShortcut("o")
            }
            CommandGroup(after: .saveItem) {
                Button("Export MIDI\u{2026}") { model.exportMIDI() }
                    .keyboardShortcut("e")
                    .disabled(model.analysis == nil)
                Button("Export Chord Chart\u{2026}") { model.exportChart() }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                    .disabled(model.analysis == nil)
            }
            CommandMenu("Playback") {
                Button(model.isPlaying ? "Pause" : "Play") { model.togglePlay() }
                    .keyboardShortcut(.space, modifiers: [])
                Divider()
                Button("Back 5 Seconds") { model.skip(-5) }
                    .keyboardShortcut(.leftArrow, modifiers: [])
                Button("Forward 5 Seconds") { model.skip(5) }
                    .keyboardShortcut(.rightArrow, modifiers: [])
                Button("Previous Chord") { model.jumpChord(-1) }
                    .keyboardShortcut(.leftArrow, modifiers: [.command])
                Button("Next Chord") { model.jumpChord(1) }
                    .keyboardShortcut(.rightArrow, modifiers: [.command])
                Divider()
                Button("Slower") { model.speed = max(0.5, model.speed - 0.1) }
                    .keyboardShortcut("-", modifiers: [])
                Button("Faster") { model.speed = min(1.5, model.speed + 0.1) }
                    .keyboardShortcut("=", modifiers: [])
                Button("Normal Speed") { model.speed = 1 }
                    .keyboardShortcut("0", modifiers: [])
            }
            CommandMenu("Chords") {
                Picker("Chord Symbols", selection: Bindable(model).style) {
                    Text("Jazz (full extensions)").tag(ChordStyle.jazz)
                    Text("Simple (triads & sevenths)").tag(ChordStyle.simple)
                }
                .pickerStyle(.inline)
                Button("Toggle Jazz / Simple") { model.style = model.style == .jazz ? .simple : .jazz }
                    .keyboardShortcut("j")
                Picker("Extensions (9, 11, 13)", selection: Bindable(model).extensions) {
                    ForEach(AppModel.Extensions.allCases) { Text($0.title).tag($0) }
                }
                Divider()
                Button("Transpose Up") { model.transpose = min(11, model.transpose + 1) }
                    .keyboardShortcut(.upArrow, modifiers: [.command])
                Button("Transpose Down") { model.transpose = max(-11, model.transpose - 1) }
                    .keyboardShortcut(.downArrow, modifiers: [.command])
                Button("Original Key") { model.transpose = 0 }
                Toggle("Transpose the Audio Too", isOn: Bindable(model).shiftAudio)
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func application(_ application: NSApplication, open urls: [URL]) {
        guard let url = urls.first else { return }
        Task { @MainActor in AppModel.shared.open(url) }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running from `swift run` gives a plain process; make it a regular app.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

extension AppModel {
    static let shared = AppModel()
}
