import CryptoKit
import Foundation

/// Fetches the audio of an online song or video (beta).
///
/// * Direct links to audio files are downloaded with URLSession.
/// * Everything else (YouTube, SoundCloud, Bandcamp, Vimeo, web pages with an
///   <audio> or <video> element, ...) goes through the open-source yt-dlp
///   tool: an existing installation (Homebrew) is used if present, otherwise
///   the official macOS build is downloaded once, checked against the
///   published SHA-256 sums, and kept in Application Support.
public final class LinkDownloader: @unchecked Sendable {
    public struct Result: Sendable {
        public var file: URL
        public var title: String
    }

    public enum Progress: Sendable {
        case installingTool(Double?)
        case resolving
        case downloading(Double?)
        case converting
    }

    public static let audioExtensions: Set<String> = ["mp3", "m4a", "aac", "wav", "aif", "aiff", "flac", "caf", "mp4", "m4b", "alac"]
    static let ytDLPRelease = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/"

    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    public init() {}

    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Chordify", isDirectory: true)
    }

    static var downloadDirectory: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Chordify/Downloads", isDirectory: true)
    }

    /// Accepts "youtube.com/watch?v=..." without a scheme too.
    public static func normalise(_ text: String) -> URL? {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !t.contains(where: \.isWhitespace) else { return nil }
        if !t.contains("://") { t = "https://" + t }
        guard let url = URL(string: t), let scheme = url.scheme?.lowercased(),
              scheme == "https" || scheme == "http", let host = url.host,
              host.contains(".") || host == "localhost" else { return nil }
        return url
    }

    public func cancel() {
        lock.lock()
        cancelled = true
        process?.terminate()
        lock.unlock()
    }

    // MARK: - yt-dlp

    /// An installed yt-dlp: our own copy first, then Homebrew / MacPorts / PATH.
    public static func installedTool() -> URL? {
        var paths = [supportDirectory.appendingPathComponent("bin/yt-dlp").path,
                     "/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp", "/opt/local/bin/yt-dlp"]
        for dir in (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":") {
            paths.append(String(dir) + "/yt-dlp")
        }
        return paths.first { FileManager.default.isExecutableFile(atPath: $0) }.map { URL(fileURLWithPath: $0) }
    }

    static func ffmpegDirectory() -> String? {
        ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"].first {
            FileManager.default.isExecutableFile(atPath: $0 + "/ffmpeg")
        }
    }

    /// Downloads the official yt-dlp macOS build and verifies its SHA-256.
    public func installTool(progress: @escaping @Sendable (Double?) -> Void) async throws -> URL {
        let bin = Self.supportDirectory.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        progress(nil)
        let (sums, _) = try await URLSession.shared.data(from: URL(string: Self.ytDLPRelease + "SHA2-256SUMS")!)
        guard let line = String(decoding: sums, as: UTF8.self).split(separator: "\n")
            .first(where: { $0.hasSuffix(" yt-dlp_macos") }),
            let expected = line.split(separator: " ").first.map(String.init)
        else { throw ChordifyError.audio("Could not read the yt-dlp release checksums.") }

        let (tmp, response) = try await URLSession.shared.download(from: URL(string: Self.ytDLPRelease + "yt-dlp_macos")!)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw ChordifyError.audio("Could not download yt-dlp.") }
        let data = try Data(contentsOf: tmp)
        let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actual == expected.lowercased() else {
            throw ChordifyError.audio("The downloaded yt-dlp did not match its published checksum, so it was not installed.")
        }
        let dest = bin.appendingPathComponent("yt-dlp")
        try? FileManager.default.removeItem(at: dest)
        try data.write(to: dest)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dest.path)
        progress(1)
        return dest
    }

    // MARK: - fetching

    /// Downloads the audio of `url` (cached by URL).
    /// `cookiesFromBrowser` ("safari", "chrome", "firefox", ...) lets yt-dlp use
    /// that browser's sign-in, for sites that ask to confirm you are not a bot.
    public func fetch(_ url: URL, installIfNeeded: Bool, cookiesFromBrowser: String? = nil,
                      progress: @escaping @Sendable (Progress) -> Void) async throws -> Result {
        lock.lock()
        cancelled = false
        lock.unlock()
        let dir = Self.downloadDirectory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let key = SHA256.hash(data: Data(url.absoluteString.utf8)).map { String(format: "%02x", $0) }.joined().prefix(24)
        let meta = dir.appendingPathComponent("\(key).json")
        if let data = try? Data(contentsOf: meta),
           let info = try? JSONDecoder().decode([String: String].self, from: data),
           let path = info["file"], FileManager.default.fileExists(atPath: path) {
            return Result(file: URL(fileURLWithPath: path), title: info["title"] ?? url.lastPathComponent)
        }

        let result: Result
        if Self.audioExtensions.contains(url.pathExtension.lowercased()) {
            result = try await directDownload(url, to: dir.appendingPathComponent("\(key).\(url.pathExtension.lowercased())"), progress: progress)
        } else {
            var tool = Self.installedTool()
            if tool == nil {
                guard installIfNeeded else { throw ChordifyError.audio("yt-dlp is not installed.") }
                tool = try await installTool { progress(.installingTool($0)) }
            }
            result = try await runYTDLP(tool!, url: url, output: dir.appendingPathComponent("\(key).%(ext)s"),
                                        cookies: cookiesFromBrowser, progress: progress)
        }
        let info = ["file": result.file.path, "title": result.title, "url": url.absoluteString]
        try? JSONEncoder().encode(info).write(to: meta)
        return result
    }

    private func directDownload(_ url: URL, to dest: URL, progress: @escaping @Sendable (Progress) -> Void) async throws -> Result {
        progress(.downloading(nil))
        let (tmp, response) = try await URLSession.shared.download(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ChordifyError.audio("The server answered with an error for \(url.absoluteString).")
        }
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
        let title = url.deletingPathExtension().lastPathComponent.removingPercentEncoding ?? url.lastPathComponent
        return Result(file: dest, title: title)
    }

    /// Lines yt-dlp prints for us (see the --print / --progress-template options).
    struct ToolOutput {
        var file: String?
        var title: String?
        var percent: Double?

        mutating func parse(_ line: String) {
            if line.hasPrefix("CHORDIFY_FILE ") { file = String(line.dropFirst(14)) }
            else if line.hasPrefix("CHORDIFY_TITLE ") { title = String(line.dropFirst(15)) }
            else if line.hasPrefix("CHORDIFY_PROGRESS ") {
                let v = line.dropFirst(18).trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "%", with: "")
                percent = Double(v).map { $0 / 100 }
            }
        }
    }

    static func arguments(url: URL, output: URL, ffmpeg: String?, cookies: String? = nil) -> [String] {
        var args = ["--no-playlist", "--no-simulate", "--newline", "--no-colors", "--no-warnings",
                    "-f", "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio[acodec^=mp4a]/best[ext=mp4]/bestaudio/best",
                    "--progress", "--progress-template", "download:CHORDIFY_PROGRESS %(progress._percent_str)s",
                    "--print", "after_move:CHORDIFY_FILE %(filepath)s",
                    "--print", "after_move:CHORDIFY_TITLE %(title)s",
                    "-o", output.path]
        if let ffmpeg {
            // with ffmpeg, any source format (Opus, WebM, ...) becomes AAC that Core Audio can read
            args += ["-x", "--audio-format", "m4a", "--ffmpeg-location", ffmpeg]
        }
        if let cookies { args += ["--cookies-from-browser", cookies] }
        return args + ["--", url.absoluteString]
    }

    private func runYTDLP(_ tool: URL, url: URL, output: URL, cookies: String?,
                          progress: @escaping @Sendable (Progress) -> Void) async throws -> Result {
        progress(.resolving)
        let p = Process()
        p.executableURL = tool
        p.arguments = Self.arguments(url: url, output: output, ffmpeg: Self.ffmpegDirectory(), cookies: cookies)
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:" + (env["PATH"] ?? "/usr/bin:/bin")
        p.environment = env
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err

        let state = OutputState()
        out.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            guard !data.isEmpty else { return }
            for line in state.append(data) {
                let parsed = state.parse(line)
                if let pct = parsed.percent { progress(pct >= 0.999 ? .converting : .downloading(pct)) }
            }
        }
        let errLines = OutputState()
        err.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            guard !data.isEmpty else { return }
            state.appendError(data)
            // progress lines may arrive on stderr, depending on the yt-dlp version
            for line in errLines.append(data) where line.hasPrefix("CHORDIFY_") {
                let parsed = state.parse(line)
                if let pct = parsed.percent { progress(pct >= 0.999 ? .converting : .downloading(pct)) }
            }
        }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                p.terminationHandler = { _ in c.resume() }
                lock.lock()
                if cancelled {
                    lock.unlock()
                    c.resume(throwing: CancellationError())
                    return
                }
                process = p
                lock.unlock()
                do { try p.run() } catch { c.resume(throwing: error) }
            }
        } onCancel: { [weak self] in
            self?.cancel()
        }
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        for line in state.append(out.fileHandleForReading.readDataToEndOfFile()) + state.flush() { _ = state.parse(line) }
        lock.lock()
        process = nil
        let wasCancelled = cancelled
        lock.unlock()
        if wasCancelled { throw CancellationError() }

        let final = state.result
        guard p.terminationStatus == 0, let path = final.file, FileManager.default.fileExists(atPath: path) else {
            let message = state.errorText.split(separator: "\n").last(where: { $0.contains("ERROR") }).map(String.init)
                ?? "yt-dlp could not fetch this link."
            let hint = message.contains("not a bot") || message.contains("Sign in")
                ? "\n\nThe site wants a signed-in browser. Choose your browser under \u{201C}Use sign-in from\u{201D} and try again."
                : "\n\nIf YouTube links fail, `brew install yt-dlp` gives a version with everything YouTube needs."
            throw ChordifyError.audio(message.replacingOccurrences(of: "ERROR: ", with: "") + hint)
        }
        let file = URL(fileURLWithPath: path)
        let readable = ["m4a", "mp3", "mp4", "aac", "wav", "aiff", "flac", "caf", "m4b"]
        guard readable.contains(file.pathExtension.lowercased()) else {
            throw ChordifyError.audio("This site only offers .\(file.pathExtension) audio, which macOS cannot decode. " +
                                      "Install ffmpeg (`brew install ffmpeg`) and try again.")
        }
        return Result(file: file, title: final.title ?? url.absoluteString)
    }
}

/// Collects yt-dlp output lines from the pipe callbacks.
private final class OutputState: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()
    private var errors = Data()
    private var output = LinkDownloader.ToolOutput()

    func append(_ data: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(data)
        var lines = [String]()
        while let nl = buffer.firstIndex(of: 0x0A) {
            lines.append(String(decoding: buffer[buffer.startIndex..<nl], as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines))
            buffer.removeSubrange(buffer.startIndex...nl)
        }
        return lines
    }

    func flush() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let rest = String(decoding: buffer, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        buffer.removeAll()
        return rest.isEmpty ? [] : [rest]
    }

    func parse(_ line: String) -> LinkDownloader.ToolOutput {
        lock.lock()
        defer { lock.unlock() }
        output.percent = nil
        output.parse(line)
        return output
    }

    func appendError(_ data: Data) {
        lock.lock()
        errors.append(data)
        lock.unlock()
    }

    var errorText: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: errors, as: UTF8.self)
    }

    var result: LinkDownloader.ToolOutput {
        lock.lock()
        defer { lock.unlock() }
        return output
    }
}
