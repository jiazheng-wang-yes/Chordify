import Foundation
import OnnxRuntimeBindings

public enum ChordifyError: LocalizedError {
    case modelsNotFound
    case audio(String)
    case inference(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotFound:
            return "The model files were not found. Rebuild the app with scripts/build_app.sh."
        case .audio(let m): return m
        case .inference(let m): return "Analysis failed: \(m)"
        }
    }
}

/// Thin wrapper around an ONNX Runtime session with float32 tensors.
final class OnnxModel {
    private static let env: ORTEnv? = try? ORTEnv(loggingLevel: .warning)
    private let session: ORTSession

    init(url: URL) throws {
        guard let env = Self.env else { throw ChordifyError.inference("could not start ONNX Runtime") }
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount))))
        session = try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
    }

    struct Tensor {
        var data: [Float]
        var shape: [Int]
    }

    func run(_ inputs: [String: Tensor], outputs: [String]) throws -> [String: Tensor] {
        var values = [String: ORTValue]()
        for (name, t) in inputs {
            let bytes = NSMutableData(length: t.data.count * MemoryLayout<Float>.stride)!
            t.data.withUnsafeBytes { src in
                bytes.mutableBytes.copyMemory(from: src.baseAddress!, byteCount: src.count)
            }
            values[name] = try ORTValue(tensorData: bytes, elementType: .float,
                                        shape: t.shape.map { NSNumber(value: $0) })
        }
        let result = try session.run(withInputs: values, outputNames: Set(outputs), runOptions: nil)
        var out = [String: Tensor]()
        for name in outputs {
            guard let v = result[name] else { throw ChordifyError.inference("missing output \(name)") }
            let shape = try v.tensorTypeAndShapeInfo().shape.map { $0.intValue }
            let data = try v.tensorData() as Data
            let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            out[name] = Tensor(data: floats, shape: shape)
        }
        return out
    }
}

/// Locates the bundled models (inside Chordify.app, or the repository's
/// Models/ folder when running from a checkout).
public enum ModelLocator {
    public static let required = ["chordnet.onnx", "beat_mel.onnx", "beat_this.onnx", "chord_vocab.json"]

    public static func directory() throws -> URL {
        var candidates = [URL]()
        if let env = ProcessInfo.processInfo.environment["CHORDIFY_MODELS"] {
            candidates.append(URL(fileURLWithPath: env))
        }
        if let res = Bundle.main.resourceURL {
            candidates.append(res.appendingPathComponent("Models"))
        }
        // Walk up from the executable and from the current directory (swift run / swift test).
        for start in [Bundle.main.executableURL, URL(fileURLWithPath: FileManager.default.currentDirectoryPath)] {
            var dir = start?.deletingLastPathComponent()
            for _ in 0..<8 {
                guard let d = dir else { break }
                candidates.append(d.appendingPathComponent("Models"))
                dir = d.deletingLastPathComponent()
            }
        }
        for c in candidates where required.allSatisfy({
            FileManager.default.fileExists(atPath: c.appendingPathComponent($0).path)
        }) {
            return c
        }
        throw ChordifyError.modelsNotFound
    }
}
