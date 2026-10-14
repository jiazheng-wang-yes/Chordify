// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Chordify",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Chordify", targets: ["Chordify"]),
    ],
    dependencies: [
        // ONNX Runtime 1.24 (static library, CPU + CoreML execution providers)
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            revision: "b7fb7f7dea8a2469e6335d95a61b8f36d0dc83b2"
        ),
    ],
    targets: [
        .target(
            name: "ChordifyCore",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ]
        ),
        .executableTarget(
            name: "Chordify",
            dependencies: ["ChordifyCore"]
        ),
        .testTarget(
            name: "ChordifyCoreTests",
            dependencies: ["ChordifyCore"],
            exclude: ["Fixtures"]
        ),
    ]
)
