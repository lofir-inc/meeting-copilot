// 画像から文字を読む（macOS の Vision）。別プロセスで動かすための小さな道具。
//
// なぜ Swift か（2026-09-16）: pyobjc から Vision を呼ぶと Python ごと落ちた
// （TextRecognition の中で EXC_BREAKPOINT・5 回連続）。落ちても会議に影響しないよう、
// **別のプロセス**に切り出す。Swift なら Vision の想定どおりの呼び方になる。
//
//   swiftc -O ocr.swift -o ocr
//   ./ocr 画像.png            → {"lines":[{"text":"…","x":0.1,"y":0.2,"w":0.3,"h":0.02,"confidence":0.9}]}
//
// 外へは何も送らない（手元で完結）。

import Foundation
import Vision
import AppKit

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    FileHandle.standardError.write("使い方: ocr <画像のパス>\n".data(using: .utf8)!)
    exit(2)
}
let path = arguments[1]
guard let image = NSImage(contentsOfFile: path),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("画像を読めません: \(path)\n".data(using: .utf8)!)
    exit(3)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false        // URL を「直されない」ようにする
request.recognitionLanguages = ["ja-JP", "en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write("読み取りに失敗: \(error)\n".data(using: .utf8)!)
    exit(4)
}

var lines: [[String: Any]] = []
for observation in request.results ?? [] {
    guard let candidate = observation.topCandidates(1).first else { continue }
    let box = observation.boundingBox
    lines.append([
        "text": candidate.string,
        "confidence": Double(candidate.confidence),
        "x": Double(box.origin.x), "y": Double(1 - box.origin.y - box.height),
        "w": Double(box.width), "h": Double(box.height),
    ])
}
let payload: [String: Any] = ["lines": lines]
let data = try JSONSerialization.data(withJSONObject: payload, options: [])
FileHandle.standardOutput.write(data)
