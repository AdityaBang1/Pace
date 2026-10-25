import ExpoModulesCore
import ImageIO
import PDFKit
import UIKit
import Vision

/**
 Apple Vision OCR for Pace, for photographs and for PDFs.

 The app needs two things from a page: its text, and where each line sits on it.
 Text is what PaceDecide reads; geometry is what lets the review screen highlight
 the line a value came from, so both leave this module together.

 A PDF is rendered page by page with PDFKit and read the same way a photograph is.
 That is deliberate: an image-only scan and a digital page then take one path, and
 every page gets boxes. (A PDF's text layer would be exact for a digital page, but
 it carries no geometry, and mixing the two paths would mean two different
 provenance stories in the same review screen.)

 All rectangles are normalised (0...1) with the origin at the *top* left — Vision
 returns them bottom-left, which is a persistent source of upside-down highlights —
 and `imageWidth`/`imageHeight` are the dimensions of the page *as it is seen*: for
 a photograph those are the oriented dimensions, because Vision reports boxes in
 the upright image and a portrait capture is stored as landscape pixels plus EXIF.
 */
public class PaceVisionModule: Module {
  /// A document longer than this is read up to the cap and reported as truncated,
  /// because one page costs roughly a second of on-device recognition.
  private static let maxPages = 25
  private static let targetLongSide: CGFloat = 1600
  private static let maxLongSide: CGFloat = 3000
  /// A 12–48 MP photograph is decoded through a thumbnail: Vision gains no accuracy
  /// from the extra pixels, and the full decode costs seconds and a large transient
  /// buffer on a phone.
  private static let maxImageLongSide = 2400

  public func definition() -> ModuleDefinition {
    Name("PaceVision")

    AsyncFunction("recognizeText") { (uri: String) -> [String: Any] in
      let document = try Self.recognizeDocument(uri: uri)
      guard let pages = document["pages"] as? [[String: Any]], let first = pages.first else {
        throw EmptyDocumentException(uri)
      }
      return first
    }

    /**
     Recognise one page per URI, each tagged with its index. Kept for callers that
     already have images in hand (a multi-photo capture).
     */
    AsyncFunction("recognizePages") { (uris: [String]) -> [[String: Any]] in
      return try uris.enumerated().map { index, uri in
        var page = try Self.recognize(uri: uri)
        page["page"] = index
        return page
      }
    }

    /**
     The entry point the app uses: one file in, every page out. Accepts a PDF or
     an image; the result says which it was, how many pages were read, and whether
     the page cap cut the document short.
     */
    AsyncFunction("recognizeDocument") { (uri: String) -> [String: Any] in
      try Self.recognizeDocument(uri: uri)
    }
  }

  // MARK: - Document

  private static func recognizeDocument(uri: String) throws -> [String: Any] {
    let started = Date()
    let url = fileURL(uri)
    let isPDF = url.pathExtension.lowercased() == "pdf" || hasPDFHeader(url)
    var pages: [[String: Any]] = []
    var truncated = false

    if isPDF {
      guard let pdf = PDFDocument(url: url) else { throw InvalidDocumentException(uri) }
      truncated = pdf.pageCount > maxPages
      let count = min(pdf.pageCount, maxPages)
      for index in 0..<count {
        guard let page = pdf.page(at: index) else { continue }
        guard let rendered = render(page: page) else {
          // A page that renders empty is skipped rather than reported as text-less:
          // an empty page in a PDF is ordinary, and OCR on it would add nothing.
          continue
        }
        pages.append(try recognize(cgImage: rendered.image, orientation: .up, page: index))
      }
      guard !pages.isEmpty else { throw EmptyDocumentException(uri) }
    } else {
      let decoded = try decodeImage(at: url)
      pages.append(try recognize(cgImage: decoded.image, orientation: decoded.orientation, page: 0))
    }

    let text = pages.compactMap { $0["text"] as? String }.joined(separator: "\n\n")
    return [
      "kind": isPDF ? "pdf" : "image",
      "pages": pages,
      "text": text,
      "pageCount": pages.count,
      "truncated": truncated,
      "durationMs": Int(Date().timeIntervalSince(started) * 1000),
    ]
  }

  /// `file://` URIs from the picker, and plain paths from the cache directory.
  private static func fileURL(_ uri: String) -> URL {
    let path = uri.hasPrefix("file://") ? String(uri.dropFirst("file://".count)) : uri
    let decoded = path.removingPercentEncoding ?? path
    return URL(fileURLWithPath: decoded)
  }

  /// Decode a photograph at a readable size, keeping its orientation separate.
  ///
  /// `CreateThumbnailWithTransform` is left **off** on purpose: baking the EXIF
  /// rotation into the bitmap and also passing the orientation to
  /// `VNImageRequestHandler` rotates the page twice, which is how a portrait
  /// photograph ends up sideways to Vision. The pixels stay as stored; the handler
  /// applies orientation exactly once, and the boxes it returns are already in the
  /// upright, normalised space the app expects.
  private static func decodeImage(at url: URL) throws -> (image: CGImage, orientation: CGImagePropertyOrientation) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
      throw InvalidDocumentException(url.path)
    }
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientation = (properties?[kCGImagePropertyOrientation] as? UInt32)
      .flatMap { CGImagePropertyOrientation(rawValue: $0) } ?? .up
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: false,
      kCGImageSourceShouldCacheImmediately: false,
      kCGImageSourceThumbnailMaxPixelSize: maxImageLongSide,
    ]
    if let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) {
      return (image, orientation)
    }
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
      throw InvalidDocumentException(url.path)
    }
    return (image, orientation)
  }

  /// The size Vision sees, not the size stored on disk.
  ///
  /// A portrait capture is landscape pixels plus an EXIF rotation, and Vision's
  /// boxes are normalised against the upright page. Reporting the raw pixel size
  /// would draw a portrait page into a landscape frame and throw every highlight
  /// off, so the two axes are swapped for a quarter-turn orientation.
  private static func orientedSize(
    _ image: CGImage,
    orientation: CGImagePropertyOrientation
  ) -> (width: Int, height: Int) {
    switch orientation {
    case .left, .leftMirrored, .right, .rightMirrored:
      return (image.height, image.width)
    default:
      return (image.width, image.height)
    }
  }

  private static func hasPDFHeader(_ url: URL) -> Bool {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
    defer { try? handle.close() }
    let head = handle.readData(ofLength: 5)
    return head == Data("%PDF-".utf8)
  }

  // MARK: - PDF rendering

  /// A page drawn into a bitmap at a size Vision reads well.
  private static func render(page: PDFPage) -> (image: CGImage, width: Int, height: Int)? {
    let bounds = page.bounds(for: .mediaBox)
    guard bounds.width > 1, bounds.height > 1 else { return nil }
    let longest = max(bounds.width, bounds.height)
    let scale = min(maxLongSide / longest, max(targetLongSide / longest, 1.0))
    let width = max(Int(bounds.width * scale), 1)
    let height = max(Int(bounds.height * scale), 1)
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else { return nil }

    // Paper, then the page drawn in its own coordinate space. PDF origin is
    // bottom-left, so the context is flipped once here rather than upside-down
    // appearing in the photograph.
    context.setFillColor(UIColor.white.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.interpolationQuality = .high
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: scale, y: -scale)
    page.draw(with: .mediaBox, to: context)

    guard let image = context.makeImage() else { return nil }
    return (image, width, height)
  }

  // MARK: - Recognition

  private static func recognize(uri: String) throws -> [String: Any] {
    let decoded = try decodeImage(at: fileURL(uri))
    return try recognize(cgImage: decoded.image, orientation: decoded.orientation, page: 0)
  }

  private static func recognize(
    cgImage: CGImage,
    orientation: CGImagePropertyOrientation,
    page: Int
  ) throws -> [String: Any] {
    let started = Date()
    let request = VNRecognizeTextRequest()
    // `.accurate` is slower but it is the difference between "40%" and "4O%" on a
    // photographed handout, and the reader downstream depends on the digits.
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    if #available(iOS 16.0, *) {
      request.automaticallyDetectsLanguage = true
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
    try handler.perform([request])

    let observations = request.results ?? []
    // Vision's origin is bottom-left, so reading order is descending midY.
    let ordered = observations.sorted { left, right in
      let leftMid = left.boundingBox.midY
      let rightMid = right.boundingBox.midY
      if abs(leftMid - rightMid) > 0.01 { return leftMid > rightMid }
      return left.boundingBox.minX < right.boundingBox.minX
    }

    var blocks: [[String: Any]] = []
    var lines: [String] = []
    for (index, observation) in ordered.enumerated() {
      guard let candidate = observation.topCandidates(1).first else { continue }
      let box = observation.boundingBox
      lines.append(candidate.string)
      blocks.append([
        "id": "p\(page)-line-\(index)",
        "text": candidate.string,
        "page": page,
        "x": Double(box.minX),
        "y": Double(1 - box.maxY),
        "width": Double(box.width),
        "height": Double(box.height),
        "confidence": Double(candidate.confidence),
      ])
    }

    let size = orientedSize(cgImage, orientation: orientation)
    return [
      "text": lines.joined(separator: "\n"),
      "blocks": blocks,
      "page": page,
      "imageWidth": size.width,
      "imageHeight": size.height,
      "durationMs": Int(Date().timeIntervalSince(started) * 1000),
    ]
  }

}

internal final class InvalidDocumentException: GenericException<String> {
  override var reason: String {
    "Pace could not open the document at \(param)."
  }
}

internal final class EmptyDocumentException: GenericException<String> {
  override var reason: String {
    "No page of \(param) could be read."
  }
}
