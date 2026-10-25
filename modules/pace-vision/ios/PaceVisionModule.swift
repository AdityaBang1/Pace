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

 A page is read by whichever method is exact for it:

 - A PDF page that carries its own characters (the ordinary case for a syllabus
   exported from a word processor or a course catalog) is read from that text
   layer. Those are the author's bytes, so a course code and a mark cannot be
   misread, and PDFKit is asked for the bounds of each line so provenance still
   points at the printed line.
 - A page with no text layer — a scan, a photograph compiled into a PDF, a page
   whose layer is empty or unusable — is rendered and read by Vision, the same
   way a photograph is read. That path is a *reading* of pixels and is the only
   one that can be wrong.

 Each page reports which method produced it (`method`), because the difference
 matters: text from a text layer is the document, text from OCR is an OCR of it.

 All rectangles are normalised (0...1) with the origin at the *top* left — Vision
 returns them bottom-left, which is a persistent source of upside-down highlights
 — and `imageWidth`/`imageHeight` are the dimensions of the page *as it is seen*:
 for a photograph those are the oriented dimensions, because Vision reports boxes
 in the upright image and a portrait capture is stored as landscape pixels plus EXIF.
 */
public class PaceVisionModule: Module {
  /// A document longer than this is read up to the cap and reported as truncated,
  /// because one page costs roughly a second of on-device recognition.
  private static let maxPages = 25
  /// The long side a page is rendered or lifted to before recognition. Below this
  /// the recogniser loses digits (a 768-pixel page read `86` for `98` in the
  /// measured ladder), and above it the extra pixels buy nothing but time.
  private static let ocrTargetLongSide: CGFloat = 2000
  private static let maxOCRScale: CGFloat = 4
  private static let pdfMaxLongSide: CGFloat = 2600
  /// A 12–48 MP photograph is decoded through a thumbnail: Vision gains no accuracy
  /// from the extra pixels, and the full decode costs seconds and a large transient
  /// buffer on a phone.
  private static let maxImageLongSide = 2400
  /// Below this, a PDF page's text layer is a stamp or a page number, not the page.
  private static let minTextLayerCharacters = 24
  /// A text layer with no line structure is worse than OCR for this reader: every
  /// rule downstream works on lines, and one 2,000-character line yields no rows.
  private static let minTextLayerLines = 2
  private static let singleLineLimit = 200
  /// Past this many lines a text layer keeps its text and drops the per-line
  /// selection work, which is the expensive part of reading a text layer.
  private static let maxTextLayerLines = 400
  /// A page that yields less than this is treated as unread and retried the other
  /// way up, because the usual cause is a page held sideways in the frame.
  private static let minReadableCharacters = 40

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
     an image; the result says which it was, how many pages were read, how many
     came from the page's own text and how many had to be recognised, and whether
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
      // A locked PDF renders as blank pages, which would arrive as "no text" and
      // send the user looking for a better copy of a document that is right here.
      if pdf.isLocked { throw LockedDocumentException(uri) }
      truncated = pdf.pageCount > maxPages
      let count = min(pdf.pageCount, maxPages)
      for index in 0..<count {
        guard let page = pdf.page(at: index) else { continue }
        if let fromText = textLayerPage(page, index: index) {
          pages.append(fromText)
          continue
        }
        guard let rendered = renderPage(page) else {
          // A page that renders empty is skipped rather than reported as text-less:
          // an empty page in a PDF is ordinary, and OCR on it would add nothing.
          continue
        }
        pages.append(try recognize(cgImage: rendered, orientation: .up, page: index, method: "ocr"))
      }
      guard !pages.isEmpty else { throw EmptyDocumentException(uri) }
    } else {
      let decoded = try decodeImage(at: url)
      pages.append(try recognizeUpright(
        cgImage: decoded.image,
        orientation: decoded.orientation,
        page: 0
      ))
    }

    let text = pages.compactMap { $0["text"] as? String }.joined(separator: "\n\n")
    let textPages = pages.filter { ($0["method"] as? String) == "text_layer" }.count
    return [
      "kind": isPDF ? "pdf" : "image",
      "pages": pages,
      "text": text,
      "pageCount": pages.count,
      "textPages": textPages,
      "ocrPages": pages.count - textPages,
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

  // MARK: - The page's own text

  /**
   A page read from its text layer, or `nil` when the page has to be recognised.

   The layer is used only when it is *usable*: long enough to be a page, and
   carrying line structure. A PDF that draws each word at an absolute position
   sometimes yields characters with no spaces or newlines at all; that text is
   exact but the document's shape is gone, and the line-based readers would find
   nothing in it, so such a page is recognised from pixels instead.

   Geometry comes from PDFKit's selection bounds, so a value can still name the
   page and the line it was printed on.
   */
  private static func textLayerPage(_ page: PDFPage, index: Int) -> [String: Any]? {
    guard let raw = page.string else { return nil }
    let pageBounds = page.bounds(for: .mediaBox)
    let sized = pageBounds.width > 1 && pageBounds.height > 1
    // A carriage return would shift every following selection offset, and PDFKit
    // does not emit one; when it appears anyway the page keeps its text and only
    // loses its line boxes.
    var geometry = sized && !raw.contains("\r")

    var lines: [String] = []
    var blocks: [[String: Any]] = []
    var offset = 0
    for rawLine in raw.components(separatedBy: "\n") {
      let length = (rawLine as NSString).length
      defer { offset += length + 1 }
      // Blank lines carry nothing and, in a text layer, are usually an artefact of
      // how the page was written rather than a blank printed line.
      let line = rawLine.trimmingCharacters(in: .whitespaces)
      guard !line.isEmpty else { continue }
      let lineIndex = lines.count
      lines.append(line)
      guard geometry else { continue }
      // A page with hundreds of lines is not worth hundreds of selections: the
      // text is what the readers use and the boxes are a convenience.
      guard lineIndex < maxTextLayerLines else {
        geometry = false
        continue
      }
      guard let selection = page.selection(for: NSRange(location: offset, length: length)) else {
        geometry = false
        continue
      }
      // `PDFSelection.bounds(for:)` returns CGRect, not an optional, so it cannot
      // join the binding above; the empty-box checks below do that job instead.
      let rect = selection.bounds(for: page)
      guard !rect.isNull, rect.width > 0, rect.height > 0 else {
        geometry = false
        continue
      }
      blocks.append(block(line: line, lineIndex: lineIndex, pageIndex: index, rect: rect, pageBounds: pageBounds))
    }

    let text = lines.joined(separator: "\n")
    guard isTextLayerUsable(text, lineCount: lines.count) else { return nil }
    // One block per line, or none: a partial set would put a highlight on the
    // wrong line, which the app cannot detect and the user would believe.
    if !geometry { blocks = [] }
    return [
      "text": text,
      "blocks": blocks,
      "page": index,
      "method": "text_layer",
      "imageWidth": Int(pageBounds.width.rounded()),
      "imageHeight": Int(pageBounds.height.rounded()),
      "durationMs": 0,
    ]
  }

  /**
   Is this layer a page, or the wreck of one?

   The counts are over the visible characters, not the whole string: a text layer
   pads columns with runs of spaces (the fixture transcript is 74 % spaces), and
   measuring against every character would reject every real table. A layer that
   is a stamp, an empty page, a wall of glyph ids, or one undifferentiated blob of
   characters is refused, and the page is recognised from pixels instead.
   */
  private static func isTextLayerUsable(_ text: String, lineCount: Int) -> Bool {
    if text.contains("(cid:") || text.contains("\u{FFFD}") { return false }
    var visible = 0
    var alphanumeric = 0
    var control = 0
    for scalar in text.unicodeScalars {
      if scalar.value < 32 && scalar != "\n" && scalar != "\t" {
        control += 1
        continue
      }
      guard !CharacterSet.whitespacesAndNewlines.contains(scalar) else { continue }
      visible += 1
      if CharacterSet.alphanumerics.contains(scalar) { alphanumeric += 1 }
    }
    guard visible >= minTextLayerCharacters else { return false }
    // Exact characters, but no layout to read: a photograph of the same page gives
    // the readers the lines they need, so recognise it rather than parse a blob.
    if lineCount < minTextLayerLines && visible >= singleLineLimit { return false }
    if Double(control) / Double(visible) > 0.02 { return false }
    return Double(alphanumeric) / Double(visible) >= 0.45
  }

  /**
   One line of a text layer, in the same normalised shape Vision's blocks use.

   `confidence` is 1 because these are the page's own characters: there is nothing
   to be uncertain about, and the app's bands then read as they should.
   */
  private static func block(
    line: String,
    lineIndex: Int,
    pageIndex: Int,
    rect: CGRect,
    pageBounds: CGRect
  ) -> [String: Any] {
    let width = pageBounds.width
    let height = pageBounds.height
    return [
      "id": "p\(pageIndex)-line-\(lineIndex)",
      "text": line,
      "page": pageIndex,
      "x": Double((rect.minX - pageBounds.minX) / width),
      "y": Double(1 - (rect.maxY - pageBounds.minY) / height),
      "width": Double(rect.width / width),
      "height": Double(rect.height / height),
      "confidence": 1.0,
    ]
  }

  // MARK: - PDF rendering

  /**
   A page drawn into a bitmap at a size Vision reads well.

   Drawn through `UIGraphicsImageRenderer`, whose context is the same top-down
   space a UIKit view uses, with the single origin flip PDFKit needs. A bitmap
   context built by hand is bottom-up, so the flip that is correct here becomes a
   double flip there and the page is recognised upside down — which reads as "the
   PDF has no text at all". `format.scale = 1` keeps the requested pixel size
   exact instead of multiplying it by the screen scale.
   */
  private static func renderPage(_ page: PDFPage) -> CGImage? {
    let bounds = page.bounds(for: .mediaBox)
    guard bounds.width > 1, bounds.height > 1 else { return nil }
    let longest = max(bounds.width, bounds.height)
    let scale = min(pdfMaxLongSide / longest, max(ocrTargetLongSide / longest, 1))
    let size = CGSize(
      width: max((bounds.width * scale).rounded(), 1),
      height: max((bounds.height * scale).rounded(), 1)
    )

    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    format.opaque = true
    let renderer = UIGraphicsImageRenderer(size: size, format: format)
    let image = renderer.image { context in
      let cg = context.cgContext
      // Paper first: a transparent page would reach the recogniser as black.
      UIColor.white.setFill()
      cg.fill(CGRect(origin: .zero, size: size))
      cg.interpolationQuality = .high
      // PDF space is bottom-up and its origin is the media box's own corner, which
      // is not always (0, 0): the flip and the offset are both needed.
      cg.translateBy(x: 0, y: size.height)
      cg.scaleBy(x: size.width / bounds.width, y: -size.height / bounds.height)
      cg.translateBy(x: -bounds.minX, y: -bounds.minY)
      page.draw(with: .mediaBox, to: cg)
    }
    return image.cgImage
  }

  // MARK: - Recognition

  private static func recognize(uri: String) throws -> [String: Any] {
    let decoded = try decodeImage(at: fileURL(uri))
    return try recognizeUpright(cgImage: decoded.image, orientation: decoded.orientation, page: 0)
  }

  /**
   Vision's languages, narrowed to the ones it actually ships for this revision.

   Without this the request recognises whatever its default model covers, which on
   a device set to another language can mean no model for the document at all. The
   device's own languages come first, English is always present because the reader
   is built for English academic documents, and the list is capped so recognition
   does not try every model it owns.
   */
  private static func recognitionLanguages() -> [String] {
    let revision = VNRecognizeTextRequest.currentRevision
    let supported = (try? VNRecognizeTextRequest.supportedRecognitionLanguages(
      for: .accurate,
      revision: revision
    )) ?? []
    guard !supported.isEmpty else { return ["en-US"] }
    var wanted: [String] = []
    for preferred in Locale.preferredLanguages {
      if let exact = supported.first(where: { $0.caseInsensitiveCompare(preferred) == .orderedSame }) {
        wanted.append(exact)
        continue
      }
      let language = String(preferred.prefix(2)).lowercased()
      if let match = supported.first(where: { $0.lowercased().hasPrefix(language + "-") }) {
        wanted.append(match)
      }
    }
    if !wanted.contains(where: { $0.lowercased().hasPrefix("en") }),
       let english = supported.first(where: { $0.lowercased().hasPrefix("en") }) {
      wanted.append(english)
    }
    var seen = Set<String>()
    return Array(wanted.filter { seen.insert($0).inserted }.prefix(4))
  }

  /// Words the recogniser would otherwise "correct" into something else. A course
  /// code and a grading label are the values the app must not invent.
  private static let vocabulary = [
    "prerequisite", "prerequisites", "corequisite", "corequisites", "syllabus",
    "grading", "grade", "grades", "weight", "weights", "credit", "credits",
    "units", "unit", "section", "sections", "CRN", "instructor", "office hours",
    "attendance", "attendance policy", "assignment", "assignments", "quiz",
    "quizzes", "midterm", "final", "semester", "term", "course", "courses",
    "catalog", "schedule", "room", "campus", "lecture", "laboratory", "lab",
    "tutorial", "seminar", "department", "program", "requirement",
  ]

  /// Lift a page whose text is small relative to the frame to the size the
  /// recogniser reads best. A screenshot of a syllabus is 1170 pixels wide and a
  /// cropped photo can be smaller still; both read reliably once lifted.
  private static func upscaleIfSmall(_ image: CGImage) -> CGImage {
    let longest = max(image.width, image.height)
    guard longest > 0, CGFloat(longest) < ocrTargetLongSide else { return image }
    let factor = min(maxOCRScale, ocrTargetLongSide / CGFloat(longest))
    guard factor > 1.05 else { return image }
    let width = Int((CGFloat(image.width) * factor).rounded())
    let height = Int((CGFloat(image.height) * factor).rounded())
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else { return image }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage() ?? image
  }

  /**
   A photograph, retried the other way up when the first reading is empty.

   EXIF covers the phone's own turn, not the page's: a landscape sheet photographed
   with the phone upright arrives as pixels on their side and reads as nothing. Only
   an unreadable page pays for the extra passes; a page that read normally is
   returned from the first one.
   */
  private static func recognizeUpright(
    cgImage: CGImage,
    orientation: CGImagePropertyOrientation,
    page: Int
  ) throws -> [String: Any] {
    var best = try recognize(cgImage: cgImage, orientation: orientation, page: page, method: "ocr")
    guard characterCount(best) < minReadableCharacters else { return best }
    for alternative in [CGImagePropertyOrientation.right, .down, .left] {
      let attempt = try recognize(cgImage: cgImage, orientation: alternative, page: page, method: "ocr")
      if characterCount(attempt) > characterCount(best) { best = attempt }
      if characterCount(best) >= minReadableCharacters { break }
    }
    return best
  }

  private static func characterCount(_ page: [String: Any]) -> Int {
    (page["text"] as? String)?.count ?? 0
  }

  private static func recognize(
    cgImage: CGImage,
    orientation: CGImagePropertyOrientation,
    page: Int,
    method: String
  ) throws -> [String: Any] {
    let started = Date()
    let prepared = upscaleIfSmall(cgImage)
    let request = VNRecognizeTextRequest()
    // `.accurate` is slower but it is the difference between "40%" and "4O%" on a
    // photographed handout, and the reader downstream depends on the digits.
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.customWords = vocabulary
    request.recognitionLanguages = recognitionLanguages()

    let handler = VNImageRequestHandler(cgImage: prepared, orientation: orientation, options: [:])
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

    let size = orientedSize(prepared, orientation: orientation)
    return [
      "text": lines.joined(separator: "\n"),
      "blocks": blocks,
      "page": page,
      "method": method,
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

internal final class LockedDocumentException: GenericException<String> {
  override var reason: String {
    "\(param) is password-protected, so its pages cannot be read."
  }
}
