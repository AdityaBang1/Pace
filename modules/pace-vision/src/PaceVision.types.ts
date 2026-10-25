/**
 * One recognised line.
 *
 * `x`/`y`/`width`/`height` are normalised to the page (0...1, origin top-left),
 * so a highlight scales to any view size without another conversion.
 */
export interface PaceVisionBlock {
  id: string;
  text: string;
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Vision's own per-line confidence (0...1). */
  confidence: number;
}

export interface PaceVisionPage {
  text: string;
  blocks: PaceVisionBlock[];
  /** Zero-based page index. Pages keep their printed position even if one is skipped. */
  page: number;
  /**
   * Where this page's characters came from.
   *
   * `text_layer` is the PDF's own text — the author's bytes, read by PDFKit.
   * `ocr` is a reading of rendered pixels, and the only one that can be wrong.
   */
  method: 'text_layer' | 'ocr';
  imageWidth: number;
  imageHeight: number;
  durationMs: number;
}

/**
 * A file the OCR boundary read: a photograph (one page) or a PDF (one entry per
 * rendered page).
 *
 * `textPages` counts the pages read from a PDF's own text layer and `ocrPages`
 * the pages that had to be recognised, so a caller can say which part of a
 * document is a reading rather than the author's characters.
 *
 * `truncated` is true when the document was longer than the page cap; the app
 * shows that rather than quietly importing the first pages as if they were all.
 */
export interface PaceVisionDocument {
  kind: 'pdf' | 'image';
  pages: PaceVisionPage[];
  /** Pages joined with a blank line, for display only — the app rejoins them itself. */
  text: string;
  pageCount: number;
  textPages: number;
  ocrPages: number;
  truncated: boolean;
  durationMs: number;
}
