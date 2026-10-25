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
  imageWidth: number;
  imageHeight: number;
  durationMs: number;
}

/**
 * A file the OCR boundary read: a photograph (one page) or a PDF (one entry per
 * rendered page).
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
  truncated: boolean;
  durationMs: number;
}
