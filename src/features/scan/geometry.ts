/**
 * From "the value starts at character 214" to "the value is on page 2, this line".
 *
 * The reader sees one string; pages are joined here with a blank line between
 * them, so a heading at the top of a page stays its own line, and each line's
 * absolute offset is recorded. A value on page 3 therefore still resolves to page
 * 3, and the source sheet shows `Page 3`.
 *
 * The page's own text is used exactly as it came, never rebuilt from the boxes.
 * That is deliberate: a PDF text layer reports the runs the author drew, and the
 * order it reports them in is the order the page was written — a row's title, the
 * values under it, the next row — even when each run arrives on its own line.
 * Merging those runs by their coordinates was tried and it is strictly worse: a
 * transcript printed in two columns below a full-width header has no page-wide
 * gutter to split on, so left-hand labels ended up spliced into right-hand course
 * titles. The reader handles the run-per-line shape by joining the lines instead
 * (see `tableLines` in the transcript reader), where the document's own order is
 * still intact.
 */
import type { PaceVisionPage } from '../../../modules/pace-vision';
import type { EvidenceMeta } from '../import/draft';
import type { AcademicRecord } from '../../engine/types';

export interface LineGeometry {
  page: number;
  start: number;
  end: number;
  bbox: { x: number; y: number; width: number; height: number } | null;
  confidence: number | null;
}

/**
 * Every page as lines, and the one string the reader sees.
 *
 * A block is one recognised line; Vision and PDFKit both report one per printed
 * line, so they pair with the page's lines one to one. When they do not (a page
 * that lost its boxes, or one whose text has blank lines), the text is kept and
 * the page contributes no highlight rather than a wrong one.
 */
export function layoutPages(pages: PaceVisionPage[]): { text: string; lines: LineGeometry[] } {
  const chunks: { text: string; page: number; bbox: LineGeometry['bbox']; confidence: number | null }[] = [];
  pages.forEach((page, pageIndex) => {
    const blocks = page.blocks ?? [];
    const lines = page.text.split('\n');
    const paired = blocks.length === lines.filter((line) => line.trim()).length;
    let taken = 0;
    for (const line of lines) {
      if (!line.trim()) {
        chunks.push({ text: line, page: page.page, bbox: null, confidence: null });
        continue;
      }
      const block = paired ? blocks[taken] : undefined;
      taken += 1;
      chunks.push({
        text: line,
        page: page.page,
        bbox: block ? { x: block.x, y: block.y, width: block.width, height: block.height } : null,
        confidence: block ? block.confidence : null,
      });
    }
    // A blank line between pages keeps a heading at the top of one its own line.
    if (pageIndex < pages.length - 1) chunks.push({ text: '', page: page.page, bbox: null, confidence: null });
  });

  const lines: LineGeometry[] = [];
  const texts: string[] = [];
  let offset = 0;
  for (const chunk of chunks) {
    texts.push(chunk.text);
    if (chunk.text.trim()) {
      lines.push({
        page: chunk.page,
        start: offset,
        end: offset + chunk.text.length,
        bbox: chunk.bbox,
        confidence: chunk.confidence,
      });
    }
    offset += chunk.text.length + 1;
  }
  return { text: texts.join('\n'), lines };
}

/** The line a span starts on. Values never span lines in this reader. */
export function locate(start: number | null, geometry: LineGeometry[]): LineGeometry | null {
  if (start == null || !geometry.length) return null;
  for (const line of geometry) {
    if (start >= line.start && start <= line.end) return line;
  }
  return null;
}

export function evidenceMetaFrom(
  record: AcademicRecord,
  geometry: LineGeometry[],
): Record<string, EvidenceMeta> {
  const out: Record<string, EvidenceMeta> = {};
  if (!geometry.length) return out;
  for (const item of record.evidence) {
    const line = locate(item.start, geometry);
    out[item.id] = {
      page: line?.page ?? null,
      bbox: line?.bbox ?? null,
      ocrConfidence: line?.confidence ?? null,
    };
  }
  return out;
}

/** Which pages contain evidence, for the "pages read" line in review. */
export function pagesWithEvidence(meta: Record<string, EvidenceMeta>): number[] {
  const pages = new Set<number>();
  for (const entry of Object.values(meta)) {
    if (entry.page != null) pages.add(entry.page);
  }
  return Array.from(pages).sort((left, right) => left - right);
}
