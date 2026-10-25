/**
 * From "the value starts at character 214" to "the value is on page 2, this line".
 *
 * The reader sees one string, so pages are joined here and their start offsets
 * remembered; Apple Vision returns one observation per line with a normalised box
 * per page. This file rebuilds the line offsets of the joined string and pairs
 * them with the boxes, which is what lets a value be highlighted on the page it
 * came from.
 *
 * If a page's block count disagrees with its line count, that page contributes no
 * geometry: a missing highlight is better than one pointing at the wrong line.
 */
import type { PaceVisionPage } from '../../../modules/pace-vision';
import type { EvidenceMeta } from '../import/draft';
import type { AcademicRecord } from '../../engine/types';

export interface LineGeometry {
  page: number;
  start: number;
  end: number;
  bbox: { x: number; y: number; width: number; height: number };
  confidence: number;
}

/** Pages joined with a blank line: a heading at the top of a page stays its own line. */
export const PAGE_SEPARATOR = '\n\n';

export function joinPages(pages: { page: number; text: string }[]): { text: string; starts: number[] } {
  const starts: number[] = [];
  let offset = 0;
  pages.forEach((page, index) => {
    starts.push(offset);
    offset += page.text.length + (index < pages.length - 1 ? PAGE_SEPARATOR.length : 0);
  });
  return { text: pages.map((page) => page.text).join(PAGE_SEPARATOR), starts };
}

export function lineGeometry(pages: PaceVisionPage[], starts: number[]): LineGeometry[] {
  const out: LineGeometry[] = [];
  pages.forEach((page, pageIndex) => {
    const pageStart = starts[pageIndex];
    if (pageStart == null) return;
    const lines = page.text.split('\n');
    if (!page.blocks.length || page.blocks.length !== lines.length) return;
    let local = 0;
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const block = page.blocks[index];
      out.push({
        page: block.page,
        start: pageStart + local,
        end: pageStart + local + line.length,
        bbox: { x: block.x, y: block.y, width: block.width, height: block.height },
        confidence: block.confidence,
      });
      local += line.length + 1;
    }
  });
  return out;
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
