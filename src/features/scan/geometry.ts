/**
 * From "the value starts at character 214" to "the value is on page 2, this line".
 *
 * The reader sees one string, so pages are laid out into lines here and their
 * start offsets remembered. Apple Vision returns one observation per line with a
 * normalised box; PDFKit returns its text layer with a box per *text run*, and a
 * PDF that draws each table cell at an absolute position hands over one run per
 * cell. Reading that as one line per cell is what turned a transcript into a
 * column of orphaned words: `Course`, `Mark`, `Weight`, `Credit`, then eight
 * course codes each on their own line. So the text the readers get is rebuilt
 * from the boxes: runs that share a baseline become the printed line, spaced by
 * the gap between them.
 *
 * A page is laid out as a table only when its boxes form a column grid. A page of
 * prose in two columns is not a table, and its columns stay separate lines
 * instead of being spliced into each other.
 */
import type { PaceVisionBlock, PaceVisionPage } from '../../../modules/pace-vision';
import type { EvidenceMeta } from '../import/draft';
import type { AcademicRecord } from '../../engine/types';

export interface LineGeometry {
  page: number;
  start: number;
  end: number;
  bbox: { x: number; y: number; width: number; height: number } | null;
  confidence: number | null;
}

/** Blocks whose x positions agree within this page fraction share a column. */
const COLUMN_TOLERANCE = 0.012;
/** A column printed on at least this many lines is structure, not coincidence. */
const COLUMN_REPEATS = 3;
/** Two boxes on one baseline further apart than this are separate columns of text. */
const COLUMN_GAP = 0.04;
/** Beyond this, a gap is page furniture, and forty spaces is already unmistakable. */
const MAX_GAP_SPACES = 40;
/** An average table cell is short; an average column of prose is not. */
const SHORT_CELL = 16;

type Block = PaceVisionBlock;

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function sameBaseline(left: Block, right: Block): boolean {
  const top = Math.max(left.y, right.y);
  const bottom = Math.min(left.y + left.height, right.y + right.height);
  const overlap = bottom - top;
  const shorter = Math.min(left.height, right.height);
  if (shorter > 0) return overlap > shorter * 0.5;
  return Math.abs(left.y - right.y) <= 0.004;
}

/** The x positions that recur down the page, with the blocks anchored at each. */
function columnsOf(blocks: Block[]): Block[][] {
  const sorted = [...blocks].sort((left, right) => left.x - right.x);
  const anchors: number[] = [];
  const columns: Block[][] = [];
  for (const block of sorted) {
    const anchor = anchors[anchors.length - 1];
    if (anchor !== undefined && block.x - anchor <= COLUMN_TOLERANCE) {
      columns[columns.length - 1].push(block);
      continue;
    }
    anchors.push(block.x);
    columns.push([block]);
  }
  return columns.filter((column) => column.length >= COLUMN_REPEATS);
}

/**
 * Is this page a table? Three or more columns that each repeat, and at least two
 * thirds of them made of short cells. A transcript, a grade report and a schedule
 * all qualify; a page of prose set in two columns does not.
 */
function isTablePage(blocks: Block[]): boolean {
  const columns = columnsOf(blocks);
  if (columns.length < 3) return false;
  const short = columns.filter(
    (column) => median(column.map((block) => block.text.trim().length)) <= SHORT_CELL,
  ).length;
  return short * 3 >= columns.length * 2;
}

/** The gap between two boxes, in characters, as the page would have printed it. */
function gapSpaces(left: Block, right: Block): number {
  const gap = right.x - (left.x + left.width);
  if (!(gap > 0)) return 1;
  const perCharacter = left.width / Math.max(left.text.length, 1);
  if (!(perCharacter > 0)) return 1;
  return Math.min(Math.max(Math.round(gap / perCharacter), 1), MAX_GAP_SPACES);
}

function joined(segment: Block[]): string {
  let text = segment[0].text.trim();
  for (let index = 1; index < segment.length; index += 1) {
    text += ' '.repeat(gapSpaces(segment[index - 1], segment[index])) + segment[index].text.trim();
  }
  return text;
}

function bboxOf(segment: Block[]): LineGeometry['bbox'] {
  if (!segment.length) return null;
  const left = Math.min(...segment.map((block) => block.x));
  const top = Math.min(...segment.map((block) => block.y));
  const right = Math.max(...segment.map((block) => block.x + block.width));
  const bottom = Math.max(...segment.map((block) => block.y + block.height));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

interface LaidLine {
  text: string;
  bbox: LineGeometry['bbox'];
  confidence: number | null;
}

/** The words a printed table header is made of. Page structure, not a field rule. */
const HEADER_WORDS = new Set([
  'DATE', 'GL', 'TITLE', 'COURSE', 'COURSES', 'MARK', 'GRADE', 'WEIGHT',
  'CREDIT', 'CREDITS', 'UNITS', 'TERM', 'SCORE', 'SUBJECT', 'PERIOD', 'HOURS',
]);

/** A run of boxes that reads as one printed table header. */
function headerRun(blocks: Block[]): boolean {
  if (blocks.length < 3) return false;
  const words = blocks.map((block) => block.text.trim().toUpperCase());
  if (!words.every((word) => HEADER_WORDS.has(word))) return false;
  return words.some((word) => word === 'MARK' || word === 'GRADE' || word === 'CREDIT');
}

/**
 * Where a page prints two tables beside each other.
 *
 * The evidence is a header line carrying two headers: a term-grade page prints
 * `Course Mark Weight Credit` above each of its two columns, and an academic
 * history prints `DATE GL TITLE MARK CREDIT` twice. That line also marks the
 * gutter, which is the one place on the page where two tables end and begin, so
 * the page can be read one column at a time instead of two rows at a time.
 */
function pageGutter(rows: Block[][]): number | null {
  const found: number[] = [];
  for (const blocksOfRow of rows) {
    const row = [...blocksOfRow].sort((left, right) => left.x - right.x);
    let widest = -1;
    let gutter = 0;
    for (let split = 1; split < row.length; split += 1) {
      if (!headerRun(row.slice(0, split)) || !headerRun(row.slice(split))) continue;
      const left = row[split - 1];
      const gap = row[split].x - (left.x + left.width);
      // Of the two headers, the gutter is the widest gap between them: a single
      // `Course Mark Weight Credit` header also splits where `Weight` ends.
      if (gap > widest) {
        widest = gap;
        gutter = (left.x + left.width + row[split].x) / 2;
      }
    }
    if (widest >= 0) found.push(gutter);
  }
  if (!found.length) return null;
  const middle = median(found);
  // Two headers that disagree about where the gutter is were two different rows,
  // not one page layout; guessing there would put a whole column on the wrong side.
  if (found.some((value) => Math.abs(value - middle) > 0.02)) return null;
  return middle;
}

/** Boxes grouped into the printed line they share. */
function rowsOf(blocks: Block[]): Block[][] {
  const rows: Block[][] = [];
  for (const block of [...blocks].sort((left, right) => left.y - right.y || left.x - right.x)) {
    const row = rows.find((candidate) => sameBaseline(candidate[0], block));
    if (row) row.push(block);
    else rows.push([block]);
  }
  return rows;
}

function lineOf(segment: Block[]): LaidLine {
  return {
    text: joined(segment),
    bbox: bboxOf(segment),
    confidence: Math.min(...segment.map((block) => block.confidence)),
  };
}

/** The printed lines of a run of boxes, one table row at a time. */
function linesOfBlocks(blocks: Block[], table: boolean): LaidLine[] {
  const out: LaidLine[] = [];
  for (const blocksOfRow of rowsOf(blocks)) {
    const row = [...blocksOfRow].sort((left, right) => left.x - right.x);
    let segment: Block[] = [];
    for (const block of row) {
      const previous = segment[segment.length - 1];
      const gap = previous ? block.x - (previous.x + previous.width) : 0;
      // In a table every box on the baseline belongs to the row; elsewhere only a
      // gap a word space could explain is glued back together.
      if (previous && (table || gap <= COLUMN_GAP)) {
        segment.push(block);
      } else {
        if (segment.length) out.push(lineOf(segment));
        segment = [block];
      }
    }
    if (segment.length) out.push(lineOf(segment));
  }
  return out;
}

/** The page's own lines, each paired with the box of the line it is. */
function rawLines(page: PaceVisionPage): LaidLine[] {
  const blocks = page.blocks ?? [];
  const lines = page.text.split('\n');
  // One block per printed line, or none: a partial pairing would put a highlight
  // on the wrong line, which the reader cannot detect and the user would believe.
  const boxes = blocks.length === lines.filter((line) => line.trim()).length ? blocks : [];
  let taken = 0;
  return lines.map((line) => {
    if (!line.trim()) {
      // The blank lines separate paragraphs, and the section reader uses them.
      return { text: line, bbox: null, confidence: null };
    }
    const block = boxes[taken];
    taken += 1;
    return {
      text: line,
      bbox: block ? { x: block.x, y: block.y, width: block.width, height: block.height } : null,
      confidence: block ? block.confidence : null,
    };
  });
}

/**
 * Did this page arrive as one text run per cell?
 *
 * PDFKit reports a text layer per run, so a table drawn cell by cell arrives as
 * `Course`, `Mark`, then a code and a title, one per line. That is the one shape
 * the layout pass exists to repair. A page whose lines are sentences is left
 * exactly as it came: its lines are already its printed lines, and re-deriving
 * them from boxes is only a way to make them worse.
 */
function isFlatPage(page: PaceVisionPage, blocks: Block[]): boolean {
  if (!blocks.length) return false;
  if (blocks.some((block) => block.text.includes('\n'))) return false;
  const lines = (page.text.trim() ? page.text : blocks.map((block) => block.text).join('\n'))
    .split('\n')
    .filter((line) => line.trim());
  if (lines.length < 4 || blocks.length !== lines.length) return false;
  const cells = lines.filter((line) => {
    const value = line.trim();
    return value.split(/\s+/).length <= 3 && value.length <= 22;
  }).length;
  return cells * 5 >= lines.length * 3;
}

/** The printed lines of one page, rebuilt from its boxes. */
function pageLines(page: PaceVisionPage): LaidLine[] {
  const blocks = page.blocks ?? [];
  if (!blocks.length) {
    // No boxes: the page's own text is still the page's text, it just cannot be
    // highlighted. Keeping it unchanged is better than dropping a page from the
    // document, and the blank lines it prints separate its paragraphs.
    return page.text.split('\n').map((line) => ({ text: line, bbox: null, confidence: null }));
  }
  // A page that already has its lines keeps them. Only a page that arrived one
  // text run per line is laid out again from its boxes.
  if (!isFlatPage(page, blocks)) return rawLines(page);

  const table = isTablePage(blocks);
  const gutter = table ? pageGutter(rowsOf(blocks)) : null;
  if (gutter !== null) {
    // Read the left column to its end and then the right one, the way the page is
    // read: each column is a table of its own, with its own heading and its own
    // term banner, and a title that wrapped stays under the course it belongs to.
    const left = blocks.filter((block) => block.x < gutter);
    const right = blocks.filter((block) => block.x >= gutter);
    return [...linesOfBlocks(left, true), ...linesOfBlocks(right, true)];
  }
  return linesOfBlocks(blocks, table);
}

/**
 * Every page as lines, and the one string the reader sees.
 *
 * Pages are separated by a blank line so a heading at the top of a page stays its
 * own line, and every line records the offset it occupies in the joined text, so
 * an evidence span still resolves to the page and the printed line it came from.
 */
export function layoutPages(pages: PaceVisionPage[]): { text: string; lines: LineGeometry[] } {
  const chunks: { text: string; page: number; line: LaidLine | null }[] = [];
  pages.forEach((page, pageIndex) => {
    for (const line of pageLines(page)) chunks.push({ text: line.text, page: page.page, line });
    // A blank line between pages keeps a heading at the top of one its own line.
    if (pageIndex < pages.length - 1) chunks.push({ text: '', page: page.page, line: null });
  });

  const lines: LineGeometry[] = [];
  const texts: string[] = [];
  let offset = 0;
  for (const chunk of chunks) {
    texts.push(chunk.text);
    if (chunk.line && chunk.text) {
      lines.push({
        page: chunk.page,
        start: offset,
        end: offset + chunk.text.length,
        bbox: chunk.line.bbox,
        confidence: chunk.line.confidence,
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
