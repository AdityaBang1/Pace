/**
 * A transcript, read deterministically.
 *
 * The syllabus reader looks for one course and its grading weights. A transcript
 * is the opposite shape: many courses, one row each, grouped by subject area or
 * by term, with a mark and a credit per row and a printed total underneath. Run
 * the syllabus readers over one and the column header becomes a course title —
 * "Course Mark Weight Credit" — so the app has to recognise the shape first.
 *
 * This is a port of the parts of `pace.transcript_extract` the app needs, kept
 * statement for statement on the decisions that matter: rows are found by their
 * printed columns (or, failing that, by the date/mark/credit shape of a line), a
 * title broken across two printed lines is completed from the line below, a term
 * banner owns the rows printed under it, and every value keeps the exact span it
 * was read from. The Python module has more readers (grade reports, attendance);
 * those layouts still fall through to the syllabus reader and are not claimed
 * here.
 */
import type { Evidence, TranscriptRecord, TranscriptTableTotal } from './types';
import { stableId } from './sha256';
import { lines, type Line } from './rules/text';

export interface Span {
  text: string;
  start: number;
  end: number;
}

export interface Cell {
  text: string;
  start: number;
  end: number;
}

export type TranscriptLayout = 'course_mark_weight_credit' | 'dated_rows' | 'none';

export interface TranscriptCourseRow {
  code: string | null;
  title: string;
  /** As printed: "94", "A-", "P". */
  mark: string | null;
  markPercent: number | null;
  weight: number | null;
  credit: number | null;
  term: string | null;
  section: string | null;
  /** The whole printed row, from the title's first character to the last number. */
  span: Span;
  confidence: number;
}

export interface TranscriptTermRow {
  value: string;
  academicYear: string;
  gradeLevel: string;
  term: string;
  start: number;
  end: number;
  courses: number;
  credits: number;
}

export interface TranscriptAssessment {
  title: string;
  score: number;
  level: number | null;
  label: string | null;
  span: Span;
}

export interface TranscriptReading {
  detected: boolean;
  layout: TranscriptLayout;
  student: string | null;
  school: string | null;
  documentTitle: string | null;
  terms: TranscriptTermRow[];
  courses: TranscriptCourseRow[];
  assessments: TranscriptAssessment[];
  totalCredits: number;
  reportedCredits: number | null;
  reportedGpa: number | null;
  /** One entry per printed table, so each can be checked against its own footer. */
  tableTotals: TranscriptTableTotal[];
  checks: Record<string, number | boolean | string | null>;
}

const MARK_VALUES = [
  'CR-', 'NC', 'CR', 'AUD', 'PASS', 'FAIL',
  'A+', 'A-', 'A', 'B+', 'B-', 'B', 'C+', 'C-', 'C', 'D+', 'D-', 'D',
  'F', 'P', 'W', 'I',
];

const MARK_SET = new Set(MARK_VALUES);

function escaped(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const MARK_ALTERNATION = MARK_VALUES.map(escaped).join('|');

const DATE_CELL = /^\d{1,2}\/\d{1,2}\/\d{2,4}$/;
const GRADE_LEVEL_CELL = /^\d{1,2}$/;
const CREDIT_CELL = /^(?:\d+(?:\.\d{1,3})?|\.\d{1,3})$/;
const SCORE_CELL = /^\d{2,4}$/;
// A printed `Credit: .500` is half a credit, not a typo: the dot may lead.
const NUMERIC_CELL = /^(?:\d+(?:\.\d{1,6})?|\.\d{1,6})$/;
const NUMERIC_TOKEN = NUMERIC_CELL;
const TERM_BANNER =
  /(?<![0-9])(\d{4}\s*-\s*\d{4})\s+Grade\s+(\d{1,2})\s+Term\s+(\d{1,2})/i;
const FOOTER_TOTALS = /(?<![A-Za-z])Credits?\s*:\s*(\d+(?:\.\d{1,4})?|\.\d{1,4})/i;
const FOOTER_GPA = /(?<![A-Za-z])GPA\s*:\s*(\d+(?:\.\d{1,6})?)/i;
// A course number may have no letter suffix (`03391 GAVS AP Psychology`) or a
// three-letter one (`200715ams Accel Geo`).
const COURSE_CODE_PREFIX = /^(\d{3,6}[A-Za-z]{0,3})\s+(?=\S)/;
const PERFORMANCE_LEVEL =
  /Performance\s+Level\s*:?[ \t]*(\d+)[ \t]+([A-Za-z][A-Za-z]*(?:[ \t]+[A-Za-z]+)*)/i;
const LABELLED_LINE = /[A-Za-z]\s*:\s+\S/;
// A row starts with its course number; a school banner with its school code.
const COURSE_CODE_START = /^\d{3,6}[A-Za-z]{0,3}(?:\s|$)/;
const SCHOOL_CODE_LINE = /^#\d{3,5}(?:\s|$)/;

type TableColumn = 'mark' | 'weight' | 'credit' | 'other';

const NUMERIC_COLUMN_NAMES: Record<string, TableColumn> = {
  MARK: 'mark',
  GRADE: 'mark',
  WEIGHT: 'weight',
  CREDIT: 'credit',
  CREDITS: 'credit',
  UNITS: 'credit',
};

/**
 * A header word that is a column of numbers but not one of ours.
 *
 * Transcripts print more numbers than a course needs: grade points, attempted
 * and earned credit, class rank. Refusing the whole table because one of them is
 * unrecognised dropped every row of it, so an unknown trailing column is read as
 * a number and then left on the page.
 */
function otherColumn(name: string): boolean {
  return /^[A-Z][A-Z .]{0,18}$/.test(name) && name !== 'DATE';
}

/** The gap, in characters, within which a printed value sits under its header. */
const COLUMN_REACH = 8;

const TABLE_HEADER_CELLS = new Set([
  'DATE', 'GL', 'TITLE', 'MARK', 'CREDIT', 'TERM', 'SCORE', 'GRADE',
  'GRADE LEVEL', 'PERFORMANCE', 'SUBJECT', 'COURSE', 'PERIOD', 'HOURS',
]);

const NON_SECTION_HEADINGS = new Set([
  'ACADEMIC HISTORY', 'ACADEMIC STANDING', 'DATE GL TITLE MARK CREDIT',
  'DATE GL TITLE MARK CREDIT TERM', 'GRADE POINT AVERAGE', 'TRANSCRIPT',
  'CUMULATIVE GPA', 'STUDENT RECORD',
]);

const INSTITUTION_WORDS = ['HIGH SCHOOL', 'UNIVERSITY', 'COLLEGE', 'DISTRICT', 'ACADEMY'];
const DOCUMENT_TITLE_HINTS = ['HISTORY', 'TRANSCRIPT', 'ACADEMIC RECORD', 'STUDENT RECORD'];

const SECTION_HEADING = /^[A-Z][A-Z0-9 &/,.()'\-]{1,60}$/;
const STUDENT_NAME = /^[A-Z][A-Za-z'’\-]+,\s+[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*)*$/;
// The labels a school prints beside an identity line, never a value of their own.
const HEADER_IDENTITY_LABEL = /\b(?:Sex|DOB|SSN|Entry Date|Exit Date|Graduation)\b/i;
// "Phone: (907) xxx-xxxx": a label with its colon is not an institution name.
const HEADER_LABEL_LINE = /^[A-Za-z][A-Za-z /]*:\s*\S/;

/** Python's `(?<![A-Za-z])MARK(?![A-Za-z])`, assembled per call. */
const LOOSE_COURSE = new RegExp(
  `(\\d{1,2}\\/\\d{1,2}\\/\\d{2,4})[ \\t]+(\\d{1,2})[ \\t]+(.+?\\S)[ \\t]+(${MARK_ALTERNATION})[ \\t]+(\\d+(?:\\.\\d{1,3})?)(?![\\d.])`,
  'g',
);
const LOOSE_ASSESSMENT =
  /(\d{1,2}\/\d{1,2}\/\d{2,4})[ \t]+(\d{1,2})[ \t]+(.+?\S)[ \t]+(\d{2,4})(?!\d|\.\d)/g;

const COLUMN_SEPARATOR = /[ ]*\t+[ ]*|[ ]{2,}|[ ]*\|[ ]*/g;

function spanOf(text: string, start: number, end: number): Span {
  return { text: text.slice(start, end), start, end };
}

function stripSpan(text: string, start: number, end: number): Span | null {
  let from = start;
  let to = end;
  while (from < to && /\s/.test(text[from])) from += 1;
  while (to > from && /\s/.test(text[to - 1])) to -= 1;
  if (from >= to) return null;
  return spanOf(text, from, to);
}

/** Split a line into column cells on the whitespace gaps a table prints. */
export function cells(body: string): Cell[] {
  const out: Cell[] = [];
  let position = 0;
  COLUMN_SEPARATOR.lastIndex = 0;
  for (let match = COLUMN_SEPARATOR.exec(body); match; match = COLUMN_SEPARATOR.exec(body)) {
    if (match.index > position) {
      out.push({ text: body.slice(position, match.index), start: position, end: match.index });
    }
    position = match.index + match[0].length;
    if (match[0].length === 0) COLUMN_SEPARATOR.lastIndex += 1;
  }
  if (position < body.length) {
    out.push({ text: body.slice(position), start: position, end: body.length });
  }
  return out
    .map((cell) => {
      const leading = cell.text.length - cell.text.trimStart().length;
      const trailing = cell.text.length - cell.text.trimEnd().length;
      return {
        text: cell.text.trim(),
        start: cell.start + leading,
        end: cell.end - trailing,
      };
    })
    .filter((cell) => cell.text.length > 0);
}

function numericColumns(list: Cell[]): TableColumn[] | null {
  const names = list.map((cell) => cell.text.toUpperCase());
  if (names.includes('DATE')) return null;
  if (!names.includes('MARK') || !names.includes('CREDIT')) return null;
  if (!names.some((name) => name === 'COURSE' || name === 'COURSES' || name === 'TITLE')) return null;
  const first = names.findIndex((name) => name in NUMERIC_COLUMN_NAMES);
  if (first <= 0) return null;
  const numeric = names.slice(first);
  if (numeric.some((name) => !(name in NUMERIC_COLUMN_NAMES) && !otherColumn(name))) return null;
  const columns: TableColumn[] = [];
  for (const name of numeric) {
    // A two-word header printed with a wide gap ("Grade  Points") arrives as two
    // cells. The second word is not a second mark column; it is more of the first
    // column's name, and the table still has one column of each.
    const known = NUMERIC_COLUMN_NAMES[name];
    columns.push(known && !columns.includes(known) ? known : 'other');
  }
  if (!columns.includes('mark') || !columns.includes('credit')) return null;
  return columns;
}

function headerColumns(list: Cell[], body: string): TableColumn[] | null {
  const direct = numericColumns(list);
  if (direct) return direct;
  if (body.split(/\s+/).filter(Boolean).length < 3) return null;
  return numericColumns(tokensOf(body));
}

/** The words of a line, each with the offset it starts at. */
function tokensOf(body: string): Cell[] {
  const tokens: Cell[] = [];
  const tokenRe = /\S+/g;
  for (let match = tokenRe.exec(body); match; match = tokenRe.exec(body)) {
    tokens.push({ text: match[0], start: match.index, end: match.index + match[0].length });
  }
  return tokens;
}

/** Where each printed column starts, so a value can be matched to its own column. */
function columnStarts(body: string, columns: TableColumn[]): { start: number; name: TableColumn }[] | null {
  const tokens = tokensOf(body);
  if (tokens.length < columns.length) return null;
  return tokens.slice(tokens.length - columns.length).map((token, index) => ({
    start: token.start,
    name: columns[index],
  }));
}

type NumericValue = { text: string; value: number | null; start: number; end: number };

function numericCell(name: TableColumn, cell: Cell): NumericValue | null {
  // A mark column may print a letter (P, NC, CR) beside its percentages: a term
  // block with a pass/fail row is still a term block.
  if (name === 'mark' && MARK_SET.has(cell.text)) {
    return { text: cell.text, value: null, start: cell.start, end: cell.end };
  }
  if (!NUMERIC_CELL.test(cell.text)) return null;
  const value = Number(cell.text);
  if (name === 'mark') {
    if (value < 0 || value > 100) return null;
  } else if (value > 100) {
    return null;
  }
  return { text: cell.text, value, start: cell.start, end: cell.end };
}

function numericRow(
  list: Cell[],
  columns: TableColumn[],
): { title: Cell[]; values: Record<string, NumericValue> } | null {
  let index = list.length;
  const values: Record<string, NumericValue> = {};
  for (const name of [...columns].reverse()) {
    index -= 1;
    if (index < 0) return null;
    const value = numericCell(name, list[index]);
    if (!value) return null;
    values[name] = value;
  }
  if (index === 0) return null;
  return { title: list.slice(0, index), values };
}

function looseNumericRow(
  body: string,
  columns: TableColumn[],
): { titleStart: number; titleEnd: number; values: Record<string, NumericValue> } | null {
  // The columns are read from the right edge of the line: a row is a title
  // followed by one number per named column, in header order. Reading right to
  // left also keeps each number's offset, which a group-index match would only
  // give with the `d` flag (not available on every engine).
  const values: Record<string, NumericValue> = {};
  let end = body.trimEnd().length;
  for (const name of [...columns].reverse()) {
    // A template literal eats an unknown escape, so the regex escapes here are
    // doubled: with single backslashes the mark column's own alternative is
    // `d+` and no number ever matches it.
    const token = name === 'mark'
      ? `(\\d+(?:\\.\\d{1,6})?|${MARK_ALTERNATION})`
      : '(\\d+(?:\\.\\d{1,6})?)';
    // The number must be its own token, not the tail of a word or the decimals of
    // another number: `1.0000` ends in `0000`, and a lazy prefix would happily
    // read that as the weight of a course whose title is `Intro to Software
    // Tech`, dropping the row. Anchoring the match to a whitespace boundary keeps
    // the whole number.
    const piece = new RegExp(`(?:^|\\s)(${token})$`).exec(body.slice(0, end));
    if (!piece) return null;
    const text = piece[1];
    const start = piece.index + piece[0].length - text.length;
    const value = numericCell(name, { text, start, end: start + text.length });
    if (!value) return null;
    values[name] = value;
    end = body.slice(0, start).trimEnd().length;
  }
  const titleEnd = end;
  const titleStart = body.length - body.trimStart().length;
  if (titleEnd <= titleStart) return null;
  return { titleStart, titleEnd, values };
}

/** One term banner, named the same way wherever it is read. */
function bannerValue(match: RegExpExecArray): string {
  return `${match[1].replace(/\s*-\s*/g, '-')} Grade ${match[2]} Term ${match[3]}`;
}

/** Read the term banners a transcript prints above its rows. */
export function termBanners(text: string, all: Line[]): TranscriptTermRow[] {
  const out: TranscriptTermRow[] = [];
  for (const line of all) {
    const match = TERM_BANNER.exec(line.text);
    if (!match) continue;
    out.push({
      value: bannerValue(match),
      academicYear: match[1].replace(/\s*-\s*/g, '-'),
      gradeLevel: match[2],
      term: match[3],
      start: line.start + match.index,
      end: line.start + match.index + match[0].length,
      courses: 0,
      credits: 0,
    });
  }
  return out;
}

/** A subject area heading, as printed ("LANGUAGE ARTS"). */
function headingText(value: string): boolean {
  if (NON_SECTION_HEADINGS.has(value) || MARK_SET.has(value)) return false;
  if (!SECTION_HEADING.test(value)) return false;
  return !INSTITUTION_WORDS.some((word) => value.includes(word));
}

function isHeaderRun(list: Cell[]): boolean {
  if (list.length < 3) return false;
  const texts = list.map((cell) => cell.text.toUpperCase());
  if (!texts.includes('DATE') || !texts.includes('TITLE')) return false;
  return texts.every((text) => TABLE_HEADER_CELLS.has(text));
}

/** Cells that are not a table header, or null when the line carries none. */
function withoutHeader(list: Cell[]): Cell[] | null {
  if (!list.length) return null;
  if (isHeaderRun(list)) return [];
  let lead = 0;
  while (lead < list.length && TABLE_HEADER_CELLS.has(list[lead].text.toUpperCase())) lead += 1;
  if (isHeaderRun(list.slice(0, lead))) {
    let remaining = list.slice(lead);
    let trail = 0;
    while (trail < remaining.length && TABLE_HEADER_CELLS.has(remaining[remaining.length - 1 - trail].text.toUpperCase())) {
      trail += 1;
    }
    if (trail && isHeaderRun(remaining.slice(remaining.length - trail))) {
      remaining = remaining.slice(0, remaining.length - trail);
    }
    return remaining;
  }
  let trail = 0;
  while (trail < list.length && TABLE_HEADER_CELLS.has(list[list.length - 1 - trail].text.toUpperCase())) trail += 1;
  if (trail && isHeaderRun(list.slice(list.length - trail))) return list.slice(0, list.length - trail);
  return null;
}

interface DatedRecord {
  kind: 'course' | 'assessment';
  date: string;
  gradeLevel: string;
  title: string;
  mark?: string;
  credit?: number;
  score?: number;
  start: number;
  end: number;
}

/** One dated row, read from its cells: date, grade level, title, mark, credit. */
function datedRecord(
  list: Cell[],
  index: number,
  body: string,
  lineStart: number,
): { record: DatedRecord | null; next: number } {
  if (index + 1 >= list.length) return { record: null, next: index + 1 };
  if (!GRADE_LEVEL_CELL.test(list[index + 1].text)) return { record: null, next: index + 1 };
  let end = index + 2;
  while (end < list.length && !DATE_CELL.test(list[end].text)) end += 1;
  const region = list.slice(index + 2, end);
  if (!region.length) return { record: null, next: end };

  for (let position = 0; position < region.length - 1; position += 1) {
    const mark = region[position];
    const credit = region[position + 1];
    if (!MARK_SET.has(mark.text) || !CREDIT_CELL.test(credit.text)) continue;
    const titleCells = region.slice(0, position);
    if (!titleCells.length) return { record: null, next: end };
    return {
      record: {
        kind: 'course',
        date: list[index].text,
        gradeLevel: list[index + 1].text,
        title: body.slice(titleCells[0].start, titleCells[titleCells.length - 1].end),
        mark: mark.text,
        credit: Number(credit.text),
        start: lineStart + list[index].start,
        end: lineStart + credit.end,
      },
      next: end,
    };
  }

  const score = region[region.length - 1];
  if (!SCORE_CELL.test(score.text)) return { record: null, next: end };
  const titleCells = region.slice(0, -1);
  if (!titleCells.length) return { record: null, next: end };
  return {
    record: {
      kind: 'assessment',
      date: list[index].text,
      gradeLevel: list[index + 1].text,
      title: body.slice(titleCells[0].start, titleCells[titleCells.length - 1].end),
      score: Number(score.text),
      start: lineStart + list[index].start,
      end: lineStart + score.end,
    },
    next: end,
  };
}

/** A dated row whose columns collapsed to single spaces. */
function looseDatedRecords(body: string, lineStart: number): DatedRecord[] {
  const out: DatedRecord[] = [];
  LOOSE_COURSE.lastIndex = 0;
  for (let match = LOOSE_COURSE.exec(body); match; match = LOOSE_COURSE.exec(body)) {
    const before = match.index > 0 ? body[match.index - 1] : '';
    if (before && /[\d/]/.test(before)) continue;
    out.push({
      kind: 'course',
      date: match[1],
      gradeLevel: match[2],
      title: match[3].trim(),
      mark: match[4],
      credit: Number(match[5]),
      start: lineStart + match.index,
      end: lineStart + match.index + match[0].length,
    });
  }
  LOOSE_ASSESSMENT.lastIndex = 0;
  for (let match = LOOSE_ASSESSMENT.exec(body); match; match = LOOSE_ASSESSMENT.exec(body)) {
    const before = match.index > 0 ? body[match.index - 1] : '';
    if (before && /[\d/]/.test(before)) continue;
    if (out.some((row) => row.start === lineStart + match.index)) continue;
    out.push({
      kind: 'assessment',
      date: match[1],
      gradeLevel: match[2],
      title: match[3].trim(),
      score: Number(match[4]),
      start: lineStart + match.index,
      end: lineStart + match.index + match[0].length,
    });
  }
  return out;
}

/** The x position where a right page column starts, when the OCR merged them. */
function columnSeam(all: Cell[][]): number | null {
  const candidates: number[] = [];
  for (const list of all) {
    const dates = list.filter((cell) => DATE_CELL.test(cell.text)).map((cell) => cell.start);
    if (dates.length >= 2) candidates.push(dates[1]);
  }
  if (!candidates.length) return null;
  const sorted = [...candidates].sort((left, right) => left - right);
  const seam = sorted[Math.floor(sorted.length / 2)];
  const agreeing = candidates.filter((value) => Math.abs(value - seam) <= 2).length;
  if (agreeing * 2 < candidates.length) return null;
  return seam;
}

/**
 * A row whose printed columns are not all filled.
 *
 * A transcript leaves the weight of an unweighted course blank, and a row with two
 * numbers where the header names three used to be dropped without a word. Matching
 * each printed number to the header column above it keeps the row, with the empty
 * column left empty instead of shifting its neighbours into the wrong fields.
 */
function alignedRow(
  list: Cell[],
  starts: { start: number; name: TableColumn }[],
): { title: Cell[]; values: Record<string, NumericValue> } | null {
  // Only the numbers at the right of the row can be columns; a digit inside a
  // course title is part of the title.
  let first = list.length;
  while (first > 0) {
    const cell = list[first - 1];
    if (!NUMERIC_TOKEN.test(cell.text) && !MARK_SET.has(cell.text)) break;
    first -= 1;
  }
  if (first >= list.length) return null;

  const values: Record<string, NumericValue> = {};
  let titleEnd = list[list.length - 1].end;
  for (let index = first; index < list.length; index += 1) {
    const cell = list[index];
    let chosen: { name: TableColumn; distance: number } | null = null;
    for (const column of starts) {
      if (values[column.name]) continue;
      const distance = Math.abs(cell.start - column.start);
      if (distance > COLUMN_REACH) continue;
      if (!chosen || distance < chosen.distance) chosen = { name: column.name, distance };
    }
    if (!chosen) return null;
    const value = numericCell(chosen.name, cell);
    if (!value) return null;
    values[chosen.name] = value;
    if (cell.start < titleEnd) titleEnd = cell.start;
  }
  if (!values.mark && !values.credit) return null;
  const title = list.filter((cell) => cell.end <= titleEnd);
  if (!title.length) return null;
  return { title, values };
}

function columnGroups(list: Cell[], seam: number | null): { column: number; cells: Cell[] }[] {
  if (seam === null) return list.length ? [{ column: 0, cells: list }] : [];
  const split: Cell[] = [];
  for (const cell of list) {
    if (cell.start < seam && seam < cell.end) {
      const rightText = cell.text.slice(seam - cell.start).trim();
      const rightStart = cell.end - cell.text.slice(seam - cell.start).trimEnd().length;
      if (rightText) {
        split.push({ text: cell.text.slice(0, seam - cell.start).trim(), start: cell.start, end: seam });
        split.push({ text: rightText, start: rightStart, end: cell.end });
        continue;
      }
    }
    split.push(cell);
  }
  const left = split.filter((cell) => cell.start < seam);
  const right = split.filter((cell) => cell.start >= seam);
  const groups: { column: number; cells: Cell[] }[] = [];
  if (left.length) groups.push({ column: 0, cells: left });
  if (right.length) groups.push({ column: 1, cells: right });
  return groups;
}

/** Read a COURSE / MARK / WEIGHT / CREDIT block; rows carry numbers, not dates. */
function numericTable(
  text: string,
  all: Line[],
  allCells: Cell[][],
  headerIndex: number,
  endIndex: number,
  columns: TableColumn[],
): TranscriptCourseRow[] {
  const out: TranscriptCourseRow[] = [];
  const starts = columnStarts(all[headerIndex].text, columns);
  for (let lineIndex = headerIndex + 1; lineIndex < endIndex; lineIndex += 1) {
    const body = all[lineIndex].text;
    const list = allCells[lineIndex];
    if (!body.trim() || !list.length) continue;
    if (TERM_BANNER.test(body)) continue;
    // A two-column page whose columns landed on one text line prints two rows on
    // it. Reading only the rightmost numbers would fold the left course's title
    // into the right one, so the line is split where both halves are rows.
    const merged = splitMergedRow(list, columns, starts);
    if (merged) {
      for (const part of merged) {
        const row = numericTableRow(text, all, allCells, lineIndex, endIndex, columns, part, starts);
        if (row) out.push(row);
      }
      continue;
    }
    const row = numericTableRow(text, all, allCells, lineIndex, endIndex, columns, list, starts);
    if (row) out.push(row);
  }
  return out;
}

/** Two printed rows that share one text line, split where both halves are rows. */
function splitMergedRow(
  list: Cell[],
  columns: TableColumn[],
  starts: { start: number; name: TableColumn }[] | null,
): Cell[][] | null {
  if (list.length < columns.length * 2 + 2) return null;
  const isRow = (cells: Cell[]) =>
    numericRow(cells, columns) !== null || (starts !== null && alignedRow(cells, starts) !== null);
  for (let split = columns.length + 1; split <= list.length - columns.length - 1; split += 1) {
    const left = list.slice(0, split);
    const right = list.slice(split);
    if (isRow(left) && isRow(right)) return [left, right];
  }
  return null;
}

function numericTableRow(
  text: string,
  all: Line[],
  allCells: Cell[][],
  lineIndex: number,
  endIndex: number,
  columns: TableColumn[],
  list: Cell[],
  starts: { start: number; name: TableColumn }[] | null,
): TranscriptCourseRow | null {
  const body = all[lineIndex].text;
  const lineStart = all[lineIndex].start;

  let titleStart: number;
  let titleEnd: number;
  let values: Record<string, NumericValue>;
  const grid = numericRow(list, columns);
  if (grid) {
    titleStart = lineStart + grid.title[0].start;
    titleEnd = lineStart + grid.title[grid.title.length - 1].end;
    values = grid.values;
  } else {
    const loose = looseNumericRow(body, columns);
    if (loose) {
      titleStart = lineStart + loose.titleStart;
      titleEnd = lineStart + loose.titleEnd;
      values = loose.values;
    } else {
      const aligned = starts ? alignedRow(list, starts) : null;
      if (!aligned) return null;
      titleStart = lineStart + aligned.title[0].start;
      titleEnd = lineStart + aligned.title[aligned.title.length - 1].end;
      values = aligned.values;
    }
  }

  let code: string | null = null;
  const prefix = COURSE_CODE_PREFIX.exec(text.slice(titleStart, titleEnd));
  if (prefix) {
    code = prefix[1];
    titleStart += prefix[0].length;
  }
  if (titleStart >= titleEnd) return null;
  const rowEnd = lineStart + list[list.length - 1].end;

  let title = text.slice(titleStart, titleEnd).replace(/\s+/g, ' ').trim();
  let end = rowEnd;
  // A title broken by a printed line break continues on the line directly
  // below, which has no numbers of its own.
  let cursor = lineIndex + 1;
  let wrapped = 0;
  while (cursor < endIndex && wrapped < 2) {
    const followingBody = all[cursor].text;
    const following = allCells[cursor];
    if (!followingBody.trim() || !following.length) break;
    const stripped = followingBody.trim();
    if (stripped === stripped.toUpperCase()) break;
    if (TERM_BANNER.test(followingBody)) break;
    if (FOOTER_TOTALS.test(followingBody) || FOOTER_GPA.test(followingBody)) break;
    if (numericColumns(following) !== null) break;
    if (following.some((cell) => NUMERIC_TOKEN.test(cell.text))) break;
    if (LABELLED_LINE.test(stripped)) break;
    // A title's tail never starts a new course and never carries the school code
    // a table is printed under. Reading either as the tail consumed the next row:
    // one course lost its own line and the one above it wore the school's name.
    if (COURSE_CODE_START.test(stripped)) break;
    if (SCHOOL_CODE_LINE.test(stripped)) break;
    if (looseNumericRow(followingBody, columns) !== null) break;
    const extra = followingBody.slice(following[0].start, following[following.length - 1].end);
    title = `${title} ${extra.replace(/\s+/g, ' ').trim()}`.trim();
    end = all[cursor].start + following[following.length - 1].end;
    cursor += 1;
    wrapped += 1;
  }

  const markValue = values.mark?.value ?? null;
  return {
    code,
    title,
    mark: values.mark?.text ?? null,
    markPercent:
      markValue == null
        ? null
        : markValue === Math.trunc(markValue)
          ? Math.trunc(markValue)
          : markValue,
    weight: values.weight?.value ?? null,
    credit: values.credit?.value ?? null,
    term: null,
    section: null,
    span: spanOf(text, titleStart, end),
    confidence: wrapped ? 0.9 : 0.95,
  };
}

/** The identity line and the school banner printed above the table. */
function readHeader(
  text: string,
  headerLines: Line[],
): { student: string | null; school: string | null; documentTitle: string | null } {
  let student: string | null = null;
  let documentTitle: string | null = null;
  const schoolLines: string[] = [];
  for (const line of headerLines) {
    const value = line.text.trim();
    if (!value) continue;
    // A term banner names the term, not the school.
    if (TERM_BANNER.test(value)) continue;
    // A labelled identity line ("Lepid, Arthur    Sex  Male") carries the student
    // name before the first label; the label values are not course data.
    if (HEADER_IDENTITY_LABEL.test(value)) {
      const name = value.split(/\s{2,}|\t/)[0].trim();
      if (!student && STUDENT_NAME.test(name)) student = name;
      continue;
    }
    // A page may print its title and the student's name side by side on one line
    // ("South Forsyth High School Transcript    Bang, Aditya Prashant"), and a
    // school's name and its address on another. Each printed column is its own
    // value, so the line is split where the page left a wide gap.
    for (const part of value.split(/\s{2,}|\t/).map((piece) => piece.trim()).filter(Boolean)) {
      if (!student && STUDENT_NAME.test(part)) {
        student = part;
        continue;
      }
      // A label and its value, an address, or a school code names no school.
      if (HEADER_LABEL_LINE.test(part) || SCHOOL_CODE_LINE.test(part) || /\d{3,}/.test(part)) continue;
      schoolLines.push(part);
    }
  }
  const keeps: string[] = [];
  for (const value of schoolLines) {
    const upper = value.toUpperCase();
    if (!documentTitle && (NON_SECTION_HEADINGS.has(upper) || DOCUMENT_TITLE_HINTS.some((hint) => upper.includes(hint)))) {
      documentTitle = value;
      continue;
    }
    keeps.push(value);
  }
  // The institution is the line that names one; failing that, the document's own
  // title (a page headed "South Forsyth High School Transcript" is that school's
  // transcript); failing that, the first line that was not an address.
  const named = keeps.find((value) =>
    INSTITUTION_WORDS.some((word) => value.toUpperCase().includes(word)),
  );
  const titled = documentTitle && INSTITUTION_WORDS.some((word) => documentTitle!.toUpperCase().includes(word))
    ? documentTitle.replace(/\s+(?:Transcript|Academic Record|Student Record|Record|History)\s*$/i, '').trim()
    : null;
  const school = named ?? titled ?? (keeps.length ? keeps[0] : null);
  return { student, school, documentTitle };
}

/**
 * The last line below a table's footer that is still one of its rows.
 *
 * Some transcripts print the column header once per page (or once per term) and
 * a footer under every page. The rows under that footer still sit in the columns
 * above them, so a table that stopped at its first footer lost every page behind
 * it — which is the shape of "it read the first term and nothing else".
 */
function tableContinues(
  all: Line[],
  allCells: Cell[][],
  from: number,
  columns: TableColumn[],
  term: string | null,
): number | null {
  let last: number | null = null;
  for (let index = from; index < all.length; index += 1) {
    const body = all[index].text;
    if (!body.trim()) continue;
    if (headerColumns(allCells[index], body) !== null) break;
    const banner = TERM_BANNER.exec(body);
    // The next page may print the same term's banner again above its rows; a
    // different term is a different table, and it needs its own columns.
    if (banner) {
      if (bannerValue(banner) !== term) break;
      // The same term printed again above the next page's rows: those rows are
      // still this table.
      continue;
    }
    const row = numericRow(allCells[index], columns) ?? looseNumericRow(body, columns);
    if (!row) break;
    last = index;
  }
  return last;
}

/** The totals a table prints under itself ("Credit: 7.500   GPA: 4.13333"). */
function footerOf(line: Line | undefined): { credits: number | null; gpa: number | null } | null {
  if (!line) return null;
  const credits = FOOTER_TOTALS.exec(line.text);
  const gpa = FOOTER_GPA.exec(line.text);
  if (!credits && !gpa) return null;
  return { credits: credits ? Number(credits[1]) : null, gpa: gpa ? Number(gpa[1]) : null };
}

function readStanding(
  all: Line[],
): { reportedCredits: number | null; reportedGpa: number | null; asOf: string | null } {
  let reportedCredits: number | null = null;
  let reportedGpa: number | null = null;
  let asOf: string | null = null;
  const heading = all.findIndex((line) => line.text.trim().toUpperCase() === 'ACADEMIC STANDING');
  if (heading === -1) return { reportedCredits, reportedGpa, asOf };
  for (const line of all.slice(heading + 1)) {
    const body = line.text;
    if (!body.trim()) continue;
    if (body.toUpperCase().includes('AS OF')) {
      const match = /\d{1,2}\/\d{1,2}\/\d{2,4}/.exec(body);
      if (match) asOf = match[0];
      continue;
    }
    const values = /(?:(\d{1,2}\/\d{1,2}\/\d{2,4}))?[ \t]+(\d+(?:\.\d{1,3})?)?[ \t]*(\d+\.\d{1,4})?[ \t]*$/.exec(body);
    if (!values || (!values[1] && !values[2] && !values[3])) continue;
    if (values[1] && !asOf) asOf = values[1];
    if (values[2] && reportedCredits === null) reportedCredits = Number(values[2]);
    if (values[3] && reportedGpa === null) reportedGpa = Number(values[3]);
    if (reportedCredits !== null) break;
  }
  return { reportedCredits, reportedGpa, asOf };
}

/** The words a printed column header can be made of, one per line or all on one. */
const SPLIT_HEADER_WORDS = new Set([
  'COURSE', 'COURSES', 'TITLE', 'MARK', 'GRADE', 'WEIGHT', 'CREDIT', 'CREDITS',
  'UNITS', 'TERM', 'SCORE', 'GPA', 'POINTS', 'ATTEMPTED', 'EARNED',
]);

/** A column header printed one word per line, as a PDF text layer reports it. */
function splitHeader(
  text: string,
  all: Line[],
  index: number,
): { line: Line; last: number; columns: TableColumn[] } | null {
  const words: string[] = [];
  let last = index;
  for (let cursor = index; cursor < all.length && words.length < 6; cursor += 1) {
    const value = all[cursor].text.trim();
    if (!value || /\s/.test(value)) break;
    const upper = value.toUpperCase();
    if (!SPLIT_HEADER_WORDS.has(upper)) break;
    words.push(upper);
    last = cursor;
  }
  if (words.length < 3) return null;
  if (!words.includes('MARK') || !words.includes('CREDIT')) return null;
  const columns = numericColumns(tokensOf(words.join(' ')));
  if (!columns) return null;
  return {
    line: { text: text.slice(all[index].start, all[last].end), start: all[index].start, end: all[last].end },
    last,
    columns,
  };
}

/**
 * Put a table back together when its cells were printed one per line.
 *
 * A PDF text layer reports the runs the author drew, which for a table is the
 * column header one word per line and then, for each course, the title and the
 * values under it on lines of their own. The document's own order is right there;
 * what is missing is the row. This joins a run of header words into one header,
 * and a line that holds only a row's values back onto the title lines above it.
 *
 * The joined line is the document's own slice, so every offset in it is still a
 * real position in the page and a value can be quoted from it.
 */
function tableLines(text: string, all: Line[]): Line[] {
  const out: Line[] = [];
  let columns: TableColumn[] | null = null;
  // The title lines above the values, and the value lines themselves: a text
  // layer may report the whole row on one line or each number on its own line.
  let pending: Line[] = [];
  let numbers: Line[] = [];

  const flush = () => {
    if (pending.length) out.push(...pending);
    if (numbers.length) out.push(...numbers);
    pending = [];
    numbers = [];
  };

  for (let index = 0; index < all.length; index += 1) {
    const line = all[index];
    const body = line.text;
    const trimmed = body.trim();
    if (!trimmed) {
      flush();
      out.push(line);
      continue;
    }

    const header = splitHeader(text, all, index);
    if (header) {
      flush();
      out.push(header.line);
      columns = header.columns;
      index = header.last;
      continue;
    }

    if (TERM_BANNER.test(body) || FOOTER_TOTALS.test(body) || FOOTER_GPA.test(body)) {
      flush();
      out.push(line);
      continue;
    }

    if (columns) {
      const width = columns;
      const tokens = trimmed.split(/\s+/);
      // Numbers only, one per column: a values line, or one column of one.
      const valuesOnly =
        tokens.length <= width.length &&
        tokens.every(
          (token, position) =>
            numericCell(width[position], { text: token, start: 0, end: token.length }) !== null,
        );
      if (valuesOnly) {
        numbers.push(line);
        if (numbers.length < width.length) continue;
        // A whole row's worth of numbers under some title lines is that row.
        if (pending.length) {
          const first = pending[0];
          const joined: Line = { text: text.slice(first.start, line.end), start: first.start, end: line.end };
          // Only when the joined line really is a row: a stray run of numbers
          // under prose must not consume the prose as a title.
          if (numericRow(cells(joined.text), width) || looseNumericRow(joined.text, width)) {
            out.push(joined);
            pending = [];
            numbers = [];
            continue;
          }
        }
        out.push(...numbers);
        numbers = [];
        continue;
      }
      if (numbers.length) {
        out.push(...numbers);
        numbers = [];
      }
    }

    pending.push(line);
    // A title is one or two lines at most; anything further above a value line
    // is a line of its own, printed as it came.
    while (pending.length > 2) {
      out.push(pending.shift() as Line);
    }
  }
  flush();
  return out;
}

export function readTranscript(text: string): TranscriptReading {
  const all = tableLines(text, lines(text));
  const allCells = all.map((line) => (line.text.trim() ? cells(line.text) : []));
  const seam = columnSeam(allCells);
  const terms = termBanners(text, all);

  // Every numeric block, not just the first: a transcript prints one per term or
  // per page, and reading only the first would silently drop every later page.
  const numericBlocks: {
    header: number;
    end: number;
    columns: TableColumn[];
    footer: number | null;
  }[] = [];
  let probe = 0;
  while (probe < all.length) {
    const columns = headerColumns(allCells[probe], all[probe].text);
    if (!columns) {
      probe += 1;
      continue;
    }
    let end = all.length;
    let footer: number | null = null;
    for (let cursor = probe + 1; cursor < all.length; cursor += 1) {
      const body = all[cursor].text;
      if (FOOTER_TOTALS.test(body) || FOOTER_GPA.test(body)) {
        end = cursor;
        footer = cursor;
        break;
      }
      if (headerColumns(allCells[cursor], body) !== null) {
        end = cursor;
        break;
      }
    }
    // A header printed once per page or per term carries over its own footer.
    if (footer !== null) {
      let term: string | null = null;
      for (const banner of terms) {
        if (banner.start <= all[probe].start) term = banner.value;
        else break;
      }
      const continued = tableContinues(all, allCells, footer + 1, columns, term);
      if (continued !== null) {
        end = continued + 1;
        // A title that wrapped below the last row is still that row's title, and
        // the rows are what the block reader walks; without the line it would be
        // cut off at the page break.
        while (end < all.length) {
          const body = all[end].text;
          const trimmed = body.trim();
          if (!trimmed) {
            end += 1;
            continue;
          }
          if (FOOTER_TOTALS.test(body) || FOOTER_GPA.test(body)) break;
          if (headerColumns(allCells[end], body) !== null) break;
          if (TERM_BANNER.test(body)) break;
          if (trimmed === trimmed.toUpperCase() || /\d/.test(trimmed)) break;
          end += 1;
        }
        if (footerOf(all[end])) footer = end;
      }
    }
    numericBlocks.push({ header: probe, end, columns, footer });
    probe = end;
  }

  const owned = (index: number) =>
    numericBlocks.some((block) => block.header <= index && index < block.end);

  interface Heading {
    name: string;
    start: number;
    column: number;
  }
  const headings: Heading[] = [];
  const dated: (DatedRecord & { column: number })[] = [];
  let columnHeaderOffset: number | null = null;

  for (let lineIndex = 0; lineIndex < all.length; lineIndex += 1) {
    const line = all[lineIndex];
    const body = line.text;
    if (!body.trim()) continue;
    if (owned(lineIndex)) {
      if (numericBlocks[0] && lineIndex === numericBlocks[0].header && columnHeaderOffset === null) {
        columnHeaderOffset = line.start;
      }
      continue;
    }
    const groups = columnGroups(allCells[lineIndex], seam);
    const lineRecords: DatedRecord[] = [];
    for (const group of groups) {
      let list = group.cells;
      const remaining = withoutHeader(list);
      if (remaining !== null) {
        if (columnHeaderOffset === null) columnHeaderOffset = line.start;
        list = remaining;
        if (!list.length) continue;
      }
      let index = 0;
      let found = false;
      while (index < list.length) {
        const cell = list[index];
        if (!DATE_CELL.test(cell.text)) {
          if (headingText(cell.text)) {
            headings.push({ name: cell.text, start: line.start + cell.start, column: group.column });
          }
          index += 1;
          continue;
        }
        const parsed = datedRecord(list, index, body, line.start);
        if (parsed.record) {
          dated.push({ ...parsed.record, column: group.column });
          found = true;
        }
        index = parsed.next;
      }
      if (!found) {
        const groupStart = list[0].start;
        const groupEnd = list[list.length - 1].end;
        for (const record of looseDatedRecords(body.slice(groupStart, groupEnd), line.start + groupStart)) {
          dated.push({ ...record, column: group.column });
        }
      }
    }
  }

  const courses: TranscriptCourseRow[] = [];
  const assessments: { title: string; score: number; level: number | null; label: string | null; span: Span }[] = [];
  for (const record of dated) {
    const section = headings.reduce<Heading | null>((chosen, heading) => {
      if (heading.column !== record.column) return chosen;
      return heading.start <= record.start ? heading : chosen;
    }, null);
    if (record.kind === 'course') {
      courses.push({
        code: null,
        title: record.title,
        mark: record.mark ?? null,
        markPercent: null,
        weight: null,
        credit: record.credit ?? null,
        term: null,
        section: section?.name ?? null,
        span: spanOf(text, record.start, record.end),
        confidence: 0.95,
      });
    } else {
      const after = text.slice(record.end, record.end + 200);
      const performance = PERFORMANCE_LEVEL.exec(after);
      assessments.push({
        title: record.title,
        score: record.score ?? 0,
        level: performance ? Number(performance[1]) : null,
        label: performance ? performance[2].trim() : null,
        span: spanOf(text, record.start, record.end),
      });
    }
  }

  const numericRows: {
    rows: TranscriptCourseRow[];
    footer: { credits: number | null; gpa: number | null } | null;
  }[] = [];
  for (const block of numericBlocks) {
    const rows = numericTable(text, all, allCells, block.header, block.end, block.columns);
    numericRows.push({ rows, footer: block.footer !== null ? footerOf(all[block.footer]) : null });
    courses.push(...rows);
  }
  if (numericBlocks.length) courses.sort((left, right) => left.span.start - right.span.start);

  // A term banner owns the rows printed below it, in its own page.
  for (const course of courses) {
    let owner: TranscriptTermRow | null = null;
    for (const term of terms) {
      if (term.start <= course.span.start) owner = term;
      else break;
    }
    course.term = owner?.value ?? null;
    if (owner) {
      owner.courses += 1;
      owner.credits = Math.round((owner.credits + (course.credit ?? 0)) * 1000) / 1000;
    }
  }

  const firstRecord = [...courses.map((row) => row.span.start), ...assessments.map((row) => row.span.start)]
    .sort((left, right) => left - right)[0] ?? text.length;
  const bodyStart = Math.min(firstRecord, columnHeaderOffset ?? text.length);
  const bodyLine = all.filter((line) => line.start < bodyStart).length;
  const header = readHeader(text, all.slice(0, Math.max(bodyLine, 0)));
  const standing = readStanding(all);

  const tableTotals: TranscriptTableTotal[] = [];
  for (const table of numericRows) {
    const term = table.rows.find((row) => row.term !== null)?.term ?? null;
    const credits =
      Math.round(table.rows.reduce((sum, row) => sum + (row.credit ?? 0), 0) * 1000) / 1000;
    // A term printed across a column or a page break prints its header and its
    // banner again and keeps its rows coming; only the last of those blocks
    // carries the term's footer. Checking each block against that footer reported
    // a missing row on a term that was complete, so the blocks of one term are
    // checked together.
    const last = tableTotals[tableTotals.length - 1];
    if (
      last &&
      term !== null &&
      last.term === term &&
      (last.reportedCredits === null || table.footer === null)
    ) {
      last.rows += table.rows.length;
      last.credits = Math.round((last.credits + credits) * 1000) / 1000;
      last.reportedCredits = table.footer?.credits ?? last.reportedCredits;
      last.reportedGpa = table.footer?.gpa ?? last.reportedGpa;
      continue;
    }
    tableTotals.push({
      term,
      rows: table.rows.length,
      credits,
      reportedCredits: table.footer?.credits ?? null,
      reportedGpa: table.footer?.gpa ?? null,
    });
  }

  const credits = courses.map((row) => row.credit).filter((value): value is number => value !== null);
  const totalCredits = Math.round(credits.reduce((sum, value) => sum + value, 0) * 1000) / 1000;
  const layout: TranscriptLayout = numericBlocks.length ? 'course_mark_weight_credit' : courses.length ? 'dated_rows' : 'none';

  // One table prints one total, and that total is the document's. Several tables
  // each print their own term's credits: those are checked one by one, and no
  // document total is claimed from them, because the page never printed one.
  const single = numericRows.length === 1 ? numericRows[0].footer : null;
  const lastFooter = standing.reportedCredits === null && numericBlocks.length === 0
    ? [...all].reverse().map((line) => footerOf(line)).find((value) => value !== null) ?? null
    : null;
  const reportedCredits = standing.reportedCredits ?? single?.credits ?? lastFooter?.credits ?? null;
  const reportedGpa = standing.reportedGpa ?? single?.gpa ?? lastFooter?.gpa ?? null;
  const creditMismatch =
    reportedCredits === null ? null : Math.abs(totalCredits - reportedCredits) >= 0.001;
  const tableMismatches = tableTotals.filter(
    (table) => table.reportedCredits !== null && Math.abs(table.credits - table.reportedCredits) >= 0.001,
  ).length;

  const checks: TranscriptReading['checks'] = {
    courseCount: courses.length,
    assessmentCount: assessments.length,
    layout,
    marksNumeric: courses.some((row) => row.markPercent !== null),
    computedTotalCredits: totalCredits,
    reportedTotalCredits: reportedCredits,
    creditSumMatchesReported:
      reportedCredits === null ? null : !creditMismatch,
    tables: tableTotals.length,
    tablesWhoseCreditsDisagree: tableMismatches,
    recordsWithoutMark: courses.filter((row) => row.mark === null).length,
    recordsWithoutTerm: courses.filter((row) => row.term === null).length,
    recordsWithoutSection: courses.filter((row) => row.section === null).length,
    columnInterleavingDetected: seam !== null,
  };

  // A table header or a dated row is the evidence that this is a transcript, and
  // rows or a student name confirm it. A syllabus matches neither shape.
  const detected =
    (numericBlocks.length > 0 || courses.length > 0) &&
    (courses.length >= 2 || header.student !== null) &&
    layout !== 'none';

  return {
    detected,
    layout,
    student: header.student,
    school: header.school,
    documentTitle: header.documentTitle,
    terms,
    courses,
    assessments,
    totalCredits,
    reportedCredits,
    reportedGpa,
    tableTotals,
    checks,
  };
}

/**
 * The record's transcript block, plus one evidence row per course.
 *
 * Each course row points at the printed line it came from, so the review screen
 * can quote it and the committed course keeps a way back to the page. Ids are
 * derived from the document and the row's position, so re-reading the same page
 * proposes the same rows.
 */
export function transcriptEvidence(
  reading: TranscriptReading,
  docId: string,
): { courses: Evidence[] } {
  const courses: Evidence[] = reading.courses.map((row, index) => {
    const start = row.span.start;
    const end = row.span.end;
    return {
      id: stableId('evidence', docId, ['transcript_course', index, start, end]),
      field: 'transcript_course',
      value: row.title,
      confidence: row.confidence,
      confidenceSource: 'rule',
      band: null,
      sourceText: row.span.text,
      start,
      end,
      groundedInSource: true,
      requiresConfirmation: true,
    };
  });
  return { courses };
}

/** The record's transcript block: rows with ids and the evidence behind each. */
export function buildTranscriptRecord(
  reading: TranscriptReading,
  docId: string,
  evidence: Evidence[],
): TranscriptRecord {
  // The same term banner is often printed twice on a page (a header and then
  // again above the table). They name one term, so they are one entry.
  const terms: TranscriptRecord['terms'] = [];
  for (const banner of reading.terms) {
    const existing = terms.find((term) => term.value === banner.value);
    if (existing) {
      existing.courses += banner.courses;
      existing.credits = Math.round((existing.credits + banner.credits) * 1000) / 1000;
    } else {
      terms.push({ value: banner.value, courses: banner.courses, credits: banner.credits });
    }
  }
  return {
    layout: reading.layout === 'course_mark_weight_credit' ? 'course_mark_weight_credit' : 'dated_rows',
    student: reading.student,
    school: reading.school,
    documentTitle: reading.documentTitle,
    terms,
    courses: reading.courses.map((row, index) => ({
      id: transcriptRowId(docId, index, row),
      term: row.term,
      code: row.code,
      title: row.title,
      mark: row.mark,
      markPercent: row.markPercent,
      credit: row.credit,
      section: row.section,
      confidence: row.confidence,
      evidenceId: evidence[index]?.id ?? '',
      requiresConfirmation: true as const,
    })),
    assessments: reading.assessments.map((row) => ({
      title: row.title,
      score: row.score,
      level: row.level,
      label: row.label,
    })),
    totalCredits: reading.totalCredits,
    reportedCredits: reading.reportedCredits,
    reportedGpa: reading.reportedGpa,
    tableTotals: reading.tableTotals,
    checks: reading.checks,
    requiresConfirmation: true,
  };
}

/** The stable id of one transcript row, used by the committed course. */
export function transcriptRowId(docId: string, index: number, row: TranscriptCourseRow): string {
  return stableId('entity', docId, ['transcript_course', index, row.span.start, row.title]);
}

