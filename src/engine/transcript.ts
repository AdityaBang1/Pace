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
import type { Evidence, TranscriptRecord } from './types';
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
const CREDIT_CELL = /^\d+(?:\.\d{1,3})?$/;
const SCORE_CELL = /^\d{2,4}$/;
const NUMERIC_CELL = /^\d+(?:\.\d{1,6})?$/;
const NUMERIC_TOKEN = /^\d+(?:\.\d{1,6})?$/;
const TERM_BANNER =
  /(?<![0-9])(\d{4}\s*-\s*\d{4})\s+Grade\s+(\d{1,2})\s+Term\s+(\d{1,2})/i;
const FOOTER_TOTALS = /(?<![A-Za-z])Credits?\s*:\s*(\d+(?:\.\d{1,4})?)/i;
const FOOTER_GPA = /(?<![A-Za-z])GPA\s*:\s*(\d+(?:\.\d{1,6})?)/i;
const COURSE_CODE_PREFIX = /^(\d{3,6}[A-Za-z]{1,2})\s+/;
const PERFORMANCE_LEVEL =
  /Performance\s+Level\s*:?[ \t]*(\d+)[ \t]+([A-Za-z][A-Za-z]*(?:[ \t]+[A-Za-z]+)*)/i;
const LABELLED_LINE = /[A-Za-z]\s*:\s+\S/;

const NUMERIC_COLUMN_NAMES: Record<string, 'mark' | 'weight' | 'credit'> = {
  MARK: 'mark',
  GRADE: 'mark',
  WEIGHT: 'weight',
  CREDIT: 'credit',
  CREDITS: 'credit',
  UNITS: 'credit',
};

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
const STUDENT_NAME = /^[A-Z][A-Za-z'’\-]+,\s*[A-Z][A-Za-z'’\-]*$/;

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

function numericColumns(list: Cell[]): ('mark' | 'weight' | 'credit')[] | null {
  const names = list.map((cell) => cell.text.toUpperCase());
  if (names.includes('DATE')) return null;
  if (!names.includes('MARK') || !names.includes('CREDIT')) return null;
  if (!names.some((name) => name === 'COURSE' || name === 'COURSES' || name === 'TITLE')) return null;
  const first = names.findIndex((name) => name in NUMERIC_COLUMN_NAMES);
  if (first <= 0) return null;
  if (names.slice(first).some((name) => !(name in NUMERIC_COLUMN_NAMES))) return null;
  const columns = names.slice(first).map((name) => NUMERIC_COLUMN_NAMES[name]);
  if (new Set(columns).size !== columns.length) return null;
  return columns;
}

function headerColumns(list: Cell[], body: string): ('mark' | 'weight' | 'credit')[] | null {
  const direct = numericColumns(list);
  if (direct) return direct;
  if (body.split(/\s+/).filter(Boolean).length < 3) return null;
  const tokens: Cell[] = [];
  const tokenRe = /\S+/g;
  for (let match = tokenRe.exec(body); match; match = tokenRe.exec(body)) {
    tokens.push({ text: match[0], start: match.index, end: match.index + match[0].length });
  }
  return numericColumns(tokens);
}

function numericCell(
  name: 'mark' | 'weight' | 'credit',
  cell: Cell,
): { text: string; value: number; start: number; end: number } | null {
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
  columns: ('mark' | 'weight' | 'credit')[],
): { title: Cell[]; values: Record<string, { text: string; value: number; start: number; end: number }> } | null {
  let index = list.length;
  const values: Record<string, { text: string; value: number; start: number; end: number }> = {};
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
  columns: ('mark' | 'weight' | 'credit')[],
): { titleStart: number; titleEnd: number; values: Record<string, { text: string; value: number; start: number; end: number }> } | null {
  // The columns are read from the right edge of the line: a row is a title
  // followed by one number per named column, in header order. Reading right to
  // left also keeps each number's offset, which a group-index match would only
  // give with the `d` flag (not available on every engine).
  const values: Record<string, { text: string; value: number; start: number; end: number }> = {};
  let end = body.trimEnd().length;
  for (const name of [...columns].reverse()) {
    const piece = /^(.*?)(\d+(?:\.\d{1,6})?)$/.exec(body.slice(0, end));
    if (!piece) return null;
    const text = piece[2];
    const start = piece[1].length;
    // The number must be its own token, not the tail of a word or another number.
    if (start > 0 && !/\s/.test(body[start - 1])) return null;
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

/** Read the term banners a transcript prints above its rows. */
export function termBanners(text: string, all: Line[]): TranscriptTermRow[] {
  const out: TranscriptTermRow[] = [];
  for (const line of all) {
    const match = TERM_BANNER.exec(line.text);
    if (!match) continue;
    out.push({
      value: `${match[1].replace(/\s*-\s*/g, '-')} Grade ${match[2]} Term ${match[3]}`,
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
  columns: ('mark' | 'weight' | 'credit')[],
): TranscriptCourseRow[] {
  const out: TranscriptCourseRow[] = [];
  for (let lineIndex = headerIndex + 1; lineIndex < endIndex; lineIndex += 1) {
    const body = all[lineIndex].text;
    const list = allCells[lineIndex];
    if (!body.trim() || !list.length) continue;
    if (TERM_BANNER.test(body)) continue;
    const lineStart = all[lineIndex].start;

    let titleStart: number;
    let titleEnd: number;
    let values: Record<string, { text: string; value: number; start: number; end: number }>;
    const grid = numericRow(list, columns);
    if (grid) {
      titleStart = lineStart + grid.title[0].start;
      titleEnd = lineStart + grid.title[grid.title.length - 1].end;
      values = grid.values;
    } else {
      const loose = looseNumericRow(body, columns);
      if (!loose) continue;
      titleStart = lineStart + loose.titleStart;
      titleEnd = lineStart + loose.titleEnd;
      values = loose.values;
    }

    let code: string | null = null;
    const prefix = COURSE_CODE_PREFIX.exec(text.slice(titleStart, titleEnd));
    if (prefix) {
      code = prefix[1];
      titleStart += prefix[0].length;
    }
    if (titleStart >= titleEnd) continue;
    const rowEnd = lineStart + (list[list.length - 1].end);

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
      if (looseNumericRow(followingBody, columns) !== null) break;
      const extra = followingBody.slice(following[0].start, following[following.length - 1].end);
      title = `${title} ${extra.replace(/\s+/g, ' ').trim()}`.trim();
      end = all[cursor].start + following[following.length - 1].end;
      cursor += 1;
      wrapped += 1;
    }

    const markValue = values.mark.value;
    out.push({
      code,
      title,
      mark: values.mark.text,
      markPercent: markValue === Math.trunc(markValue) ? Math.trunc(markValue) : markValue,
      weight: values.weight ? values.weight.value : null,
      credit: values.credit.value,
      term: null,
      section: null,
      span: spanOf(text, titleStart, end),
      confidence: wrapped ? 0.9 : 0.95,
    });
  }
  return out;
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
    if (/\b(?:Sex|DOB|SSN|Entry Date|Exit Date|Graduation)\b/i.test(value)) {
      const name = value.split(/\s{2,}|\t/)[0].trim();
      if (!student && STUDENT_NAME.test(name)) student = name;
      continue;
    }
    if (!student && STUDENT_NAME.test(value)) {
      student = value;
      continue;
    }
    schoolLines.push(value);
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
  // The first remaining line is the institution; the rest is its address.
  const school = keeps.length ? keeps[0] : null;
  return { student, school, documentTitle };
}

function readStanding(
  text: string,
  all: Line[],
): { reportedCredits: number | null; reportedGpa: number | null; asOf: string | null } {
  let reportedCredits: number | null = null;
  let reportedGpa: number | null = null;
  let asOf: string | null = null;
  const heading = all.findIndex((line) => line.text.trim().toUpperCase() === 'ACADEMIC STANDING');
  if (heading === -1) {
    const credits = FOOTER_TOTALS.exec(text);
    const gpa = FOOTER_GPA.exec(text);
    reportedCredits = credits ? Number(credits[1]) : null;
    reportedGpa = gpa ? Number(gpa[1]) : null;
    return { reportedCredits, reportedGpa, asOf };
  }
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

export function readTranscript(text: string): TranscriptReading {
  const all = lines(text);
  const allCells = all.map((line) => (line.text.trim() ? cells(line.text) : []));
  const seam = columnSeam(allCells);
  const terms = termBanners(text, all);

  // Every numeric block, not just the first: a transcript prints one per term or
  // per page, and reading only the first would silently drop every later page.
  const numericBlocks: { header: number; end: number; columns: ('mark' | 'weight' | 'credit')[] }[] = [];
  let probe = 0;
  while (probe < all.length) {
    const columns = headerColumns(allCells[probe], all[probe].text);
    if (!columns) {
      probe += 1;
      continue;
    }
    let end = all.length;
    for (let cursor = probe + 1; cursor < all.length; cursor += 1) {
      const body = all[cursor].text;
      if (FOOTER_TOTALS.test(body) || FOOTER_GPA.test(body)) {
        end = cursor;
        break;
      }
      if (headerColumns(allCells[cursor], body) !== null) {
        end = cursor;
        break;
      }
    }
    numericBlocks.push({ header: probe, end, columns });
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

  for (const block of numericBlocks) {
    courses.push(...numericTable(text, all, allCells, block.header, block.end, block.columns));
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
  const standing = readStanding(text, all);

  const credits = courses.map((row) => row.credit).filter((value): value is number => value !== null);
  const totalCredits = Math.round(credits.reduce((sum, value) => sum + value, 0) * 1000) / 1000;
  const layout: TranscriptLayout = numericBlocks.length ? 'course_mark_weight_credit' : courses.length ? 'dated_rows' : 'none';
  const creditMismatch =
    standing.reportedCredits === null ? null : Math.abs(totalCredits - standing.reportedCredits) >= 0.001;

  const checks: TranscriptReading['checks'] = {
    courseCount: courses.length,
    assessmentCount: assessments.length,
    layout,
    marksNumeric: courses.some((row) => row.markPercent !== null),
    computedTotalCredits: totalCredits,
    reportedTotalCredits: standing.reportedCredits,
    creditSumMatchesReported:
      standing.reportedCredits === null ? null : !creditMismatch,
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
    reportedCredits: standing.reportedCredits,
    reportedGpa: standing.reportedGpa,
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
    checks: reading.checks,
    requiresConfirmation: true,
  };
}

/** The stable id of one transcript row, used by the committed course. */
export function transcriptRowId(docId: string, index: number, row: TranscriptCourseRow): string {
  return stableId('entity', docId, ['transcript_course', index, row.span.start, row.title]);
}

