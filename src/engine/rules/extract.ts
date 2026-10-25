/**
 * The deterministic half of the reader.
 *
 * The model decides *what a section is*; these rules decide *what it says*, and
 * they do it with patterns rather than probabilities because "40%" and
 * "TU 11:00-12:15" are not classification problems. That split is the reason the
 * app can show a value with its exact source span: every value here is a slice of
 * the document.
 *
 * Ported from `pace.field_extract` (grading, credits, code, instructor, title,
 * meetings, policies). Where the Python reader is more elaborate than the app
 * needs, the comment says so instead of pretending the port is complete.
 */

import type { DetectedSection, RuleTables } from './sections';
import { policyBody, sectionBody } from './sections';
import {
  DAY_TOKEN,
  MEETING_LINE,
  dayGroups,
  dayWords,
  daysIn,
  EXAM_LINE,
  fold,
  hasClock,
  lines,
  normSpace,
  OFFICE_HOURS_LINE,
  openingSentence,
  parseClock,
  roomFrom,
  cleanCandidate,
} from './text';

export interface Value<T = string | number> {
  value: T;
  confidence: number;
  rule: string;
  evidence: { text: string; start: number; end: number };
}

// The gap between the category and its percentage may be a colon, a dash, or the
// dot leaders a printed grade table uses ("Essays ...... 30%"). Without the dot
// run, the commonest layout in a photographed syllabus read as zero categories.
const WEIGHT_FIRST =
  /^\s*[-\u2022*]?\s*(?<cat>[A-Za-z][A-Za-z0-9 &/+'-]{1,40}?)(?:\s*\([^)%]{0,40}\))?[\s.:\-\u2013]*\(?(?<w>\d{1,3})\s*%/;
const PERCENT_FIRST =
  /^\s*[-\u2022*]?\s*(?<w>\d{1,3})\s*%\s*[:\-\u2013]?\s*(?<cat>[A-Za-z][A-Za-z &/+'-]{1,40})$/;
const PERCENT = /(?<w>\d{1,3}(?:\.\d{1,2})?)\s*%/g;
const BARE_PERCENT = /^\d{1,3}(?:\.\d{1,2})?\s*%$/;
const PAIR_BREAK = /[\u2022\u00b7,;|+]|\band\b|&/i;
const LEADING_NOISE = /^[\s\d.)\-(•·]+/;

const CATEGORY_STOP = new Set([
  'total', 'points', 'grade', 'grades', 'grading', 'percent', 'percentage',
  'weight', 'weighting', 'score', 'score total',
]);
const BAD_TAIL = new Set([
  'is', 'are', 'be', 'was', 'were', 'counts', 'count', 'worth', 'equals', 'and', 'or',
  'the', 'a', 'an', 'of', 'on', 'in', 'to', 'for', 'with', 'this', 'that', 'will',
  'should', 'must', 'as', 'if', 'when', 'after', 'before', 'up', 'at', 'by', 'from',
]);
const WEIGHT_STOP = new Set([
  'the', 'of', 'at', 'least', 'more', 'than', 'those', 'who', 'will', 'result',
  'accumulation', 'accumulate', 'possible', 'total', 'points', 'percent', 'grade',
  'grades', 'grading', 'earn', 'earned', 'must', 'should', 'this', 'week', 'each',
  'and', 'or', 'for',
]);
const CATEGORY_LEAD_STOP = new Set([
  'with', 'and', 'or', 'for', 'of', 'to', 'in', 'on', 'at', 'by', 'from', 'that',
  'this', 'these', 'those', 'your', 'our', 'their', 'is', 'are', 'was', 'were',
  'will', 'can', 'should', 'must',
]);
const QTY_PREFIX = /^(?:the|a|an|each|every|one|two|three|four|five|both|all|your|my|our)\s+/i;

export function categoryOk(raw: string): boolean {
  const text = cleanCandidate(raw).toLowerCase();
  const words = text.split(/\s+/).filter(Boolean);
  const letters = (text.match(/[A-Za-z]/g) ?? []).length;
  if (text.length < 3 || text.length > 40 || letters < 3) return false;
  if (CATEGORY_STOP.has(text) || words.length > 4 || words.length === 0) return false;
  if (CATEGORY_LEAD_STOP.has(words[0])) return false;
  if (BAD_TAIL.has(words[words.length - 1])) return false;
  return !words.some((word) => WEIGHT_STOP.has(word));
}

function stripCategoryPrefix(raw: string): string {
  let value = cleanCandidate(raw);
  const lower = value.toLowerCase();
  let tail = '';
  if (lower.startsWith('of ')) tail = value.slice(3).trim();
  else if (lower.includes(' of ')) tail = value.slice(lower.lastIndexOf(' of ') + 4).trim();
  if (tail && categoryOk(tail)) value = tail;
  for (let round = 0; round < 3; round += 1) {
    const stripped = value.replace(QTY_PREFIX, '').trim();
    if (stripped === value) break;
    value = stripped;
  }
  return value;
}

function inParens(line: string, position: number): boolean {
  const opens = line.slice(0, position).split('(').length - 1;
  const closes = line.slice(0, position).split(')').length - 1;
  return opens > closes;
}

function componentParen(line: string, position: number): boolean {
  if (!inParens(line, position)) return false;
  const opened = line.lastIndexOf('(', position);
  const closed = line.indexOf(')', position);
  const group = line.slice(opened, closed === -1 ? line.length : closed);
  if ((group.match(PERCENT) ?? []).length >= 2) return true;
  for (const match of line.matchAll(PERCENT)) {
    if ((match.index ?? 0) + match[0].length <= opened) return true;
  }
  return false;
}

function parentheticalComponents(line: string): { category: string; weight: number }[] {
  const parts: { category: string; weight: number }[] = [];
  for (const match of line.matchAll(PERCENT)) {
    const position = match.index ?? 0;
    if (!componentParen(line, position)) continue;
    const opened = line.lastIndexOf('(', position);
    const segments = line.slice(opened + 1, position).split(PAIR_BREAK);
    const category = cleanCandidate(segments[segments.length - 1]);
    if (category && categoryOk(category)) {
      parts.push({ category, weight: Math.trunc(Number(match.groups?.w)) });
    }
  }
  return parts;
}

function subweightLine(line: string): boolean {
  for (const group of line.match(/\([^()]*\)/g) ?? []) {
    if ((group.match(PERCENT) ?? []).length >= 2) return true;
  }
  return false;
}

function sentencePairs(line: string): { category: string; weight: number }[] {
  const matches = Array.from(line.matchAll(PERCENT));
  if (matches.length < 2) return [];
  const pairs: { category: string; weight: number }[] = [];
  for (const match of matches) {
    const position = match.index ?? 0;
    if (componentParen(line, position)) continue;
    const before = line.slice(0, position);
    if (/\bof\s+(?:your|the)\s+(?:course\s+|final\s+|overall\s+|semester\s+)?grade\s*$/i.test(before)) {
      continue;
    }
    const leftSegments = before.split(PAIR_BREAK);
    let left = (leftSegments[leftSegments.length - 1] ?? '')
      .replace(/\([^)]*$/, ' ')
      .replace(LEADING_NOISE, '')
      .replace(/^[\s:\-–(]+|[\s:\-–(]+$/g, '');
    const rightSegments = line.slice(position + match[0].length).split(PAIR_BREAK);
    let right = (rightSegments[0] ?? '')
      .replace(/\([^)]*\)/g, ' ')
      .replace(/^[\s:\-–.]+|[\s:\-–.]+$/g, '');
    const weight = Math.trunc(Number(match.groups?.w));
    const candidates = [left, right];
    for (const candidate of candidates) {
      const cleaned = cleanCandidate(candidate);
      if (cleaned.split(/\s+/).length <= 4 && categoryOk(cleaned)) {
        pairs.push({ category: cleaned, weight });
        break;
      }
    }
    left = right = '';
  }
  return pairs;
}

export interface GradingRow extends Value<number> {
  category: string;
  /** The percentage. Same number as `value`, named for readability at call sites. */
  weight: number;
  weightsSum?: number;
  weightsSumConsistent?: boolean;
}

export function extractGrading(text: string): GradingRow[] {
  const found: GradingRow[] = [];
  const components = new Map<string, string>();
  let previous: { text: string; start: number; end: number } | null = null;

  for (const line of lines(text)) {
    const stripped = line.text.trim();
    if (!stripped) continue;
    if (stripped.length > 300) {
      previous = line;
      continue;
    }
    if (BARE_PERCENT.test(stripped) && previous) {
      const category = cleanCandidate(previous.text.replace(/:$/, ''));
      const weight = Math.trunc(Number((stripped.match(PERCENT) ?? ['0'])[0].replace('%', '').trim()));
      if (categoryOk(category) && category.split(/\s+/).length <= 4 && weight >= 1 && weight <= 100) {
        found.push({
          category,
          weight,
          value: weight,
          confidence: 0.82,
          rule: 'weight_on_next_line',
          evidence: {
            text: `${previous.text.slice(0, 80)} ${stripped}`,
            start: previous.start,
            end: line.end,
          },
        });
      }
    }
    if (!stripped.includes('%')) {
      previous = line;
      continue;
    }
    const pairs: { category: string; weight: number }[] = [];
    let top: { category: string; weight: number } | null = null;
    const weightFirst = WEIGHT_FIRST.exec(stripped);
    const percentFirst = weightFirst ? null : PERCENT_FIRST.exec(stripped);
    const match = weightFirst ?? percentFirst;
    if (match?.groups) {
      top = {
        category: stripCategoryPrefix(match.groups.cat),
        weight: Math.trunc(Number(match.groups.w)),
      };
      pairs.push(top);
    }
    if (top) {
      for (const component of parentheticalComponents(stripped)) {
        if (component.category.toLowerCase() !== top.category.toLowerCase()) {
          components.set(`${component.category.toLowerCase()}|${component.weight}`, top.category);
        }
      }
    }
    let sentence = subweightLine(stripped) ? [] : sentencePairs(stripped);
    if (top && sentence.length && sentence.reduce((sum, pair) => sum + pair.weight, 0) === top.weight) {
      sentence = [];
    }
    pairs.push(...sentence);
    if (!pairs.length) {
      previous = line;
      continue;
    }
    let confidence = 0.9;
    if (/\b(?:is|are|be|will|should|must|counts?|based on)\b/i.test(stripped)) confidence = 0.8;
    if (pairs.length >= 6) confidence = Math.min(confidence, 0.8);
    for (const pair of pairs) {
      if (!categoryOk(pair.category)) continue;
      if (pair.weight < 1 || pair.weight > 100) continue;
      const key = `${pair.category.toLowerCase()}|${pair.weight}`;
      if (components.has(key)) continue;
      found.push({
        category: pair.category,
        weight: pair.weight,
        value: pair.weight,
        confidence,
        rule: 'grading_line',
        evidence: { text: stripped.slice(0, 160), start: line.start, end: line.end },
      });
    }
    previous = line;
  }

  const deduped = new Map<string, GradingRow>();
  for (const item of found) {
    if (components.has(`${item.category.toLowerCase()}|${item.weight}`)) continue;
    const key = item.category.toLowerCase();
    if (deduped.has(key)) continue;
    deduped.set(key, item);
  }
  let kept = Array.from(deduped.values());
  if (kept.length < 2) return [];
  kept = subsetSummingTo100(kept);
  if (!kept.length) return [];
  const total = kept.reduce((sum, item) => sum + item.weight, 0);
  const consistent = Math.abs(total - 100) <= 5;
  for (const item of kept) {
    if (consistent) item.confidence = Math.min(0.97, Number((item.confidence + 0.05).toFixed(4)));
    item.weightsSum = total;
    item.weightsSumConsistent = consistent;
  }
  return kept;
}

function subsetSummingTo100(items: GradingRow[]): GradingRow[] {
  const total = items.reduce((sum, item) => sum + item.weight, 0);
  if (Math.abs(total - 100) <= 5) return items;
  const count = items.length;
  if (count > 12) return [];
  let best: GradingRow[] = [];
  let bestKey: [number, number] | null = null;
  for (let mask = 1; mask < 1 << count; mask += 1) {
    const subset = items.filter((_, index) => mask & (1 << index));
    if (subset.length < 2) continue;
    const weight = subset.reduce((sum, item) => sum + item.weight, 0);
    if (Math.abs(weight - 100) > 5) continue;
    const key: [number, number] = [subset.length, -Math.abs(weight - 100)];
    if (!bestKey || key[0] > bestKey[0] || (key[0] === bestKey[0] && key[1] > bestKey[1])) {
      bestKey = key;
      best = subset;
    }
  }
  if (!best.length && total < 95) return items;
  return best;
}

/* ------------------------------------------------------------------ meetings */

export interface MeetingValue {
  days: string[];
  startTime: string | null;
  endTime: string | null;
  room: string | null;
  alternatives: boolean;
  confidence: number;
  evidence: { text: string; start: number; end: number };
}

export function extractMeetings(text: string): MeetingValue[] {
  const extracted: MeetingValue[] = [];
  const seen = new Set<string>();
  for (const line of lines(text)) {
    const probe = fold(line.text);
    if (OFFICE_HOURS_LINE.test(probe)) continue;
    if (EXAM_LINE.test(probe)) continue;
    if (dayGroups(probe) > 1) continue;
    const days = daysIn(probe);
    if (!days.length) continue;
    const listed = dayWords(probe);
    if (listed.length && listed.some((day) => !days.includes(day))) continue;
    const ranges = Array.from(probe.matchAll(MEETING_LINE));
    const room = roomFrom(probe, ranges.length ? (ranges[ranges.length - 1].index ?? 0) + ranges[ranges.length - 1][0].length : 0);
    const bareToken = DAY_TOKEN.exec(probe);
    if (!ranges.length) {
      if (!room) continue;
      const key = `${days.join(',')}||${room}`;
      if (seen.has(key)) continue;
      seen.add(key);
      extracted.push({
        days,
        startTime: null,
        endTime: null,
        room,
        alternatives: false,
        confidence: 0.82,
        evidence: { text: line.text.trim().slice(0, 160), start: line.start, end: line.end },
      });
      continue;
    }
    const confidence = room ? 0.95 : 0.9;
    for (const range of ranges) {
      const raw = range[0];
      const startRaw = range.groups?.start ?? '';
      const endRaw = range.groups?.end ?? '';
      const bareHours =
        Boolean(bareToken) &&
        [startRaw, endRaw].every((part) => /^\d{1,2}\s*(?:[ap]\.?m\.?)?$/i.test(part.trim()));
      if (!raw.includes(':') && !/[ap]\.?m\.?/i.test(raw) && !bareHours) continue;
      const startTime = parseClock(startRaw);
      const endTime = parseClock(endRaw);
      if (!startTime || !endTime) continue;
      const key = `${days.join(',')}|${startTime}|${endTime}|${room ?? ''}`;
      if (seen.has(key)) continue;
      seen.add(key);
      extracted.push({
        days,
        startTime,
        endTime,
        room,
        alternatives: ranges.length > 1,
        confidence,
        evidence: { text: line.text.trim().slice(0, 160), start: line.start, end: line.end },
      });
    }
  }
  return extracted;
}

/* ------------------------------------------------------------- course fields */

const CREDITS =
  /\b(?<n>\d{1,2}(?:\.\d{1,2})?)\s*(?:semester\s+|academic\s+)?(?:units?|credits?|credit\s+hours?|semester\s+hours?)\b/i;
const CREDITS_INLINE = /\b(?:credits?|units?)\s*[:\-]\s*(?<n>\d{1,2}(?:\.\d{1,2})?)\b/i;
const CREDIT_CAVEAT = /\b(prereq\w*|requirement|minimum|up to|at least|max(?:imum)?|elective|repeat)\b/i;

export function extractCredits(text: string): Value<number> | null {
  const candidates: Value<number>[] = [];
  for (const line of lines(text)) {
    for (const pattern of [CREDITS, CREDITS_INLINE]) {
      const match = pattern.exec(line.text);
      if (!match?.groups?.n) continue;
      const number = Number(match.groups.n);
      if (!(number > 0 && number <= 30)) continue;
      const stripped = line.text.trim();
      let confidence = 0.95;
      if (CREDIT_CAVEAT.test(stripped)) confidence = 0.7;
      else if (stripped.length > 90) confidence = 0.85;
      candidates.push({
        value: number,
        confidence,
        rule: 'credit_pattern',
        evidence: { text: match[0], start: line.start, end: line.end },
      });
    }
  }
  if (!candidates.length) return null;
  return candidates.reduce((best, item) => (item.confidence > best.confidence ? item : best));
}

const COURSE_CODE = /\b(?<dept>[A-Z]{2,6})[\s-]?(?<num>\d{1,4}[A-Z]?(?:-\d{1,3})?)\b/g;
const CODE_STOP = new Set([
  'ROOM', 'PAGE', 'FALL', 'SPRING', 'SUMMER', 'WINTER', 'V', 'WWW', 'HTTP', 'HTTPS',
  'PDF', 'HTML', 'TBA', 'TBD', 'N', 'A', 'ISBN', 'ISSN', 'DOI', 'TEL', 'FAX', 'PHONE',
  'EMAIL', 'ZIP', 'SKYPE', 'OFFICE', 'UNIT',
]);
const SESSION_CODE = /^(?:SA|FA|SP|SU|WI|WS|SS|S)[A-Z]?\d{2}$/i;
const LOCATION_CUE = /\b(office|room|rm\.?|bldg|building|hall|lab|center|university)\b/i;
const LABELLED_CODE = /\b(?:course\s+(?:code|number|no\.?)|catalog\s+(?:number|no\.?)|course\s*#)\s*[:#]/i;

export function extractCourseCode(text: string): Value<string> | null {
  const all = lines(text);
  const head = normSpace(all.slice(0, 6).map((line) => line.text).join(' '));
  const candidates: Value<string>[] = [];
  for (const line of all) {
    const stripped = line.text.trim();
    if (!stripped) continue;
    const labelled = LABELLED_CODE.test(stripped);
    if (line.start > 3000 && !labelled) continue;
    if (DAY_TOKEN.test(stripped)) continue;
    for (const match of stripped.matchAll(COURSE_CODE)) {
      const code = match[0].trim();
      if (CODE_STOP.has(match.groups?.dept ?? '')) continue;
      const number = match.groups?.num ?? '';
      if (/^(19|20)\d\d$/.test(number)) continue;
      if (SESSION_CODE.test(code.replace(/\s/g, ''))) continue;
      const after = stripped.slice((match.index ?? 0) + code.length, (match.index ?? 0) + code.length + 4);
      if (after.startsWith('.') && /\d/.test(after[1] ?? '')) continue;
      const before = stripped.slice(Math.max(0, (match.index ?? 0) - 14), match.index ?? 0).toLowerCase();
      if (/(room|rm\.?|office|bldg|building|hall|lab|in)\s*$/i.test(before)) continue;
      if (LOCATION_CUE.test(stripped) && !labelled) continue;
      let confidence = 0.5;
      if (labelled) confidence = 0.97;
      else if (head.includes(normSpace(code))) confidence = 0.93;
      else if (line.start < 400) confidence = 0.85;
      candidates.push({
        value: code,
        confidence,
        rule: 'code_pattern',
        evidence: { text: code, start: line.start, end: line.end },
      });
    }
  }
  if (!candidates.length) return null;
  return candidates.reduce((best, item) => (item.confidence > best.confidence ? item : best));
}

const INSTRUCTOR = /^\s*(?:instructors?|teachers?|taught by|professor)\s*[:\-]\s*(?<v>.+)$/i;
const EMAIL = /[\w.+-]+@[\w-]+\.[\w.]+/;

export function extractInstructor(text: string): Value<string> | null {
  for (const line of lines(text)) {
    const match = INSTRUCTOR.exec(line.text);
    if (!match?.groups?.v) continue;
    let value = normSpace(match.groups.v);
    const email = EMAIL.exec(value);
    // Python reads the labelled line at 0.95 and drops to 0.7 when the value
    // carries a digit or an address, because that is an office or an email and
    // not the teacher's name. It keeps the address in the value so the source
    // shown to the user is the line as printed.
    if (email) value = normSpace(value.replace(email[0], ''));
    value = value.replace(/[\s,;|]+$/, '');
    if (!value || value.split(/\s+/).length > 6) continue;
    const printed = normSpace(match.groups.v);
    return {
      value,
      confidence: /[\d@]/.test(printed) ? 0.7 : 0.95,
      rule: 'instructor_label',
      evidence: { text: value, start: line.start, end: line.end },
    };
  }
  return null;
}

const TITLE_TAIL = new Set(['of', 'and', 'the', 'for', 'to', 'in', 'with', 'a', 'an', 'or', 'on', 'at', 'by']);
const TITLE_DOCUMENT = /\bsyllabus\b|\bcourse\s+outline\b|\.(?:pdf|docx?|html?|txt|rtf)\b|_|\bpage\s+\d+(?:\s*(?:of|\/)\s*\d+)?\b|^\s*\d{1,3}\s*\/\s*\d{1,3}\s*$|[|\\]/i;
const TITLE_WEIGHT = /\d{1,3}(?:\.\d{1,2})?\s*%\s*$/;
const TITLE_ATTRIBUTE = /^([^:]{3,60}):\s*\S/;
const ATTRIBUTE_LABELS = new Set([
  'flag', 'flags', 'format', 'formats', 'credit', 'credits', 'prerequisite',
  'prerequisites', 'requirement', 'requirements', 'policy', 'policies', 'information',
  'info', 'hours', 'time', 'times', 'location', 'instructor', 'instructors', 'author',
  'authors', 'contact', 'website', 'url', 'link', 'email', 'office', 'description',
]);
const TITLE_SESSION = /\b(?:spring|summer|fall|autumn|winter|term|semester|section|period)\b\s*\d{0,4}|^\s*(?:spring|summer|fall|autumn|winter)\s+\d{4}/i;
const TITLE_CLOCK = /\b\d{1,2}:\d{2}\b/;
const TITLE_PLACE = /\b(?:room|hall|building|center|auditorium|campus)\b/i;
const TITLE_CONTACT = /@|https?:\/\/|www\./i;
const TITLE_PERSON = /^(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z]/;
const TITLE_DATE = /\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b|\b(?:19|20)\d{2}\b/i;
const GENERIC_TITLES = new Set([
  'introduction', 'syllabus', 'catalog description', 'course description', 'overview',
  'contents', 'resources', 'policies', 'welcome', 'description', 'general information',
  'course information', 'class information', 'course catalog',
]);
const GENERIC_HEADING =
  /^(?:general|catalog|course|class|key|basic|important|other|additional)\s+(?:information|details|description|overview|contents|requirements|policies|policy|schedule|materials|objectives|expectations|contact|staff|grading|resources|readings|calendar|hours|outline|topics)\s*$|\b(?:information|details|description|requirements|expectations|objectives|contact|policy|policies|schedule|materials|grading|assistants?|staff|readings|resources|calendar|hours|content)\s*$/i;
const SENTENCE_VERB = /\b(?:is|are|was|were|will|would|should|must|may|might|can|could|has|have|had|meets?|covers?|introduces?|provides?)\b/i;
const NAV_WORDS = /\b(menu|login|log in|sign in|navigation|breadcrumb|skip to|copyright|privacy|search|home page|user account|toggle|cookie|javascript|news center|campus calendar|sitemap|directory|all rights reserved)\b/i;

export function titlePlausible(text: string): boolean {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 2 || words.length > 10) return false;
  if (TITLE_TAIL.has(words[words.length - 1].toLowerCase())) return false;
  if (TITLE_DOCUMENT.test(text)) return false;
  if (text.split(' - ').length - 1 >= 2) return false;
  return !SENTENCE_VERB.test(text);
}

/**
 * The course name: the earliest header line that carries a catalogue code, or
 * else the earliest title-shaped header line.
 *
 * This is a reduced port of `extract_course_name`. The Python reader scores
 * candidates with the section model and corroborates them by repetition; that
 * picker exists to survive badge-heavy university pages. The app's first slice
 * reads the header block, which is what a photographed syllabus has, and the
 * value is always a proposal. `docs/MOBILE.md` records the difference.
 */
export function extractCourseTitle(text: string, sections: DetectedSection[]): Value<string> | null {
  const header = text.slice(0, 800);
  const headings = new Set(sections.map((section) => section.text.toLowerCase()));
  let fallback: { value: string; start: number; end: number } | null = null;
  for (const line of lines(header)) {
    const candidate = cleanCandidate(line.text);
    if (candidate.length < 4 || candidate.length > 90) continue;
    if (GENERIC_TITLES.has(candidate.toLowerCase().replace(/:$/, ''))) continue;
    if (TITLE_DOCUMENT.test(candidate) || TITLE_WEIGHT.test(candidate)) continue;
    const attribute = TITLE_ATTRIBUTE.exec(candidate);
    if (attribute) {
      const words = attribute[1].trim().split(/\s+/);
      if (ATTRIBUTE_LABELS.has(words[words.length - 1].toLowerCase())) continue;
    }
    if (TITLE_CLOCK.test(candidate) || TITLE_PLACE.test(candidate)) continue;
    if (TITLE_CONTACT.test(candidate) || TITLE_PERSON.test(candidate)) continue;
    if (TITLE_DATE.test(candidate) || TITLE_SESSION.test(candidate)) continue;
    if (GENERIC_HEADING.test(candidate) || headings.has(candidate.toLowerCase())) continue;
    if (NAV_WORDS.test(candidate)) continue;
    if (!/^[A-Z]/.test(candidate) || !/[A-Za-z]{3,}/.test(candidate)) continue;
    if (!titlePlausible(candidate)) continue;
    const hasCode = /[A-Z]{2,6}[\s-]?\d{1,4}/.test(candidate);
    if (hasCode) {
      return {
        value: candidate,
        confidence: 0.86,
        rule: 'code_bearing_title',
        evidence: { text: candidate, start: line.start, end: line.end },
      };
    }
    if (!fallback) fallback = { value: candidate, start: line.start, end: line.end };
  }
  if (!fallback) return null;
  return {
    value: fallback.value,
    confidence: 0.7,
    rule: 'title_line',
    evidence: { text: fallback.value, start: fallback.start, end: fallback.end },
  };
}

/* ------------------------------------------------------------------ policies */

const SECTION_VALUES: Record<string, string[]> = {
  office_hours: ['office_hours'],
  attendance_policy: ['attendance_policy'],
  academic_integrity: ['academic_integrity'],
  disability_services: ['disability_services'],
  materials: ['materials'],
};

const PLACEHOLDER = /^[\s•·\-–—:.]*(?:tbd|tba|n\/?a|none|to be announced|to be determined)\b/i;
const POINTER = /(posted|canvas|e-?learning|website|web site|online|\bsee\b|e-?mail|schedul\w*|contact|appointment)/i;
const AVAILABILITY = /(?:\bappointment\b|\boffice hours?\b|\bstudent hours?\b|\bconsultation hours?\b|\b(?:drop|walk)[- ]?in\b|\bopen hours?\b|\b(?:stop|come) by\b|\bby e-?mail\b|\bschedul\w*)/i;
const SCHEDULE_HEADING = /^\s*(?:(?:class|lecture|laborator(?:y|ies)|lab|seminar|recitation|meeting)\s+(?:times?|schedules?|meetings?|hours?)|class\s+schedule)\b/i;
const VALUE_CUT = /\s+(?=(?:course dates|lecture times?|lecture place|class (?:time|room)|meeting (?:times?|place)|screening|course (?:code|number)|semester|term)\s*:)/i;

export interface PolicyValue {
  value: string | null;
  confidence: number;
  reason: string | null;
  evidence: { text: string; start: number; end: number };
}

export function policyCue(tables: RuleTables, field: string): RegExp | null {
  const pattern = tables.policy_cues[field];
  if (!pattern) return null;
  try {
    return new RegExp(pattern, 'i');
  } catch {
    return null;
  }
}

function namesAvailability(text: string): boolean {
  const opening = openingSentence(text).slice(0, 160);
  return AVAILABILITY.test(opening) || hasClock(opening) || DAY_TOKEN.test(opening) || /\b(?:mon|tue|wed|thu|fri)[a-z]*\b/i.test(opening);
}

function nextSectionLabel(text: string): string | null {
  const match = /^\s*([A-Z][A-Za-z &/.'-]{2,40}?)\s*:\s*\S/.exec(text);
  return match ? match[1].replace(/\s+/g, ' ').trim().toLowerCase() : null;
}

export function extractPolicies(
  text: string,
  sections: DetectedSection[],
  labels: string[],
  tables: RuleTables,
): Record<string, PolicyValue> {
  const out: Record<string, PolicyValue> = {};
  sections.forEach((section, index) => {
    const label = labels[index];
    const fields = SECTION_VALUES[label];
    if (!fields) return;
    const field = fields[0];
    const { body, inline } = policyBody(text, sections, section);
    const raw = normSpace(body);
    const valueText = raw.split(VALUE_CUT)[0].trim().replace(/[ .;]+$/, '');
    const evidence = { text: section.text, start: section.start, end: section.end };
    const reject = (reason: string) => {
      if (!(field in out)) out[field] = { value: null, confidence: 0, reason, evidence };
    };
    if (!valueText) return reject('empty body');
    if (PLACEHOLDER.test(valueText)) return reject("placeholder ('TBD'/'TBA'/'None')");
    if (field === 'office_hours' && SCHEDULE_HEADING.test(normSpace(section.text))) {
      return reject('a class-time heading, not office hours');
    }
    const lead = nextSectionLabel(valueText);
    if (lead && tables.section_map[lead] && tables.section_map[lead] !== field) {
      return reject(`body opens another field ('${lead}')`);
    }
    if (field === 'office_hours' && !namesAvailability(valueText)) {
      return reject('body opens without naming when the instructor is available');
    }
    const cue = policyCue(tables, field);
    if (cue && !cue.test(valueText)) return reject('body is not about the field');
    if (!inline && valueText.length < 40 && !hasClock(valueText)) {
      return reject('body too short to be a value');
    }
    let confidence = 0.8;
    if (field === 'office_hours') {
      if (hasClock(valueText)) confidence = inline ? 0.95 : 0.9;
      else if (DAY_TOKEN.test(valueText) || /\b(?:mon|tue|wed|thu|fri)[a-z]*\b/i.test(valueText)) confidence = 0.85;
      else if (POINTER.test(valueText)) confidence = 0.72;
      else confidence = 0.6;
    } else {
      confidence = inline ? 0.85 : 0.8;
    }
    out[field] = {
      value: valueText.slice(0, 300),
      confidence: Number(confidence.toFixed(4)),
      reason: null,
      evidence,
    };
  });
  return out;
}

export { sectionBody };
