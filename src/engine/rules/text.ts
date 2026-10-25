/**
 * The text primitives the reader shares with `pace.field_extract`.
 *
 * These are deliberately boring: offsets, folding, weekday tokens, clock times,
 * rooms. Every value the reader emits is a slice of the source text found by one
 * of these helpers, which is what makes "view source" in the app honest.
 */

export interface Line {
  text: string;
  start: number;
  end: number;
}

const HTML_TAG = /<[^>]{0,4000}>/g;
const HTML_ENTITY = /&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,10});/g;
const NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  ndash: '\u2013',
  mdash: '\u2014',
  rsquo: '\u2019',
  lsquo: '\u2018',
  ldquo: '\u201c',
  rdquo: '\u201d',
};

function decodeEntity(entity: string): string {
  if (entity.startsWith('&#')) {
    const hex = entity[2] === 'x' || entity[2] === 'X';
    const code = parseInt(entity.slice(hex ? 3 : 2, -1), hex ? 16 : 10);
    if (Number.isFinite(code) && code > 0) return String.fromCodePoint(code);
    return ' ';
  }
  const name = entity.slice(1, -1).toLowerCase();
  return NAMED_ENTITIES[name] ?? ' ';
}

/**
 * `pace.field_corpus.clean_html`: tags become a space, entities their character.
 * The offset map is not returned because nothing in the app maps back into raw
 * HTML; OCR text has no tags at all.
 */
export function cleanHtml(text: string): string {
  const tagged = text.replace(HTML_TAG, ' ');
  return tagged.replace(HTML_ENTITY, (entity) => decodeEntity(entity));
}

/** Python's `splitlines(keepends=True)` over `\n`-separated text, with offsets. */
export function lines(text: string): Line[] {
  const out: Line[] = [];
  let offset = 0;
  for (const raw of text.split('\n')) {
    const withoutReturn = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
    out.push({ text: withoutReturn, start: offset, end: offset + withoutReturn.length });
    offset += raw.length + 1;
  }
  return out;
}

/** `_clean_candidate`: collapse whitespace, trim punctuation-ish edges. */
export function cleanCandidate(value: string): string {
  // The hyphen must be escaped: inside a class, `;-\u2013` is a range from ';'
  // to en-dash, which silently eats every letter after the first punctuation.
  return value.replace(/\s+/g, ' ').replace(/^[\s.,:;\-\u2013]+|[\s.,:;\-\u2013]+$/g, '');
}

export function normSpace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

const LIGATURES: Record<string, string> = {
  '\ufb00': 'ff',
  '\ufb01': 'fi',
  '\ufb02': 'fl',
  '\ufb03': 'ffi',
  '\ufb04': 'ffl',
  '\ufb05': 'ft',
  '\u00a0': ' ',
  '\u2009': ' ',
  '\u202f': ' ',
};

/** `_fold`: ligatures and non-breaking spaces out, case preserved. */
export function fold(text: string): string {
  return text
    .replace(/[\ufb00-\ufb05\u00a0\u2009\u202f]/g, (char) => LIGATURES[char] ?? char)
    .replace(/\u2019/g, "'");
}

export const DAY_ORDER = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU'];

const SINGLE_DAY: Record<string, string> = {
  M: 'MO',
  T: 'TU',
  W: 'WE',
  R: 'TH',
  F: 'FR',
  S: 'SA',
};

const DAY_WORDS: Record<string, string> = {
  mon: 'MO',
  tue: 'TU',
  wed: 'WE',
  thu: 'TH',
  fri: 'FR',
  sat: 'SA',
  sun: 'SU',
};

const DAY_PAIR = '(?:MO|TU|WE|TH|FR|SA|SU)(?:[ ,/&]+(?:MO|TU|WE|TH|FR|SA|SU))+';
export const DAY_TOKEN = new RegExp(
  `(?<![A-Za-z])(?<days>${DAY_PAIR}|MTWThF|MTWTh|MWThF|MWTh|MWF|MThF|TuTh|TTh|TR|MW|WF|MTW|WTh|M/W/F|M/W|T/R)(?![A-Za-z])`,
  'i',
);

const DAY_WORD = /\b(?:Mon(?:day)?s?|Tues?(?:day)?s?|Wed(?:nesday)?s?|Thu(?:rs?(?:day)?)?s?|Fri(?:day)?s?|Sat(?:urday)?s?|Sun(?:day)?s?)\b/gi;

export function daysFromToken(token: string): string[] {
  const separated = token
    .trim()
    .split(/[ ,/&]+/)
    .filter(Boolean);
  if (separated.length > 1 && separated.every((part) => part.toUpperCase() in DAY_WORDS && part.length <= 3)) {
    return Array.from(new Set(separated.map((part) => part.toUpperCase())));
  }
  const cleaned = token.replace(/\//g, '');
  const lower = cleaned.toLowerCase();
  if (lower === 'tuth' || lower === 'tth' || lower === 'tr') return ['TU', 'TH'];
  const days: string[] = [];
  let index = 0;
  while (index < cleaned.length) {
    if (cleaned.slice(index, index + 2).toLowerCase() === 'th') {
      days.push('TH');
      index += 2;
      continue;
    }
    const code = SINGLE_DAY[cleaned[index].toUpperCase()];
    if (code) days.push(code);
    index += 1;
  }
  return Array.from(new Set(days));
}

const DAY_NAME =
  '(?:Mon(?:day)?|Tues?(?:day)?|Wed(?:nesday)?|Thu(?:rs?(?:day)?)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)s?';
const DAY_PHRASE = new RegExp(
  `(?<![A-Za-z])${DAY_NAME}(?:\\s*(?:,|and|&|through|thru|to|-|\u2013|/)\\s*${DAY_NAME})+(?![A-Za-z])`,
  'i',
);
const DAY_CONCAT = /(?<![A-Za-z])(?<run>(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun){2,5})(?![A-Za-z])/i;
const DAY_LETTERS = /(?<![A-Za-z])(?<run>[MTWRFS](?:\s*[,/]\s*[MTWRFS]){2,})(?![A-Za-z])/;

/** `_concat_days`: "MonWedFri" written as one run. */
export function concatDays(text: string): string[] {
  const match = DAY_CONCAT.exec(text);
  if (!match?.groups?.run) return [];
  const run = match.groups.run;
  const codes: string[] = [];
  for (let index = 0; index < run.length; index += 3) {
    const code = DAY_WORDS[run.slice(index, index + 3).toLowerCase()];
    if (code && !codes.includes(code)) codes.push(code);
  }
  return DAY_ORDER.filter((day) => codes.includes(day));
}

/** `_spelled_days`: "Monday, Wednesday, Friday", and ranges written "Tue through Thu". */
export function spelledDays(text: string): string[] {
  const phrase = DAY_PHRASE.exec(text);
  if (!phrase) return [];
  const value = phrase[0];
  const codes: string[] = [];
  for (const match of value.matchAll(DAY_WORD)) {
    const code = DAY_WORDS[match[0].slice(0, 3).toLowerCase()];
    if (code && !codes.includes(code)) codes.push(code);
  }
  let days = codes;
  if (codes.length === 2 && /\bthrough\b|\bthru\b/i.test(value)) {
    const first = DAY_ORDER.indexOf(codes[0]);
    const last = DAY_ORDER.indexOf(codes[1]);
    const [low, high] = first <= last ? [first, last] : [last, first];
    days = DAY_ORDER.slice(low, high + 1);
  }
  return DAY_ORDER.filter((day) => days.includes(day));
}

/** `_letter_days`: "M, W, F" as single letters. */
export function letterDays(text: string): string[] {
  const match = DAY_LETTERS.exec(text);
  if (!match?.groups?.run) return [];
  const codes = (match.groups.run.match(/[MTWRFS]/g) ?? []).map((letter) => SINGLE_DAY[letter]);
  return DAY_ORDER.filter((day) => codes.includes(day));
}

export function daysIn(text: string): string[] {
  const match = DAY_TOKEN.exec(text);
  if (match) return daysFromToken(match.groups?.days ?? '');
  // A spelled run ("Monday, Wednesday, Friday") is how most syllabi write a
  // meeting, and the compact code token above never matches it. Without these
  // fallbacks the reader silently found no meetings in real documents.
  for (const reader of [concatDays, spelledDays, letterDays]) {
    const days = reader(text);
    if (days.length) return days;
  }
  return [];
}

/** How many separate day groups a line names. */
export function dayGroups(text: string): number {
  return (
    (text.match(new RegExp(DAY_PHRASE.source, 'gi')) ?? []).length +
    (text.match(new RegExp(DAY_CONCAT.source, 'gi')) ?? []).length +
    (text.match(new RegExp(DAY_PAIR, 'gi')) ?? []).length +
    (text.match(new RegExp(DAY_LETTERS.source, 'g')) ?? []).length
  );
}

export function dayWords(text: string): string[] {
  const out: string[] = [];
  for (const match of text.matchAll(DAY_WORD)) {
    const key = match[0].slice(0, 3).toLowerCase();
    if (key in DAY_WORDS) out.push(DAY_WORDS[key]);
  }
  return out;
}

const CLOCK = /\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b/i;

export function hasClock(text: string): boolean {
  return CLOCK.test(text);
}

export const MEETING_LINE =
  /(?<![\d.])(?<start>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?)\s*(?:-{1,3}|\u2013|\u2014|to)\s*(?<end>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?)(?![\d.])/gi;

/** `_clock`: "11:00", "8am", "2.5"? -> 24h "HH:MM", or null. */
export function parseClock(raw: string): string | null {
  const text = raw.trim().toLowerCase().replace(/\./g, '');
  const meridiem = /(am|pm)$/.exec(text)?.[1];
  const numbers = text.replace(/(am|pm)$/, '').trim();
  const parts = numbers.split(':');
  const hour = Number(parts[0]);
  const minute = parts.length > 1 ? Number(parts[1]) : 0;
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null;
  if (hour > 23 || minute > 59) return null;
  let value = hour;
  if (meridiem === 'pm' && hour < 12) value = hour + 12;
  if (meridiem === 'am' && hour === 12) value = 0;
  if (!meridiem && hour <= 6 && parts.length > 1) value = hour + 12;
  return `${String(value).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

const ROOM = /\b(?:room|rm\.?|hall|building|bldg\.?)\s*[#:]?\s*(?<v>[A-Z0-9][A-Z0-9-]{1,10})\b/i;
const ROOM_CODE = /\b(?<v>[A-Z]{2,6}[- ]?\d{1,3}(?:\.\d{1,3})?[A-Z]?)\b/;
const ROOM_NAMED = /\b(?<v>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:Room\s+)?\d{1,4}[A-Z]?)\b/;
const ROOM_NUMBER_NAME = /\b(?<v>\d{2,4}[A-Z]?\s+[A-Z][a-z]{3,})\b/;
const ROOM_GLUED = /\b(?<v>\d{3,4}[A-Z][a-z]{2,})\b/;
const ROOM_STOP =
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|fall|spring|summer|winter|week|weeks|semester|quarter|credit|credits|unit|units|page|pages|section|sections|percent|am|pm|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lecture|lectures|lab|labs|recitation|recitations|class|classes|discussion|tutorial|exam|exams|quiz|quizzes|problem|problems|homework|chapter|part|module|modules|assignment|assignments)\b/i;

/** `_room_from`: the room named after the clock range, if any. */
export function roomFrom(text: string, from = 0): string | null {
  const tail = text.slice(from);
  const keyword = ROOM.exec(tail);
  if (keyword?.groups?.v && !ROOM_STOP.test(keyword.groups.v)) return keyword.groups.v;
  const named = ROOM_NAMED.exec(tail);
  if (named?.groups?.v && !ROOM_STOP.test(named.groups.v)) return named.groups.v;
  const code = ROOM_CODE.exec(tail);
  if (code?.groups?.v && !ROOM_STOP.test(code.groups.v)) return code.groups.v;
  const numberName = ROOM_NUMBER_NAME.exec(tail);
  if (numberName?.groups?.v) return numberName.groups.v;
  const glued = ROOM_GLUED.exec(tail);
  if (glued?.groups?.v) return glued.groups.v;
  return null;
}

export const OFFICE_HOURS_LINE = /office\s*hours?|\boh\s*:/i;
export const EXAM_LINE = /\b(?:final|midterm|make-?up)\s+exam|\bexam(?:ination)?s?\s*\d*\s*:/i;

/** The first sentence, without splitting at "Dr." or "Prof.". */
export function openingSentence(text: string): string {
  const pattern = /[.!?](?=\s)/g;
  for (const match of text.matchAll(pattern)) {
    const head = text.slice(0, (match.index ?? 0) + 1);
    if (/(?:\b(?:dr|mr|mrs|ms|prof|professor|st|vs)\.)$/i.test(head)) continue;
    return head;
  }
  return text;
}
