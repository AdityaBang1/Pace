/**
 * Section detection: the headings a syllabus actually prints.
 *
 * The vocabulary comes from the exported `rules.model` (the same 254 headings
 * `pace.field_corpus` recognises), so a heading means the same thing in both
 * readers. Model classification happens afterwards, in the engine, because it
 * needs the tokeniser and the ONNX session.
 */

import { cleanCandidate, lines, normSpace } from './text';

export interface RuleTables {
  section_map: Record<string, string>;
  boundary_only: string[];
  heading_max_chars: number;
  section_values: Record<string, string>;
  policy_cues: Record<string, string>;
}

export interface DetectedSection {
  text: string;
  start: number;
  end: number;
  inline: string;
  /** The vocabulary label, when the heading is a known one. */
  label: string | null;
}

const HEADING_NUMBER = /^\s*(?:[0-9]{1,2}[.)]|[ivxlcdm]{1,7}[.)]|[a-z][.)]|[\u2022\-*\u25aa\u25e6]+)\s+/i;
const HEADING_TRAILER = /[:\s\-\u2013\u2014]+$/;

export function normalizeHeading(line: string): string {
  let text = line.trim();
  text = text.replace(HEADING_NUMBER, '');
  text = text.replace(HEADING_TRAILER, '');
  return text.replace(/\s+/g, ' ').trim().toLowerCase();
}

export function detectSections(text: string, rules: RuleTables): DetectedSection[] {
  const out: DetectedSection[] = [];
  const boundary = new Set(rules.boundary_only);
  for (const line of lines(text)) {
    const stripped = line.text.trim();
    if (!stripped || stripped.length > rules.heading_max_chars) continue;
    const normalized = normalizeHeading(stripped);
    if (!normalized) continue;
    let label: string | null = rules.section_map[normalized] ?? null;
    let inline = '';
    if (label === null) {
      const colon = stripped.indexOf(':');
      if (colon > 0 && colon <= rules.heading_max_chars) {
        const head = normalizeHeading(stripped.slice(0, colon));
        const found = rules.section_map[head];
        if (found) {
          label = found;
          inline = stripped.slice(colon + 1).trim();
        }
      }
    }
    if (label === null && !boundary.has(normalized)) continue;
    // Python spans a heading with its trailing newline (`start + len(line)` over
    // `splitlines(keepends=True)`), and the policy evidence inherits that span.
    // Ending at the last character instead made every heading one byte short.
    const end = line.end + (text[line.end] === '\n' ? 1 : 0);
    out.push({ text: stripped, start: line.start, end, inline, label });
  }
  return out;
}

const HEADING_FILLER = new Set(['and', 'of', 'for', 'the', 'to', 'with', 'in', 'on', 'a', 'an', '&']);
const NEXT_SECTION_LABEL = /^\s*(?<label>[A-Z][A-Za-z &/.'-]{2,40}?)\s*:\s*\S/;

/** `_heading_like`: a short capitalised line that names a section. */
export function headingLike(line: string): boolean {
  if (line.length < 2 || line.length > 48 || line.endsWith('.')) return false;
  if (/[0-9]/.test(line) || line.includes(':')) return false;
  const words = line.split(/\s+/);
  if (words.length < 1 || words.length > 5) return false;
  return words.every(
    (word) => /^[A-Z]/.test(word) || HEADING_FILLER.has(word.toLowerCase()),
  );
}

/** `_section_body`: the section's own text, stopped by the next heading. */
export function sectionBody(
  text: string,
  all: DetectedSection[],
  section: DetectedSection,
  limit = 400,
): string {
  const index = all.indexOf(section);
  let end = text.length;
  for (const later of all.slice(index + 1)) {
    if (later.start > section.end) {
      end = later.start;
      break;
    }
  }
  const kept: string[] = [];
  let seen = 0;
  for (const line of text.slice(section.end, end).split('\n')) {
    const stripped = line.trim();
    if (
      stripped &&
      kept.some((entry) => entry.trim()) &&
      NEXT_SECTION_LABEL.test(stripped) &&
      stripped.length <= 70
    ) {
      break;
    }
    if (seen >= 30 && stripped && headingLike(stripped) && kept.length && !kept[kept.length - 1].trim()) {
      break;
    }
    kept.push(line);
    seen += stripped.length;
  }
  return normSpace(kept.join('\n')).slice(0, limit);
}

/** `_policy_body`: prefer the value printed on the heading line. */
export function policyBody(
  text: string,
  all: DetectedSection[],
  section: DetectedSection,
): { body: string; inline: boolean } {
  const inline = normSpace(section.inline || '');
  if (inline) return { body: inline, inline: true };
  return { body: sectionBody(text, all, section), inline: false };
}

export { cleanCandidate };
