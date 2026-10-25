/** Small, shared formatters. Every screen shows weights, days and dates the same way. */

export function weightLabel(weight: number): string {
  const rounded = Math.round(weight * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}%` : `${rounded.toFixed(1)}%`;
}

export function creditsLabel(credits: number | null): string | null {
  if (credits == null) return null;
  const rounded = Math.round(credits * 100) / 100;
  return `${Number.isInteger(rounded) ? rounded : rounded.toFixed(1)} credits`;
}

const DAY_NAMES: Record<string, string> = {
  MO: 'Mon',
  TU: 'Tue',
  WE: 'Wed',
  TH: 'Thu',
  FR: 'Fri',
  SA: 'Sat',
  SU: 'Sun',
};

export function daysLabel(days: string[]): string {
  // "Tue, Thu" reads better than "Tue / Thu" on a phone, and the full stop matters
  // in a course card where the days sit above the time.
  return days.map((day) => DAY_NAMES[day] ?? day).join(', ');
}

/** "11:00 AM – 12:15 PM" from 24h strings; anything unparseable is passed through. */
export function timeRangeLabel(start: string | null, end: string | null): string | null {
  if (!start && !end) return null;
  const pretty = (value: string | null) => {
    if (!value) return null;
    const match = /^(\d{1,2}):(\d{2})$/.exec(value);
    if (!match) return value;
    const hours = Number(match[1]);
    const minutes = match[2];
    const suffix = hours >= 12 ? 'PM' : 'AM';
    const display = hours % 12 === 0 ? 12 : hours % 12;
    return `${display}:${minutes} ${suffix}`;
  };
  const left = pretty(start);
  const right = pretty(end);
  if (left && right) return `${left} – ${right}`;
  return left ?? right;
}

export function relativeDay(iso: string): string {
  const then = new Date(iso);
  const now = new Date();
  const days = Math.floor((now.getTime() - then.getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function documentTypeLabel(label: string | null | undefined): string {
  if (!label) return 'Document';
  return label
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function evidenceExcerpt(text: string | null, start: number | null, end: number | null): string {
  if (!text) return '';
  if (start == null || end == null) return text;
  const lineStart = text.lastIndexOf('\n', start) + 1;
  const lineEnd = text.indexOf('\n', end);
  return text.slice(lineStart, lineEnd === -1 ? text.length : lineEnd);
}
