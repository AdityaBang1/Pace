/**
 * The academic record, as the screens need it.
 *
 * Pure functions over rows the database already returned: no SQLite, no React, so
 * the home summary, the term groups and the What If baseline cannot disagree about
 * what "13.0 credits earned" means. A course in progress is never counted as
 * earned, and a printed GPA is never presented as one Pace calculated.
 */
import type { CourseRow } from '../database/repo';
import { levelOf, letterFor, type CourseLevel } from './gpa';

export interface HistoryCourse {
  id: string;
  title: string;
  code: string | null;
  /** As printed: "94", "A". */
  mark: string | null;
  credits: number;
  term: string | null;
  inProgress: boolean;
  level: CourseLevel;
  /** A letter the course can be scaled by, when it has one. */
  letter: string | null;
}

export interface TermGroup {
  term: string;
  /** "2025–26 · Term 2", or the printed term when it is already short. */
  label: string;
  courses: HistoryCourse[];
  credits: number;
}

export interface AcademicSummary {
  /** Credits of finished courses, which is what the page's own Total counts. */
  earned: number;
  /** Credits of courses being taken now, kept apart. */
  inProgress: number;
  /** Every course, finished or not. */
  count: number;
  current: HistoryCourse[];
  /** Terms newest first, each holding its finished courses in printed order. */
  terms: TermGroup[];
  /** Courses with a letter, for a GPA baseline. */
  scaled: { credits: number; level: CourseLevel; grade: string }[];
}

function toCourse(row: CourseRow): HistoryCourse {
  const level = levelOf(row.title, true);
  return {
    id: row.id,
    title: row.title ?? row.code ?? 'Course',
    code: row.code,
    mark: row.grade,
    credits: row.credits ?? 0,
    term: row.term,
    inProgress: row.in_progress === 1,
    level,
    letter: letterFor(row.grade),
  };
}

/**
 * "2025-2026 Grade 09   Term 2" reads as "2025–26 · Term 2".
 *
 * The grade level is dropped because it is the school's bookkeeping, not the
 * student's: a term is a term. A term printed in another shape is passed through
 * as it came rather than maimed by a pattern that does not fit it.
 */
export function termLabel(term: string | null): string {
  if (!term) return 'No term printed';
  const years = /(\d{4})\s*[-–]\s*(\d{4})/.exec(term);
  const part = /Term\s*(\d+)/i.exec(term);
  if (years && part) return `${years[1]}–${years[2].slice(2)} · Term ${part[1]}`;
  if (years) return `${years[1]}–${years[2].slice(2)}`;
  return term.trim();
}

/** Sort key for a printed term, newest first, without parsing it into a date. */
function termOrder(term: string): number {
  const years = /(\d{4})/.exec(term);
  const part = /Term\s*(\d+)/i.exec(term);
  const year = years ? Number(years[1]) : 0;
  return year * 10 + (part ? Number(part[1]) : 9);
}

export function summarise(rows: CourseRow[]): AcademicSummary {
  const courses = rows.map(toCourse);
  const finished = courses.filter((course) => !course.inProgress);

  const groups = new Map<string, HistoryCourse[]>();
  for (const course of finished) {
    const key = course.term ?? '';
    groups.set(key, [...(groups.get(key) ?? []), course]);
  }
  const terms: TermGroup[] = [...groups.entries()]
    .sort((left, right) => termOrder(right[0]) - termOrder(left[0]))
    .map(([term, list]) => ({
      term,
      label: termLabel(term || null),
      courses: list,
      credits: Math.round(list.reduce((sum, course) => sum + course.credits, 0) * 100) / 100,
    }));

  const round = (value: number) => Math.round(value * 100) / 100;
  return {
    earned: round(finished.reduce((sum, course) => sum + course.credits, 0)),
    inProgress: round(
      courses.filter((course) => course.inProgress).reduce((sum, course) => sum + course.credits, 0),
    ),
    count: courses.length,
    current: courses.filter((course) => course.inProgress),
    terms,
    scaled: finished
      .filter((course) => course.letter !== null && course.credits > 0)
      .map((course) => ({ credits: course.credits, level: course.level, grade: course.letter as string })),
  };
}

/**
 * A transcript prints its own credit summary by area ("Mathematics 3.000"). That
 * is the only place Pace can learn the areas without inventing them, so it is kept
 * as the page stated it.
 */
export interface CreditArea {
  area: string;
  attempted: number | null;
  earned: number | null;
}

/** "13.0" — one decimal when it has one, no trailing zero when it does not. */
export function creditsValue(credits: number | null): string {
  if (credits == null) return '—';
  const rounded = Math.round(credits * 100) / 100;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

/** "4.24" from a printed 4.23529, or "—". */
export function gpaValue(gpa: number | null): string {
  if (gpa == null) return '—';
  return gpa.toFixed(2);
}
