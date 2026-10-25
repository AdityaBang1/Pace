/**
 * Pathways: one chosen schedule per academic year.
 *
 * A pathway does not hold courses of its own. It points at a schedule the user made,
 * so a scenario is a real schedule with real courses rather than a list of names,
 * and editing it goes through the same editor as the actual record. A year with no
 * explicit choice falls back to that year's main schedule — the rule the UI states
 * and this module implements.
 */
import type { AcademicYearRow, PathwayRow, ScheduleRow } from '../database/academicRepo';
import type { CourseRow } from '../database/repo';
import type { GpaScale } from './gpa';
import { computeGpa, creditsOf, type CreditTotals, type GpaFigures } from './record';

export interface PathwayYear {
  year: AcademicYearRow;
  schedule: ScheduleRow | null;
  /** True when the pathway chose this schedule; false when it fell back to main. */
  explicit: boolean;
  courses: CourseRow[];
  credits: CreditTotals;
  gpa: GpaFigures;
}

export interface PathwayView {
  pathway: PathwayRow;
  years: PathwayYear[];
  courses: CourseRow[];
  credits: CreditTotals;
  gpa: GpaFigures;
  /** Courses with no expected grade, so a projection can say what it is missing. */
  needingGrades: number;
}

export function selectionsOf(pathway: PathwayRow): Record<string, string> {
  try {
    const parsed: unknown = JSON.parse(pathway.selections);
    if (!parsed || typeof parsed !== 'object') return {};
    const out: Record<string, string> = {};
    for (const [year, schedule] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof schedule === 'string') out[year] = schedule;
    }
    return out;
  } catch {
    return {};
  }
}

export function buildPathway(
  pathway: PathwayRow,
  scale: GpaScale | null,
  years: AcademicYearRow[],
  schedules: ScheduleRow[],
  courses: CourseRow[],
): PathwayView {
  const selections = selectionsOf(pathway);
  const rows: PathwayYear[] = years.map((year) => {
    const explicitId = selections[year.id];
    const candidates = schedules.filter((schedule) => schedule.academic_year_id === year.id);
    const chosen =
      candidates.find((schedule) => schedule.id === explicitId) ??
      candidates.find((schedule) => schedule.is_main === 1) ??
      candidates[0] ??
      null;
    const inSchedule = chosen ? courses.filter((course) => course.schedule_id === chosen.id) : [];
    return {
      year,
      schedule: chosen,
      explicit: Boolean(chosen && explicitId === chosen.id),
      courses: inSchedule,
      credits: creditsOf(inSchedule),
      gpa: computeGpa(scale, inSchedule),
    };
  });
  const all = rows.flatMap((row) => row.courses);
  return {
    pathway,
    years: rows,
    courses: all,
    credits: creditsOf(all),
    gpa: computeGpa(scale, all),
    needingGrades: all.filter(
      (course) =>
        (course.status === 'planned' || course.status === 'current') && !course.expected_grade,
    ).length,
  };
}

export interface YearDiff {
  year: AcademicYearRow;
  scheduleA: ScheduleRow | null;
  scheduleB: ScheduleRow | null;
  same: boolean;
  added: string[];
  removed: string[];
  changed: { title: string; from: string; to: string }[];
}

export interface PathwayComparison {
  a: PathwayView;
  b: PathwayView;
  weightedDelta: number | null;
  unweightedDelta: number | null;
  creditDelta: number;
  years: YearDiff[];
  differenceCount: number;
}

const titles = (courses: CourseRow[]) =>
  courses.map((course) => (course.title ?? course.code ?? 'Course').trim());

function markOf(course: CourseRow): string {
  return course.grade ?? course.expected_grade ?? '—';
}

export function comparePathways(a: PathwayView, b: PathwayView): PathwayComparison {
  const years: YearDiff[] = a.years.map((yearA, index) => {
    const yearB = b.years[index];
    const left = titles(yearA.courses);
    const right = titles(yearB?.courses ?? []);
    const leftOnly = left.filter((title) => !right.includes(title));
    const rightOnly = right.filter((title) => !left.includes(title));
    const changed: YearDiff['changed'] = [];
    for (const course of yearA.courses) {
      const twin = (yearB?.courses ?? []).find(
        (other) => (other.title ?? other.code) === (course.title ?? course.code),
      );
      if (twin && markOf(twin) !== markOf(course)) {
        changed.push({ title: course.title ?? course.code ?? 'Course', from: markOf(course), to: markOf(twin) });
      }
    }
    return {
      year: yearA.year,
      scheduleA: yearA.schedule,
      scheduleB: yearB?.schedule ?? null,
      same: leftOnly.length === 0 && rightOnly.length === 0 && changed.length === 0,
      added: rightOnly,
      removed: leftOnly,
      changed,
    };
  });
  const differenceCount = years.filter((year) => !year.same).length;
  return {
    a,
    b,
    weightedDelta:
      a.gpa.weighted != null && b.gpa.weighted != null
        ? Math.round((b.gpa.weighted - a.gpa.weighted) * 1000) / 1000
        : null,
    unweightedDelta:
      a.gpa.unweighted != null && b.gpa.unweighted != null
        ? Math.round((b.gpa.unweighted - a.gpa.unweighted) * 1000) / 1000
        : null,
    creditDelta: Math.round((b.credits.all - a.credits.all) * 100) / 100,
    years,
    differenceCount,
  };
}
