/**
 * The academic record as a structure, not a list.
 *
 * The transcript produces courses; this module says what they add up to — per year,
 * per schedule, earned against in progress against planned — using the same scale
 * the What If screen uses. Screens read it; none of them do the arithmetic.
 */
import type { AcademicYearRow, ScheduleRow, TermRow } from '../database/academicRepo';
import type { CourseRow } from '../database/repo';
import { pointsFor, unweightedPoints, type GpaScale } from './gpa';

export type CourseStatus = 'completed' | 'current' | 'planned';

export interface CreditTotals {
  earned: number;
  inProgress: number;
  planned: number;
  /** Every credit the year or schedule holds, whichever state it is in. */
  all: number;
}

export interface GpaFigures {
  weighted: number | null;
  unweighted: number | null;
  /** Courses counted, so a screen can say what the number rests on. */
  counted: number;
  /** Courses with no usable mark: a projection that ignores them says so. */
  missing: number;
}

export interface ScheduleView {
  schedule: ScheduleRow;
  courses: CourseRow[];
  credits: CreditTotals;
  gpa: GpaFigures;
  /** "4 AP · 2 Honors · 1 Regular", only for types the courses actually use. */
  mix: { label: string; count: number }[];
}

export interface YearView {
  year: AcademicYearRow;
  terms: TermRow[];
  schedules: ScheduleView[];
  main: ScheduleView | null;
  courses: CourseRow[];
  credits: CreditTotals;
  gpa: GpaFigures;
}

export interface RecordView {
  years: YearView[];
  currentYear: YearView | null;
  credits: CreditTotals;
  gpa: GpaFigures;
  current: CourseRow[];
  courseCount: number;
}

const round = (value: number) => Math.round(value * 100) / 100;

export function statusOf(course: CourseRow): CourseStatus {
  if (course.status === 'completed' || course.status === 'current' || course.status === 'planned') {
    return course.status;
  }
  return course.in_progress === 1 ? 'current' : 'completed';
}

export function creditsOf(courses: CourseRow[]): CreditTotals {
  const total: CreditTotals = { earned: 0, inProgress: 0, planned: 0, all: 0 };
  for (const course of courses) {
    const credits = course.credits ?? 0;
    total.all += credits;
    const status = statusOf(course);
    if (status === 'completed') total.earned += credits;
    else if (status === 'current') total.inProgress += credits;
    else total.planned += credits;
  }
  total.earned = round(total.earned);
  total.inProgress = round(total.inProgress);
  total.planned = round(total.planned);
  total.all = round(total.all);
  return total;
}

/**
 * A GPA over the courses that carry a mark Pace can score.
 *
 * A completed course is scored on its final mark. A current or planned course is
 * scored on its expected grade, and one without an expected grade is counted as
 * missing rather than as a zero. `include_in_gpa` is respected: a course the user
 * excluded contributes nothing.
 */
export function computeGpa(scale: GpaScale | null, courses: CourseRow[]): GpaFigures {
  if (!scale) return { weighted: null, unweighted: null, counted: 0, missing: 0 };
  let weightedPoints = 0;
  let unweightedPointsTotal = 0;
  let credits = 0;
  let counted = 0;
  let missing = 0;
  for (const course of courses) {
    if (course.include_in_gpa === 0) continue;
    const status = statusOf(course);
    const mark = status === 'completed' ? course.grade : (course.expected_grade ?? course.grade);
    const credit = course.credits ?? 0;
    if (credit <= 0) continue;
    const weighted = pointsFor(scale, course.level ?? 'regular', mark);
    const unweighted = unweightedPoints(scale, mark);
    if (weighted == null || unweighted == null) {
      missing += 1;
      continue;
    }
    weightedPoints += weighted * credit;
    unweightedPointsTotal += unweighted * credit;
    credits += credit;
    counted += 1;
  }
  if (credits === 0) return { weighted: null, unweighted: null, counted: 0, missing };
  return {
    weighted: Math.round((weightedPoints / credits) * 1000) / 1000,
    unweighted: Math.round((unweightedPointsTotal / credits) * 1000) / 1000,
    counted,
    missing,
  };
}

const LEVEL_LABELS: Record<string, string> = { regular: 'Regular', honors: 'Honors', ap: 'AP' };

function mixOf(scale: GpaScale | null, courses: CourseRow[]): { label: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const course of courses) {
    const id = course.level ?? 'regular';
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([id, count]) => ({
      label: scale?.weights?.find((row) => row.id === id)?.label ?? LEVEL_LABELS[id] ?? id,
      count,
    }))
    .sort((left, right) => right.count - left.count);
}

function scheduleView(scale: GpaScale | null, schedule: ScheduleRow, courses: CourseRow[]): ScheduleView {
  return {
    schedule,
    courses,
    credits: creditsOf(courses),
    gpa: computeGpa(scale, courses),
    mix: mixOf(scale, courses),
  };
}

export function buildRecord(
  scale: GpaScale | null,
  years: AcademicYearRow[],
  terms: TermRow[],
  schedules: ScheduleRow[],
  courses: CourseRow[],
): RecordView {
  const views: YearView[] = years.map((year) => {
    const yearSchedules = schedules
      .filter((schedule) => schedule.academic_year_id === year.id)
      .sort((left, right) => right.is_main - left.is_main || left.sort_order - right.sort_order);
    const viewsOfSchedules = yearSchedules.map((schedule) =>
      scheduleView(
        scale,
        schedule,
        courses.filter((course) => course.schedule_id === schedule.id),
      ),
    );
    const yearCourses = courses.filter((course) => course.academic_year_id === year.id);
    return {
      year,
      terms: terms.filter((term) => term.academic_year_id === year.id),
      schedules: viewsOfSchedules,
      main: viewsOfSchedules.find((view) => view.schedule.is_main === 1) ?? viewsOfSchedules[0] ?? null,
      courses: yearCourses,
      credits: creditsOf(yearCourses),
      gpa: computeGpa(scale, yearCourses),
    };
  });

  const sorted = [...views].sort((left, right) => {
    const order = { past: 0, current: 1, future: 2 } as const;
    return order[left.year.status] - order[right.year.status] || left.year.sort_order - right.year.sort_order;
  });

  return {
    years: sorted,
    currentYear: sorted.find((view) => view.year.status === 'current') ?? null,
    credits: creditsOf(courses),
    gpa: computeGpa(scale, courses),
    current: courses.filter((course) => statusOf(course) === 'current'),
    courseCount: courses.length,
  };
}

/** Courses of one schedule, in the order the user arranged them. */
export function scheduleCourses(courses: CourseRow[], scheduleId: string): CourseRow[] {
  return courses
    .filter((course) => course.schedule_id === scheduleId)
    .sort((left, right) => (left.sort_order ?? 0) - (right.sort_order ?? 0));
}

/** What a course's status reads as on a tile. */
export function statusLabel(course: CourseRow): string {
  const status = statusOf(course);
  if (status === 'current') return 'In progress';
  if (status === 'planned') return course.expected_grade ? `Expected ${course.expected_grade}` : 'Planned';
  return course.grade ?? 'Completed';
}
