/**
 * Requirements the student defines, and how far their courses get them.
 *
 * Pace has no idea what a district requires, so it never assumes: a requirement is
 * a name and a credit count the user typed, and a course counts towards it only
 * because a person said so. The split is the same three-way one the rest of the app
 * uses — earned, in progress, planned — never one merged number.
 */
import type { RequirementRow } from '../database/academicRepo';
import type { CourseRow } from '../database/repo';
import { creditsOf, statusOf } from './record';

export interface RequirementProgress {
  requirement: RequirementRow;
  earned: number;
  inProgress: number;
  planned: number;
  /** Counts what is finished: a credit in progress has not been earned yet. */
  remaining: number;
  complete: boolean;
  courses: CourseRow[];
}

export function requirementProgress(
  requirements: RequirementRow[],
  links: Record<string, string[]>,
  courses: CourseRow[],
): RequirementProgress[] {
  const byId = new Map(courses.map((course) => [course.id, course]));
  return requirements.map((requirement) => {
    const ids = links[requirement.id] ?? [];
    const mapped = ids.map((id) => byId.get(id)).filter((course): course is CourseRow => Boolean(course));
    const totals = creditsOf(mapped);
    const remaining = Math.max(0, Math.round((requirement.required_credits - totals.earned) * 100) / 100);
    return {
      requirement,
      earned: totals.earned,
      inProgress: totals.inProgress,
      planned: totals.planned,
      remaining,
      complete: remaining === 0 && requirement.required_credits > 0,
      courses: mapped,
    };
  });
}

/** Courses that are not mapped to any requirement, so setup can point at them. */
export function unmappedCourses(courses: CourseRow[], links: Record<string, string[]>): CourseRow[] {
  const mapped = new Set(Object.values(links).flat());
  return courses.filter((course) => !mapped.has(course.id) && statusOf(course) !== 'planned');
}
