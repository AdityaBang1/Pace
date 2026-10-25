/**
 * GPA, deterministically.
 *
 * Pace never guesses a school's weighting scheme. A transcript states a GPA, and
 * Pace shows what the page said next to the word "Reported". A projection is only
 * ever made with a scale the user has set, so a What If number is arithmetic on a
 * stated rule rather than a claim about a district's policy.
 */

export type CourseLevel = 'regular' | 'honors' | 'ap';

export const LEVELS: { value: CourseLevel; label: string }[] = [
  { value: 'regular', label: 'Regular' },
  { value: 'honors', label: 'Honors' },
  { value: 'ap', label: 'AP' },
];

/** The points a letter is worth at each level. One accent-free rule per level. */
export interface GpaScale {
  /** e.g. [4, 3, 2, 1, 0] for A, B, C, D, F. */
  regular: Record<string, number>;
  honors: Record<string, number>;
  ap: Record<string, number>;
  /**
   * A printed percentage read as a letter. Without bands, a numeric mark is left
   * out of a calculated GPA rather than guessed at.
   */
  bands?: GradeBand[];
  /** Extra course types the student's school uses, on top of the regular table. */
  weights?: WeightType[];
}

export interface GradeBand {
  min: number;
  letter: string;
}

export interface WeightType {
  id: string;
  label: string;
  /** Points added to the regular table's value for the same letter. */
  bonus: number;
}

/** The bands most US high schools print, offered as a default to edit. */
export const DEFAULT_BANDS: GradeBand[] = [
  { min: 90, letter: 'A' },
  { min: 80, letter: 'B' },
  { min: 70, letter: 'C' },
  { min: 60, letter: 'D' },
  { min: 0, letter: 'F' },
];

export const LETTERS = ['A', 'B', 'C', 'D', 'F'] as const;
export type Letter = (typeof LETTERS)[number];

/**
 * The scale a US high school most often prints: 4 / 3 / 2 / 1 / 0, plus a one
 * point lift for honors and AP. It is a *default to edit*, not a claim.
 */
export const DEFAULT_SCALE: GpaScale = {
  regular: { A: 4, B: 3, C: 2, D: 1, F: 0 },
  honors: { A: 4.5, B: 3.5, C: 2.5, D: 1.5, F: 0 },
  ap: { A: 5, B: 4, C: 3, D: 2, F: 0 },
};

export function scalePoints(scale: GpaScale, level: CourseLevel, grade: string): number | null {
  const table = scale[level] ?? scale.regular;
  const points = table[grade.toUpperCase()];
  return typeof points === 'number' && Number.isFinite(points) ? points : null;
}

/** A printed percentage as a letter, using the scale's own bands. */
export function letterFromPercent(percent: number, bands: GradeBand[] = DEFAULT_BANDS): string | null {
  if (!Number.isFinite(percent)) return null;
  const sorted = [...bands].sort((left, right) => right.min - left.min);
  return sorted.find((band) => percent >= band.min)?.letter ?? null;
}

/** A mark — "94", "A", "A-" — as a letter this scale can score. */
export function letterOf(mark: string | null, bands: GradeBand[] = DEFAULT_BANDS): string | null {
  if (!mark) return null;
  const trimmed = mark.trim();
  const numeric = Number(trimmed.replace('%', ''));
  if (Number.isFinite(numeric) && trimmed !== '') return letterFromPercent(numeric, bands);
  return letterFor(trimmed);
}

/**
 * The points one course is worth: its level's own table when the scale has one,
 * otherwise the regular value plus the course type's bonus. A custom type the scale
 * does not know scores as regular — the honest fallback, not a guess at a lift.
 */
export function pointsFor(
  scale: GpaScale,
  level: string,
  grade: string | null,
  bands: GradeBand[] = scale.bands ?? DEFAULT_BANDS,
): number | null {
  const letter = letterOf(grade, bands);
  if (!letter) return null;
  const known = level === 'honors' || level === 'ap' || level === 'regular' ? scale[level] : null;
  if (known) return typeof known[letter] === 'number' ? known[letter] : null;
  const base = scale.regular[letter];
  if (typeof base !== 'number') return null;
  const type = scale.weights?.find((row) => row.id === level);
  return base + (type?.bonus ?? 0);
}

/** Points against the regular table only: the GPA without any lift. */
export function unweightedPoints(
  scale: GpaScale,
  grade: string | null,
  bands: GradeBand[] = scale.bands ?? DEFAULT_BANDS,
): number | null {
  const letter = letterOf(grade, bands);
  if (!letter) return null;
  const points = scale.regular[letter];
  return typeof points === 'number' ? points : null;
}

/** A printed mark as the letter it is worth, when it is one Pace can scale. */
export function letterFor(mark: string | null): Letter | null {
  if (!mark) return null;
  const head = mark.trim().toUpperCase().charAt(0);
  return (LETTERS as readonly string[]).includes(head) ? (head as Letter) : null;
}

export interface WeightedCourse {
  credits: number;
  level: CourseLevel;
  grade: string;
}

/**
 * A GPA over courses that carry a letter and a credit. A course whose mark is a
 * percentage, a pass, or nothing at all contributes nothing — the alternative is
 * inventing a letter for it.
 */
export function weightedGpa(scale: GpaScale, courses: WeightedCourse[]): number | null {
  let points = 0;
  let credits = 0;
  for (const course of courses) {
    const value = scalePoints(scale, course.level, course.grade);
    if (value == null) continue;
    const weight = course.credits > 0 ? course.credits : 0;
    if (weight === 0) continue;
    points += value * weight;
    credits += weight;
  }
  if (credits === 0) return null;
  return Math.round((points / credits) * 1000) / 1000;
}

/* ---------------------------------------------------------------------------
 * Reading a level off the printed title
 * ------------------------------------------------------------------------- */

/**
 * A level is read only from a title that says so — `AP Biology`, `Biology Honors`,
 * `Accel Algebra`. A transcript whose comments define "Accelerated denotes Honors"
 * is handled by `acceleratedIsHonors`.
 */
export function levelOf(title: string | null, acceleratedIsHonors: boolean): CourseLevel {
  if (!title) return 'regular';
  const text = ` ${title.toLowerCase()} `;
  if (/(^|\s)ap(\s|$)|advanced placement/.test(text)) return 'ap';
  if (/honors?(\s|$)/.test(text)) return 'honors';
  if (acceleratedIsHonors && /accel/.test(text)) return 'honors';
  // A dual-enrollment course is college level, and nothing here models that:
  // calling it regular is the honest default rather than an edit the user did not
  // make.
  return 'regular';
}

/** The highest of a set of levels, used to name a term's own character. */
export function levelLabel(level: CourseLevel): string {
  return LEVELS.find((row) => row.value === level)?.label ?? 'Regular';
}

/* ---------------------------------------------------------------------------
 * What If
 * ------------------------------------------------------------------------- */

export interface PlannedInput {
  credits: number;
  level: CourseLevel;
  expectedGrade: string;
}

export interface Projection {
  /** The projected weighted GPA, or null when it cannot be computed. */
  gpa: number | null;
  /** Credits the projection is based on. */
  credits: number;
  /** Why there is no number, when there is none. */
  reason: 'ok' | 'no-history' | 'no-courses' | 'no-scale';
}

/**
 * Project the weighted GPA the planned courses would produce.
 *
 * The baseline is the courses the user already has a letter for, weighted by the
 * same scale, so the projection is comparable to what it extends. Without such a
 * baseline there is nothing to project from, and Pace says so rather than showing
 * a number that is really just the planned term's average.
 */
export function projectGpa(
  scale: GpaScale | null,
  history: WeightedCourse[],
  planned: PlannedInput[],
): Projection {
  if (!scale) return { gpa: null, credits: 0, reason: 'no-scale' };
  if (!planned.length) return { gpa: null, credits: 0, reason: 'no-courses' };
  const baseline = weightedGpa(scale, history);
  if (baseline == null) return { gpa: null, credits: 0, reason: 'no-history' };
  const combined: WeightedCourse[] = [
    ...history,
    ...planned.map((course) => ({
      credits: course.credits,
      level: course.level,
      grade: course.expectedGrade,
    })),
  ];
  const credits = combined.reduce((sum, course) => sum + (course.credits > 0 ? course.credits : 0), 0);
  return { gpa: weightedGpa(scale, combined), credits: Math.round(credits * 100) / 100, reason: 'ok' };
}

/** Parse the scale Pace stores in `settings`, falling back to the default. */
export function parseScale(json: string | null): GpaScale | null {
  if (!json) return null;
  try {
    const parsed = JSON.parse(json) as Partial<GpaScale>;
    if (!parsed || typeof parsed !== 'object') return null;
    const pick = (table: unknown): Record<string, number> | null => {
      if (!table || typeof table !== 'object') return null;
      const out: Record<string, number> = {};
      for (const letter of LETTERS) {
        const value = (table as Record<string, unknown>)[letter];
        if (typeof value !== 'number' || !Number.isFinite(value)) return null;
        out[letter] = value;
      }
      return out;
    };
    const regular = pick(parsed.regular);
    const honors = pick(parsed.honors);
    const ap = pick(parsed.ap);
    if (!regular || !honors || !ap) return null;
    const bands = Array.isArray(parsed.bands)
      ? parsed.bands.filter(
          (band): band is GradeBand =>
            !!band && typeof band.min === 'number' && Number.isFinite(band.min) && typeof band.letter === 'string',
        )
      : DEFAULT_BANDS;
    const weights = Array.isArray(parsed.weights)
      ? parsed.weights.filter(
          (row): row is WeightType =>
            !!row && typeof row.id === 'string' && typeof row.label === 'string' && Number.isFinite(row.bonus),
        )
      : undefined;
    return { regular, honors, ap, bands: bands.length ? bands : DEFAULT_BANDS, weights };
  } catch {
    return null;
  }
}

/** The scale as JSON, for `settings`. */
export function serialiseScale(scale: GpaScale): string {
  return JSON.stringify({
    regular: scale.regular,
    honors: scale.honors,
    ap: scale.ap,
    bands: scale.bands ?? DEFAULT_BANDS,
    weights: scale.weights ?? [],
  });
}


