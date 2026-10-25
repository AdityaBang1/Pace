/**
 * The academic structure: years, terms, schedules, pathways, requirements, scores.
 *
 * Separate from `repo.ts` on purpose. `repo.ts` owns documents, evidence and the
 * import transaction — the part of the store that a parser writes. This file owns
 * the part a person edits: which year a course belongs to, which of a year's
 * schedules they are looking at, which schedules a pathway selects.
 *
 * Nothing here deletes a course. A year or a schedule that goes away takes its
 * structure with it, and the courses it held are re-placed by `ensureStructure`
 * rather than silently dropped.
 */
import type { SQLiteDatabase } from 'expo-sqlite';

import { levelOf } from '../domain/gpa';
import type { CourseRow } from './repo';

export type YearStatus = 'past' | 'current' | 'future';
export type ScheduleType = 'actual' | 'planned' | 'scenario';

export interface AcademicYearRow {
  id: string;
  name: string;
  school_year_label: string | null;
  grade_level: string | null;
  status: YearStatus;
  sort_order: number;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface TermRow {
  id: string;
  academic_year_id: string;
  name: string;
  status: string;
  sort_order: number;
}

export interface ScheduleRow {
  id: string;
  academic_year_id: string;
  name: string;
  is_main: number;
  type: ScheduleType;
  source: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface PathwayRow {
  id: string;
  name: string;
  is_active: number;
  /** JSON: academic year id → schedule id. A missing year falls back to main. */
  selections: string;
  created_at: string;
  updated_at: string;
}

export interface RequirementRow {
  id: string;
  name: string;
  required_credits: number;
  is_total: number;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface TestSection {
  name: string;
  score: string;
}

export interface TestScoreRow {
  id: string;
  kind: string;
  name: string;
  taken_on: string | null;
  total_score: string | null;
  max_score: string | null;
  sections: string;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface AcademicStructure {
  years: AcademicYearRow[];
  terms: TermRow[];
  schedules: ScheduleRow[];
  pathways: PathwayRow[];
  requirements: RequirementRow[];
  /** requirement id → the course ids mapped to it. */
  links: Record<string, string[]>;
  tests: TestScoreRow[];
}

let counter = 0;

/** A local id. These rows are this device's, so there is no server to ask. */
export function newId(prefix: string): string {
  counter = (counter + 1) % 1000;
  return `${prefix}-${Date.now().toString(36)}-${counter}-${Math.random().toString(36).slice(2, 7)}`;
}

export function parseSelections(row: PathwayRow): Record<string, string> {
  try {
    const parsed: unknown = JSON.parse(row.selections);
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

export async function readStructure(db: SQLiteDatabase): Promise<AcademicStructure> {
  const [years, terms, schedules, pathways, requirements, links, tests] = await Promise.all([
    db.getAllAsync<AcademicYearRow>(
      'SELECT * FROM academic_years ORDER BY sort_order ASC, created_at ASC',
    ),
    db.getAllAsync<TermRow>(
      'SELECT * FROM academic_terms ORDER BY academic_year_id ASC, sort_order ASC, name ASC',
    ),
    db.getAllAsync<ScheduleRow>(
      'SELECT * FROM schedules ORDER BY academic_year_id ASC, sort_order ASC, created_at ASC',
    ),
    db.getAllAsync<PathwayRow>('SELECT * FROM pathways ORDER BY created_at ASC'),
    db.getAllAsync<RequirementRow>('SELECT * FROM requirements ORDER BY sort_order ASC, created_at ASC'),
    db.getAllAsync<{ requirement_id: string; course_id: string }>(
      'SELECT requirement_id, course_id FROM course_requirements',
    ),
    db.getAllAsync<TestScoreRow>('SELECT * FROM test_scores ORDER BY taken_on DESC, created_at DESC'),
  ]);
  const byRequirement: Record<string, string[]> = {};
  for (const link of links) {
    byRequirement[link.requirement_id] = [...(byRequirement[link.requirement_id] ?? []), link.course_id];
  }
  return { years, terms, schedules, pathways, requirements, links: byRequirement, tests };
}

/* ---------------------------------------------------------------------------
 * Placing courses
 * ------------------------------------------------------------------------- */

/** "2025-2026 Grade 09   Term 2" → { label: '2025-2026', grade: '09', term: 'Term 2' }. */
export function parseTerm(term: string | null): { label: string | null; grade: string | null; name: string | null } {
  if (!term) return { label: null, grade: null, name: null };
  const years = /(\d{4})\s*[-–]\s*(\d{4})/.exec(term);
  const grade = /Grade\s*(\d{1,2})/i.exec(term);
  const part = /(Term|Semester|Quarter|Trimester)\s*(\d+)/i.exec(term);
  return {
    label: years ? `${years[1]}-${years[2]}` : null,
    grade: grade ? grade[1] : null,
    name: part ? `${part[1][0].toUpperCase()}${part[1].slice(1).toLowerCase()} ${part[2]}` : term.trim() || null,
  };
}

/** "2025-2026" → "2025–26". A label that does not look like a year is passed through. */
export function yearName(label: string | null, grade: string | null): string {
  const name = label ? `${label.slice(0, 4)}–${label.slice(7, 9)}` : null;
  if (name && grade) return `${name} · Grade ${Number(grade)}`;
  if (name) return name;
  if (grade) return `Grade ${Number(grade)}`;
  return 'Imported';
}

async function yearFor(
  db: SQLiteDatabase,
  key: { id: string; name: string; label: string | null; grade: string | null; status: YearStatus },
): Promise<string> {
  const existing = await db.getFirstAsync<{ id: string }>('SELECT id FROM academic_years WHERE id = ?', [
    key.id,
  ]);
  if (existing) return key.id;
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM academic_years',
  );
  const now = new Date().toISOString();
  await db.runAsync(
    `INSERT INTO academic_years
       (id, name, school_year_label, grade_level, status, sort_order, source, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, 'transcript', ?, ?)`,
    [key.id, key.name, key.label, key.grade, key.status, (max?.value ?? 0) + 1, now, now],
  );
  return key.id;
}

async function scheduleFor(db: SQLiteDatabase, yearId: string, name: string, type: ScheduleType): Promise<string> {
  const existing = await db.getFirstAsync<{ id: string }>(
    'SELECT id FROM schedules WHERE academic_year_id = ? AND name = ?',
    [yearId, name],
  );
  if (existing) return existing.id;
  const now = new Date().toISOString();
  const id = newId('schedule');
  await db.runAsync(
    `INSERT INTO schedules (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
     VALUES (?, ?, ?, (SELECT COUNT(*) = 0 FROM schedules WHERE academic_year_id = ?), ?, 'transcript', 0, ?, ?)`,
    [id, yearId, name, yearId, type, now, now],
  );
  return id;
}

async function termFor(db: SQLiteDatabase, yearId: string, name: string, status: string): Promise<string> {
  const existing = await db.getFirstAsync<{ id: string }>(
    'SELECT id FROM academic_terms WHERE academic_year_id = ? AND name = ?',
    [yearId, name],
  );
  if (existing) return existing.id;
  const id = newId('term');
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM academic_terms WHERE academic_year_id = ?',
    [yearId],
  );
  await db.runAsync(
    'INSERT INTO academic_terms (id, academic_year_id, name, status, sort_order) VALUES (?, ?, ?, ?, ?)',
    [id, yearId, name, status, (max?.value ?? 0) + 1],
  );
  return id;
}

/**
 * Give every course that has no place one, derived from what the transcript said.
 *
 * Idempotent: it only touches rows whose `academic_year_id` is null, so it can be
 * called on every screen focus without rewriting a user's arrangement. A course in
 * progress lands in the newest year's "Current" schedule; a finished course lands
 * in the year its printed term names.
 */
export async function ensureStructure(db: SQLiteDatabase): Promise<void> {
  const rows = await db.getAllAsync<{
    id: string;
    title: string | null;
    term: string | null;
    in_progress: number;
  }>('SELECT id, title, term, in_progress FROM courses WHERE academic_year_id IS NULL');
  if (!rows.length) return;

  const finishedYears = new Map<string, { id: string; rows: typeof rows }>();
  const current: typeof rows = [];
  for (const row of rows) {
    if (row.in_progress === 1 || !row.term) {
      current.push(row);
      continue;
    }
    const parsed = parseTerm(row.term);
    const id = `year-${(parsed.label ?? row.term).replace(/[^0-9a-zA-Z]+/g, '-').toLowerCase()}${
      parsed.grade ? `-g${parsed.grade}` : ''
    }`;
    const bucket = finishedYears.get(id) ?? { id, rows: [] as typeof rows };
    bucket.rows.push(row);
    finishedYears.set(id, bucket);
  }

  for (const [, bucket] of finishedYears) {
    const sample = parseTerm(bucket.rows[0].term);
    const yearId = await yearFor(db, {
      id: bucket.id,
      name: yearName(sample.label, sample.grade),
      label: sample.label,
      grade: sample.grade,
      status: 'past',
    });
    const scheduleId = await scheduleFor(db, yearId, 'Actual', 'actual');
    for (const row of bucket.rows) {
      const parsed = parseTerm(row.term);
      const termId = parsed.name
        ? await termFor(db, yearId, parsed.name, 'completed')
        : null;
      await db.runAsync(
        `UPDATE courses SET academic_year_id = ?, schedule_id = ?, academic_term_id = ?,
           level = ?, status = 'completed' WHERE id = ?`,
        [yearId, scheduleId, termId, levelOf(row.title, true), row.id],
      );
    }
  }

  if (current.length) {
    // A page printed before today's enrolment lists finished terms and work in
    // progress, and the work in progress belongs to the year after them. Pace does
    // not know what that year is called, so it says "Current" rather than putting
    // grade 10's courses under grade 9's year. A page with no finished terms at all
    // is its own current year.
    const target = await yearFor(db, {
      id: 'year-current',
      name: 'Current',
      label: null,
      grade: null,
      status: 'current',
    });
    await db.runAsync("UPDATE academic_years SET status = 'current' WHERE id = ?", [target]);
    const scheduleId = await scheduleFor(db, target, 'Current', 'actual');
    for (const row of current) {
      await db.runAsync(
        `UPDATE courses SET academic_year_id = ?, schedule_id = ?, level = ?,
           status = 'current' WHERE id = ?`,
        [target, scheduleId, levelOf(row.title, true), row.id],
      );
    }
  }

  // Exactly one current year, and it is the latest one that has courses.
  const years = await db.getAllAsync<AcademicYearRow>(
    'SELECT * FROM academic_years ORDER BY sort_order ASC, created_at ASC',
  );
  const currentYear = [...years].reverse().find((year) => year.status === 'current');
  if (currentYear) {
    await db.runAsync("UPDATE academic_years SET status = 'past' WHERE status = 'current' AND id <> ?", [
      currentYear.id,
    ]);
  }
}

/* ---------------------------------------------------------------------------
 * Years and terms
 * ------------------------------------------------------------------------- */

export async function listYears(db: SQLiteDatabase): Promise<AcademicYearRow[]> {
  return db.getAllAsync<AcademicYearRow>(
    'SELECT * FROM academic_years ORDER BY sort_order ASC, created_at ASC',
  );
}

export interface YearInput {
  id?: string;
  name: string;
  schoolYearLabel?: string | null;
  gradeLevel?: string | null;
  status: YearStatus;
}

export async function saveYear(db: SQLiteDatabase, input: YearInput): Promise<string> {
  const now = new Date().toISOString();
  const id = input.id ?? newId('year');
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM academic_years',
  );
  await db.runAsync(
    `INSERT INTO academic_years
       (id, name, school_year_label, grade_level, status, sort_order, source, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, 'manual', ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       name = excluded.name,
       school_year_label = excluded.school_year_label,
       grade_level = excluded.grade_level,
       status = excluded.status,
       updated_at = excluded.updated_at`,
    [id, input.name, input.schoolYearLabel ?? null, input.gradeLevel ?? null, input.status, (max?.value ?? 0) + 1, now, now],
  );
  if (input.status === 'current') {
    await db.runAsync("UPDATE academic_years SET status = 'past' WHERE status = 'current' AND id <> ?", [id]);
  }
  return id;
}

/** A new year arrives with one main schedule and no terms: "Main" is enough. */
export async function saveYearWithSchedule(db: SQLiteDatabase, input: YearInput): Promise<string> {
  const id = await saveYear(db, input);
  await scheduleForNamed(db, id, input.status === 'current' ? 'Current' : 'Main', input.status === 'future' ? 'planned' : 'actual');
  return id;
}

async function scheduleForNamed(db: SQLiteDatabase, yearId: string, name: string, type: ScheduleType): Promise<string> {
  const existing = await db.getFirstAsync<{ id: string }>(
    'SELECT id FROM schedules WHERE academic_year_id = ? AND is_main = 1',
    [yearId],
  );
  if (existing) return existing.id;
  const now = new Date().toISOString();
  const id = newId('schedule');
  await db.runAsync(
    `INSERT INTO schedules (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
     VALUES (?, ?, ?, 1, ?, 'manual', 0, ?, ?)`,
    [id, yearId, name, type, now, now],
  );
  return id;
}

/**
 * Delete a year. Its courses are not deleted: they are un-placed, and
 * `ensureStructure` puts them back in an imported year on the next focus.
 */
export async function deleteYear(db: SQLiteDatabase, id: string): Promise<void> {
  await db.withTransactionAsync(async () => {
    await db.runAsync('UPDATE courses SET academic_year_id = NULL, schedule_id = NULL, academic_term_id = NULL WHERE academic_year_id = ?', [id]);
    await db.runAsync('DELETE FROM academic_years WHERE id = ?', [id]);
  });
}

export async function saveTerm(
  db: SQLiteDatabase,
  input: { id?: string; academicYearId: string; name: string; status?: string },
): Promise<string> {
  const id = input.id ?? newId('term');
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM academic_terms WHERE academic_year_id = ?',
    [input.academicYearId],
  );
  await db.runAsync(
    `INSERT INTO academic_terms (id, academic_year_id, name, status, sort_order)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET name = excluded.name, status = excluded.status`,
    [id, input.academicYearId, input.name, input.status ?? 'future', (max?.value ?? 0) + 1],
  );
  return id;
}

/* ---------------------------------------------------------------------------
 * Schedules
 * ------------------------------------------------------------------------- */

export async function listSchedules(db: SQLiteDatabase): Promise<ScheduleRow[]> {
  return db.getAllAsync<ScheduleRow>(
    'SELECT * FROM schedules ORDER BY academic_year_id ASC, is_main DESC, sort_order ASC, created_at ASC',
  );
}

export async function saveSchedule(
  db: SQLiteDatabase,
  input: { id?: string; academicYearId: string; name: string; type?: ScheduleType },
): Promise<string> {
  const id = input.id ?? newId('schedule');
  const now = new Date().toISOString();
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM schedules WHERE academic_year_id = ?',
    [input.academicYearId],
  );
  await db.runAsync(
    `INSERT INTO schedules (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
     VALUES (?, ?, ?, (SELECT COUNT(*) = 0 FROM schedules WHERE academic_year_id = ?), ?, 'manual', ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at`,
    [id, input.academicYearId, input.name, input.academicYearId, input.type ?? 'planned', (max?.value ?? 0) + 1, now, now],
  );
  return id;
}

export async function setMainSchedule(db: SQLiteDatabase, id: string): Promise<void> {
  const row = await db.getFirstAsync<{ academic_year_id: string }>(
    'SELECT academic_year_id FROM schedules WHERE id = ?',
    [id],
  );
  if (!row) return;
  await db.withTransactionAsync(async () => {
    await db.runAsync('UPDATE schedules SET is_main = 0 WHERE academic_year_id = ?', [row.academic_year_id]);
    await db.runAsync('UPDATE schedules SET is_main = 1 WHERE id = ?', [id]);
  });
}

/**
 * Copy a schedule and its courses.
 *
 * The copy is the user's scenario, so its courses carry no document, no evidence and
 * no import: editing the copy can never rewrite what a transcript stated. Completed
 * grades are kept as the grade the copy starts from, and its status becomes planned
 * when the original was a plan and stays as it was when the original was history.
 */
export async function duplicateSchedule(
  db: SQLiteDatabase,
  id: string,
  name: string,
): Promise<string | null> {
  const source = await db.getFirstAsync<ScheduleRow>('SELECT * FROM schedules WHERE id = ?', [id]);
  if (!source) return null;
  const newScheduleId = newId('schedule');
  const now = new Date().toISOString();
  await db.withTransactionAsync(async () => {
    const max = await db.getFirstAsync<{ value: number | null }>(
      'SELECT MAX(sort_order) AS value FROM schedules WHERE academic_year_id = ?',
      [source.academic_year_id],
    );
    await db.runAsync(
      `INSERT INTO schedules (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
       VALUES (?, ?, ?, 0, 'scenario', 'duplicate', ?, ?, ?)`,
      [newScheduleId, source.academic_year_id, name, (max?.value ?? 0) + 1, now, now],
    );
    const courses = await db.getAllAsync<CourseRow>(
      'SELECT * FROM courses WHERE schedule_id = ? ORDER BY sort_order ASC, created_at ASC',
      [id],
    );
    for (const course of courses) {
      await db.runAsync(
        `INSERT INTO courses
           (id, title, code, instructor, credits, term, grade, grade_percent, evidence_id, document_id,
            import_id, grading_total_percent, grading_total_consistent, document_type, in_progress,
            academic_year_id, schedule_id, academic_term_id, level, expected_grade, status, include_in_gpa,
            origin, sort_order, edited, original_grade, original_title, original_credits, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 'scenario',
                 ?, 0, NULL, NULL, NULL, ?, ?)`,
        [
          newId('course'),
          course.title,
          course.code,
          course.instructor,
          course.credits,
          course.term,
          course.grade,
          course.grade_percent,
          course.status === 'current' ? 0 : 1,
          course.academic_year_id,
          newScheduleId,
          course.academic_term_id,
          course.level ?? 'regular',
          course.expected_grade,
          course.status === 'completed' ? 'completed' : 'planned',
          course.include_in_gpa ?? 1,
          course.sort_order ?? 0,
          new Date().toISOString(),
          new Date().toISOString(),
        ],
      );
    }
  });
  return newScheduleId;
}

/**
 * Delete a schedule. Its scenario courses go with it — they exist only for the
 * schedule a person made — while a course that came from a document is kept and
 * re-placed, because undoing a plan must never lose the record.
 */
export async function deleteSchedule(db: SQLiteDatabase, id: string): Promise<void> {
  await db.withTransactionAsync(async () => {
    await db.runAsync("DELETE FROM courses WHERE schedule_id = ? AND origin <> 'transcript'", [id]);
    await db.runAsync('UPDATE courses SET schedule_id = NULL, academic_year_id = NULL WHERE schedule_id = ?', [id]);
    await db.runAsync('DELETE FROM schedules WHERE id = ?', [id]);
  });
}

/* ---------------------------------------------------------------------------
 * Pathways
 * ------------------------------------------------------------------------- */

export async function listPathways(db: SQLiteDatabase): Promise<PathwayRow[]> {
  return db.getAllAsync<PathwayRow>('SELECT * FROM pathways ORDER BY created_at ASC');
}

export async function savePathway(
  db: SQLiteDatabase,
  input: { id?: string; name: string; selections: Record<string, string>; active?: boolean },
): Promise<string> {
  const id = input.id ?? newId('pathway');
  const now = new Date().toISOString();
  await db.runAsync(
    `INSERT INTO pathways (id, name, is_active, selections, created_at, updated_at)
     VALUES (?, ?, 0, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET name = excluded.name, selections = excluded.selections,
       updated_at = excluded.updated_at`,
    [id, input.name, JSON.stringify(input.selections), now, now],
  );
  if (input.active) await activatePathway(db, id);
  return id;
}

export async function activatePathway(db: SQLiteDatabase, id: string): Promise<void> {
  await db.withTransactionAsync(async () => {
    await db.runAsync('UPDATE pathways SET is_active = 0');
    await db.runAsync('UPDATE pathways SET is_active = 1 WHERE id = ?', [id]);
  });
}

export async function deletePathway(db: SQLiteDatabase, id: string): Promise<void> {
  const row = await db.getFirstAsync<{ is_active: number }>('SELECT is_active FROM pathways WHERE id = ?', [id]);
  await db.runAsync('DELETE FROM pathways WHERE id = ?', [id]);
  // Deleting the active pathway leaves the product without one; the first
  // remaining pathway becomes active so Home still has a plan to describe.
  if (row?.is_active === 1) {
    const next = await db.getFirstAsync<{ id: string }>('SELECT id FROM pathways ORDER BY created_at ASC LIMIT 1');
    if (next) await activatePathway(db, next.id);
  }
}

/* ---------------------------------------------------------------------------
 * Requirements and scores
 * ------------------------------------------------------------------------- */

export async function saveRequirement(
  db: SQLiteDatabase,
  input: { id?: string; name: string; requiredCredits: number; isTotal?: boolean },
): Promise<string> {
  const now = new Date().toISOString();
  const id = input.id ?? newId('requirement');
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM requirements',
  );
  await db.runAsync(
    `INSERT INTO requirements (id, name, required_credits, is_total, sort_order, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET name = excluded.name, required_credits = excluded.required_credits,
       is_total = excluded.is_total, updated_at = excluded.updated_at`,
    [id, input.name, input.requiredCredits, input.isTotal ? 1 : 0, (max?.value ?? 0) + 1, now, now],
  );
  return id;
}

export async function deleteRequirement(db: SQLiteDatabase, id: string): Promise<void> {
  await db.runAsync('DELETE FROM requirements WHERE id = ?', [id]);
}

export async function setCourseRequirements(
  db: SQLiteDatabase,
  courseId: string,
  requirementIds: string[],
): Promise<void> {
  await db.withTransactionAsync(async () => {
    await db.runAsync('DELETE FROM course_requirements WHERE course_id = ?', [courseId]);
    for (const requirementId of requirementIds) {
      await db.runAsync('INSERT OR IGNORE INTO course_requirements (course_id, requirement_id) VALUES (?, ?)', [
        courseId,
        requirementId,
      ]);
    }
  });
}

export async function saveTestScore(
  db: SQLiteDatabase,
  input: {
    id?: string;
    kind: string;
    name: string;
    takenOn?: string | null;
    totalScore?: string | null;
    maxScore?: string | null;
    sections?: TestSection[];
    notes?: string | null;
  },
): Promise<string> {
  const now = new Date().toISOString();
  const id = input.id ?? newId('test');
  await db.runAsync(
    `INSERT INTO test_scores (id, kind, name, taken_on, total_score, max_score, sections, notes, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, name = excluded.name, taken_on = excluded.taken_on,
       total_score = excluded.total_score, max_score = excluded.max_score, sections = excluded.sections,
       notes = excluded.notes, updated_at = excluded.updated_at`,
    [
      id,
      input.kind,
      input.name,
      input.takenOn ?? null,
      input.totalScore ?? null,
      input.maxScore ?? null,
      JSON.stringify(input.sections ?? []),
      input.notes ?? null,
      now,
      now,
    ],
  );
  return id;
}

export async function deleteTestScore(db: SQLiteDatabase, id: string): Promise<void> {
  await db.runAsync('DELETE FROM test_scores WHERE id = ?', [id]);
}

/* ---------------------------------------------------------------------------
 * Editing one course
 * ------------------------------------------------------------------------- */

export interface CoursePatch {
  title?: string;
  code?: string | null;
  credits?: number | null;
  grade?: string | null;
  expectedGrade?: string | null;
  level?: string;
  status?: string;
  includeInGpa?: boolean;
  academicYearId?: string | null;
  scheduleId?: string | null;
  academicTermId?: string | null;
}

/**
 * Edit a course, keeping what the transcript originally said.
 *
 * The first edit records the imported value in `original_*`, so the source sheet
 * can say "Transcript said 90" long after 91 was typed. A course that came from a
 * document keeps its evidence and its document; only the value changes.
 */
export async function updateCourse(db: SQLiteDatabase, id: string, patch: CoursePatch): Promise<void> {
  const row = await db.getFirstAsync<CourseRow>('SELECT * FROM courses WHERE id = ?', [id]);
  if (!row) return;
  const edited = row.edited === 1;
  const original = {
    grade: edited ? row.original_grade : row.grade,
    title: edited ? row.original_title : row.title,
    credits: edited ? row.original_credits : row.credits,
  };
  const next = {
    title: patch.title ?? row.title,
    code: patch.code === undefined ? row.code : patch.code,
    credits: patch.credits === undefined ? row.credits : patch.credits,
    grade: patch.grade === undefined ? row.grade : patch.grade,
    expectedGrade: patch.expectedGrade === undefined ? row.expected_grade : patch.expectedGrade,
    level: patch.level ?? row.level ?? 'regular',
    status: patch.status ?? row.status,
    includeInGpa: patch.includeInGpa === undefined ? row.include_in_gpa : patch.includeInGpa ? 1 : 0,
    year: patch.academicYearId === undefined ? row.academic_year_id : patch.academicYearId,
    schedule: patch.scheduleId === undefined ? row.schedule_id : patch.scheduleId,
    term: patch.academicTermId === undefined ? row.academic_term_id : patch.academicTermId,
  };
  await db.runAsync(
    `UPDATE courses SET title = ?, code = ?, credits = ?, grade = ?, expected_grade = ?, level = ?,
       status = ?, include_in_gpa = ?, academic_year_id = ?, schedule_id = ?, academic_term_id = ?,
       edited = 1, original_grade = ?, original_title = ?, original_credits = ?, updated_at = ?
     WHERE id = ?`,
    [
      next.title,
      next.code,
      next.credits,
      next.grade,
      next.expectedGrade,
      next.level,
      next.status,
      next.includeInGpa,
      next.year,
      next.schedule,
      next.term,
      original.grade,
      original.title,
      original.credits,
      new Date().toISOString(),
      id,
    ],
  );
}

/** Add a course to a schedule. A manual course carries no document or evidence. */
export async function addCourse(
  db: SQLiteDatabase,
  input: {
    title: string;
    credits: number;
    level: string;
    expectedGrade: string | null;
    grade?: string | null;
    status: 'completed' | 'current' | 'planned';
    academicYearId: string;
    scheduleId: string;
    academicTermId?: string | null;
  },
): Promise<string> {
  const id = newId('course');
  const now = new Date().toISOString();
  const max = await db.getFirstAsync<{ value: number | null }>(
    'SELECT MAX(sort_order) AS value FROM courses WHERE schedule_id = ?',
    [input.scheduleId],
  );
  await db.runAsync(
    `INSERT INTO courses
       (id, title, credits, level, expected_grade, grade, status, include_in_gpa, origin,
        academic_year_id, schedule_id, academic_term_id, sort_order, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'manual', ?, ?, ?, ?, ?, ?)`,
    [
      id,
      input.title,
      input.credits,
      input.level,
      input.expectedGrade,
      input.grade ?? null,
      input.status,
      input.academicYearId,
      input.scheduleId,
      input.academicTermId ?? null,
      (max?.value ?? 0) + 1,
      now,
      now,
    ],
  );
  return id;
}

export async function deleteCourse(db: SQLiteDatabase, id: string): Promise<void> {
  await db.runAsync('DELETE FROM courses WHERE id = ?', [id]);
}

/** Courses of one schedule, in the order the user arranged them. */
export async function listScheduleCourses(db: SQLiteDatabase, scheduleId: string): Promise<CourseRow[]> {
  return db.getAllAsync<CourseRow>(
    `SELECT c.*, (SELECT COUNT(*) FROM grading_categories g WHERE g.course_id = c.id) AS grading_count
     FROM courses c WHERE c.schedule_id = ? ORDER BY c.sort_order ASC, c.created_at ASC`,
    [scheduleId],
  );
}

/* ---------------------------------------------------------------------------
 * First arrival
 * ------------------------------------------------------------------------- */

/**
 * One pathway and one scale exist before the user asks for them, so Home can talk
 * about a plan instead of an empty shell. Called once, from the root layout.
 */
export async function ensureDefaults(db: SQLiteDatabase): Promise<void> {
  const pathways = await db.getFirstAsync<{ count: number }>('SELECT COUNT(*) AS count FROM pathways');
  if (!(pathways?.count ?? 0)) {
    await savePathway(db, { name: 'Main', selections: {}, active: true });
  } else {
    const active = await db.getFirstAsync<{ id: string }>('SELECT id FROM pathways WHERE is_active = 1');
    if (!active) {
      const first = await db.getFirstAsync<{ id: string }>('SELECT id FROM pathways ORDER BY created_at ASC LIMIT 1');
      if (first) await activatePathway(db, first.id);
    }
  }
}
