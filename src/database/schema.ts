/**
 * The local academic store.
 *
 * Two rules shape this schema:
 *
 * 1. Nothing the model produced reaches `courses` without passing review. A
 *    document and an `imports` row are written in the same transaction as the
 *    course, so an import can be undone as one unit.
 * 2. Every imported value keeps the id of the evidence row it came from. The
 *    review screen, the course screen and the "view source" sheet all join back
 *    to `evidence`, which is what makes provenance a feature rather than a claim.
 */
import type { SQLiteDatabase } from 'expo-sqlite';

export const SCHEMA_VERSION = 6;

const STATEMENTS = [
  `PRAGMA journal_mode = WAL;`,
  `PRAGMA foreign_keys = ON;`,
  `CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY NOT NULL,
    document_type TEXT NOT NULL,
    academic INTEGER NOT NULL,
    document_type_confidence REAL,
    source_kind TEXT NOT NULL,
    source_uri TEXT,
    ocr_text TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    warnings TEXT NOT NULL,
    timings TEXT NOT NULL,
    created_at TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS imports (
    id TEXT PRIMARY KEY NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    committed_at TEXT,
    undone_at TEXT,
    course_id TEXT,
    summary TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT,
    code TEXT,
    instructor TEXT,
    credits REAL,
    term TEXT,
    grade TEXT,
    grade_percent REAL,
    evidence_id TEXT,
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    import_id TEXT REFERENCES imports(id) ON DELETE SET NULL,
    grading_total_percent REAL,
    grading_total_consistent INTEGER,
    document_type TEXT,
    in_progress INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS grading_categories (
    id TEXT PRIMARY KEY NOT NULL,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    weight REAL NOT NULL,
    position INTEGER NOT NULL,
    evidence_id TEXT
  );`,
  `CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY NOT NULL,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    days TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    room TEXT,
    alternatives INTEGER NOT NULL DEFAULT 0,
    evidence_id TEXT
  );`,
  `CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY NOT NULL,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    evidence_id TEXT
  );`,
  `CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    page INTEGER,
    text TEXT,
    start INTEGER,
    end INTEGER,
    bbox TEXT,
    ocr_confidence REAL,
    model_confidence REAL,
    confidence_source TEXT NOT NULL,
    extraction_method TEXT NOT NULL
  );`,
  `CREATE INDEX IF NOT EXISTS idx_courses_created ON courses (created_at DESC);`,
  `CREATE INDEX IF NOT EXISTS idx_grading_course ON grading_categories (course_id, position);`,
  `CREATE INDEX IF NOT EXISTS idx_meetings_course ON meetings (course_id);`,
  `CREATE INDEX IF NOT EXISTS idx_policies_course ON policies (course_id);`,
  `CREATE INDEX IF NOT EXISTS idx_evidence_document ON evidence (document_id);`,
  `CREATE INDEX IF NOT EXISTS idx_imports_document ON imports (document_id);`,
  // Used by "Undo import" on every course open, and by the cleanup that follows a
  // deleted document; without them those are table scans on a growing store.
  `CREATE INDEX IF NOT EXISTS idx_imports_course ON imports (course_id);`,
  `CREATE INDEX IF NOT EXISTS idx_grading_evidence ON grading_categories (evidence_id);`,
  `CREATE INDEX IF NOT EXISTS idx_meetings_evidence ON meetings (evidence_id);`,
  `CREATE INDEX IF NOT EXISTS idx_policies_evidence ON policies (evidence_id);`,
  // v2: a tiny key/value table, so a preference is stored the same way as data
  // (and survives an app restart) without adding a storage dependency.
  `CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
  );`,
  // v5: the transcript as an object of its own, not just the courses it created.
  // One row per imported transcript, keyed by its document, so Home can say what
  // the page itself stated (the GPA it printed, the credits it totalled) without
  // re-reading the page or re-deriving a number Pace only read.
  `CREATE TABLE IF NOT EXISTS transcripts (
    document_id TEXT PRIMARY KEY NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    import_id TEXT,
    student_name TEXT,
    school TEXT,
    reported_gpa REAL,
    earned_credits REAL NOT NULL DEFAULT 0,
    in_progress_credits REAL NOT NULL DEFAULT 0,
    course_count INTEGER NOT NULL DEFAULT 0,
    -- The GPA each term printed for itself, keyed by the term's own label. A term
    -- that printed none has no entry, and Pace never invents one for it.
    term_gpas TEXT,
    imported_at TEXT NOT NULL
  );`,
  // v5: a planned course is the user's, not a document's. It carries no evidence
  // and no import, so undo and delete-all can never remove something a person
  // typed — except through the explicit "delete all data" path.
  `CREATE TABLE IF NOT EXISTS planned_courses (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    credits REAL NOT NULL DEFAULT 1,
    level TEXT NOT NULL DEFAULT 'regular',
    expected_grade TEXT NOT NULL DEFAULT 'A',
    term TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  // v6: the academic structure a course belongs to. A year is what a person says
  // ("Grade 10", "2026–27"); a term is one of its parts; a schedule is a set of
  // courses for one year, so an alternative can be tried without touching the
  // record the transcript produced.
  `CREATE TABLE IF NOT EXISTS academic_years (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    school_year_label TEXT,
    grade_level TEXT,
    status TEXT NOT NULL DEFAULT 'future',
    sort_order INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS academic_terms (
    id TEXT PRIMARY KEY NOT NULL,
    academic_year_id TEXT NOT NULL REFERENCES academic_years(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'future',
    sort_order INTEGER NOT NULL DEFAULT 0
  );`,
  `CREATE INDEX IF NOT EXISTS idx_terms_year ON academic_terms (academic_year_id, sort_order);`,
  `CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY NOT NULL,
    academic_year_id TEXT NOT NULL REFERENCES academic_years(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    is_main INTEGER NOT NULL DEFAULT 0,
    type TEXT NOT NULL DEFAULT 'actual',
    source TEXT NOT NULL DEFAULT 'manual',
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `CREATE INDEX IF NOT EXISTS idx_schedules_year ON schedules (academic_year_id, sort_order);`,
  // A pathway picks one schedule per year. The selections are a JSON map of year id
  // to schedule id: a year with no entry falls back to that year's main schedule,
  // which is the rule the projection engine implements.
  `CREATE TABLE IF NOT EXISTS pathways (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    selections TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS requirements (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    required_credits REAL NOT NULL DEFAULT 0,
    is_total INTEGER NOT NULL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `CREATE TABLE IF NOT EXISTS course_requirements (
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    requirement_id TEXT NOT NULL REFERENCES requirements(id) ON DELETE CASCADE,
    PRIMARY KEY (course_id, requirement_id)
  );`,
  `CREATE TABLE IF NOT EXISTS test_scores (
    id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    taken_on TEXT,
    total_score TEXT,
    max_score TEXT,
    sections TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
];

/**
 * Migrations are append-only: version 2 adds statements, it never edits v1.
 * `PRAGMA user_version` is the only bookkeeping, which keeps a fresh install and
 * an upgrade on the same code path.
 */
export async function migrate(db: SQLiteDatabase): Promise<void> {
  for (const statement of STATEMENTS) await db.execAsync(statement);
  // v3 adds a transcript row's own columns, v4 the in-progress flag. SQLite has no
  // "ADD COLUMN IF NOT EXISTS", so an existing store is read first and only the
  // missing columns are added; a fresh install already has them from the CREATE
  // above. A course imported before v4 is a finished one, which is the default.
  const columns = await db.getAllAsync<{ name: string }>('PRAGMA table_info(courses)');
  const present = new Set(columns.map((column) => column.name));
  const added: [string, string][] = [
    ['term', 'TEXT'],
    ['grade', 'TEXT'],
    ['grade_percent', 'REAL'],
    ['evidence_id', 'TEXT'],
    ['in_progress', 'INTEGER NOT NULL DEFAULT 0'],
  ];
  for (const [name, type] of added) {
    if (!present.has(name)) await db.execAsync(`ALTER TABLE courses ADD COLUMN ${name} ${type}`);
  }
  const transcriptColumns = await db.getAllAsync<{ name: string }>('PRAGMA table_info(transcripts)');
  const transcriptPresent = new Set(transcriptColumns.map((column) => column.name));
  if (transcriptColumns.length && !transcriptPresent.has('term_gpas')) {
    await db.execAsync('ALTER TABLE transcripts ADD COLUMN term_gpas TEXT');
  }
  // v6 gives a course its place in the academic structure, its level, and the
  // difference between the mark it was read with and the mark a person corrected.
  const addedCourse: [string, string][] = [
    ['academic_year_id', 'TEXT'],
    ['schedule_id', 'TEXT'],
    ['academic_term_id', 'TEXT'],
    ['level', "TEXT NOT NULL DEFAULT 'regular'"],
    ['expected_grade', 'TEXT'],
    ['status', "TEXT NOT NULL DEFAULT 'completed'"],
    ['include_in_gpa', 'INTEGER NOT NULL DEFAULT 1'],
    ['origin', "TEXT NOT NULL DEFAULT 'transcript'"],
    ['sort_order', 'INTEGER NOT NULL DEFAULT 0'],
    ['edited', 'INTEGER NOT NULL DEFAULT 0'],
    ['original_grade', 'TEXT'],
    ['original_title', 'TEXT'],
    ['original_credits', 'REAL'],
  ];
  for (const [name, type] of addedCourse) {
    if (!present.has(name)) await db.execAsync(`ALTER TABLE courses ADD COLUMN ${name} ${type}`);
  }
  await backfillStructure(db);
  const row = await db.getFirstAsync<{ user_version: number }>('PRAGMA user_version');
  const current = row?.user_version ?? 0;
  if (current < SCHEMA_VERSION) {
    await db.execAsync(`PRAGMA user_version = ${SCHEMA_VERSION}`);
  }
}

/**
 * Give courses imported before v6 a year and a schedule, and move a planned course
 * into one too.
 *
 * The rule is conservative on purpose: a course that Pace cannot place keeps its
 * row and lands in a bucket named for what it is, because losing a course to a
 * schema change would be worse than an untidy year name. Runs only when a course
 * has no place yet, so it is safe on every launch.
 */
async function backfillStructure(db: SQLiteDatabase): Promise<void> {
  const orphans = await db.getAllAsync<{
    id: string;
    in_progress: number;
    term: string | null;
  }>('SELECT id, in_progress, term FROM courses WHERE academic_year_id IS NULL');
  if (orphans.length) {
    const now = new Date().toISOString();
    const ensure = async (
      id: string,
      name: string,
      status: 'past' | 'current' | 'future',
      scheduleId: string,
      scheduleName: string,
      type: 'actual' | 'planned',
      sortOrder: number,
    ) => {
      await db.runAsync(
        `INSERT OR IGNORE INTO academic_years
           (id, name, school_year_label, grade_level, status, sort_order, source, created_at, updated_at)
         VALUES (?, ?, NULL, NULL, ?, ?, 'transcript', ?, ?)`,
        [id, name, status, sortOrder, now, now],
      );
      await db.runAsync(
        `INSERT OR IGNORE INTO schedules
           (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
         VALUES (?, ?, ?, 1, ?, 'transcript', 0, ?, ?)`,
        [scheduleId, id, scheduleName, type, now, now],
      );
    };
    await ensure('year-imported', 'Imported', 'past', 'schedule-imported', 'Actual', 'actual', 0);
    await ensure('year-current', 'Current', 'current', 'schedule-current', 'Current', 'actual', 1);
    for (const course of orphans) {
      await db.runAsync(
        `UPDATE courses SET academic_year_id = ?, schedule_id = ?, status = ? WHERE id = ?`,
        course.in_progress === 1
          ? ['year-current', 'schedule-current', 'current', course.id]
          : ['year-imported', 'schedule-imported', 'completed', course.id],
      );
    }
  }
  const planned = await db.getAllAsync<{ count: number }>('SELECT COUNT(*) AS count FROM planned_courses');
  if (!(planned[0]?.count ?? 0)) return;
  const moved = await db.getAllAsync<{ count: number }>(
    "SELECT COUNT(*) AS count FROM courses WHERE origin = 'manual'",
  );
  if (moved[0]?.count ?? 0) return;
  const now = new Date().toISOString();
  await db.runAsync(
    `INSERT OR IGNORE INTO academic_years
       (id, name, school_year_label, grade_level, status, sort_order, source, created_at, updated_at)
     VALUES ('year-planned', 'Planned', NULL, NULL, 'future', 2, 'manual', ?, ?)`,
    [now, now],
  );
  await db.runAsync(
    `INSERT OR IGNORE INTO schedules
       (id, academic_year_id, name, is_main, type, source, sort_order, created_at, updated_at)
     VALUES ('schedule-planned', 'year-planned', 'Main', 1, 'planned', 'manual', 0, ?, ?)`,
    [now, now],
  );
  await db.runAsync(
    `INSERT INTO courses
       (id, title, credits, level, expected_grade, status, origin, academic_year_id, schedule_id,
        include_in_gpa, sort_order, created_at, updated_at)
     SELECT id, name, credits, level, expected_grade, 'planned', 'manual', 'year-planned',
            'schedule-planned', 1, 0, created_at, updated_at
     FROM planned_courses
     WHERE id NOT IN (SELECT id FROM courses)`,
  );
}
