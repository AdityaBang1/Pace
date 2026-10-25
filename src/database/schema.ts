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

export const SCHEMA_VERSION = 5;

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
  const row = await db.getFirstAsync<{ user_version: number }>('PRAGMA user_version');
  const current = row?.user_version ?? 0;
  if (current < SCHEMA_VERSION) {
    await db.execAsync(`PRAGMA user_version = ${SCHEMA_VERSION}`);
  }
}
