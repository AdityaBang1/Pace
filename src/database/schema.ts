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

export const SCHEMA_VERSION = 2;

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
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    import_id TEXT REFERENCES imports(id) ON DELETE SET NULL,
    grading_total_percent REAL,
    grading_total_consistent INTEGER,
    document_type TEXT,
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
];

/**
 * Migrations are append-only: version 2 adds statements, it never edits v1.
 * `PRAGMA user_version` is the only bookkeeping, which keeps a fresh install and
 * an upgrade on the same code path.
 */
export async function migrate(db: SQLiteDatabase): Promise<void> {
  for (const statement of STATEMENTS) await db.execAsync(statement);
  const row = await db.getFirstAsync<{ user_version: number }>('PRAGMA user_version');
  const current = row?.user_version ?? 0;
  if (current < SCHEMA_VERSION) {
    await db.execAsync(`PRAGMA user_version = ${SCHEMA_VERSION}`);
  }
}
