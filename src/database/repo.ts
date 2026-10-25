/**
 * Reads and writes against the local store.
 *
 * The only multi-row write is `commitImport`, and it runs as one transaction:
 * either the document, the import record, the course and every proposed value
 * land together, or nothing does. The review screen cannot half-import a
 * syllabus, and "Undo import" can remove exactly the rows an import created.
 */
import type { SQLiteDatabase } from 'expo-sqlite';

export interface DocumentInput {
  id: string;
  documentType: string;
  academic: boolean;
  documentTypeConfidence: number | null;
  sourceKind: 'photo' | 'library' | 'file' | 'text';
  sourceUri: string | null;
  ocrText: string;
  engineVersion: string;
  schemaVersion: string;
  characterCount: number;
  warnings: string[];
  timings: Record<string, number>;
}

/** One printed transcript row, committed as its own course. */
export interface TranscriptCourseInput {
  id: string;
  title: string;
  code: string | null;
  term: string | null;
  grade: string | null;
  gradePercent: number | null;
  credits: number | null;
  evidenceId: string | null;
  /** Printed under an "In-Progress Courses" heading: a credit, no mark yet. */
  inProgress: boolean;
}

export interface CourseInput {
  title: string | null;
  code: string | null;
  instructor: string | null;
  credits: number | null;
  gradingTotalPercent: number | null;
  gradingTotalConsistent: boolean | null;
  documentType: string;
}

export interface GradingInput {
  id: string;
  name: string;
  weight: number;
  position: number;
  evidenceId: string | null;
}

export interface MeetingInput {
  id: string;
  days: string[];
  startTime: string | null;
  endTime: string | null;
  room: string | null;
  alternatives: boolean;
  evidenceId: string | null;
}

export interface PolicyInput {
  id: string;
  field: string;
  value: string;
  evidenceId: string | null;
}

export interface EvidenceInput {
  id: string;
  field: string;
  page: number | null;
  text: string | null;
  start: number | null;
  end: number | null;
  bbox: { x: number; y: number; width: number; height: number } | null;
  ocrConfidence: number | null;
  modelConfidence: number | null;
  confidenceSource: 'rule' | 'model';
  extractionMethod: string;
}

export interface CommitInput {
  importId: string;
  /** Reuse an existing course instead of creating one. */
  linkToCourseId: string | null;
  courseId: string;
  course: CourseInput | null;
  /** The courses a transcript import creates, one per printed row. */
  transcriptCourses?: TranscriptCourseInput[] | null;
  /** The transcript's own summary, recorded beside the courses it created. */
  transcript?: TranscriptInput | null;
  document: DocumentInput;
  grading: GradingInput[];
  meetings: MeetingInput[];
  policies: PolicyInput[];
  evidence: EvidenceInput[];
  summary: Record<string, unknown>;
}

export interface CourseRow {
  id: string;
  title: string | null;
  code: string | null;
  instructor: string | null;
  credits: number | null;
  /** Printed on a transcript row; null for a syllabus. */
  term: string | null;
  grade: string | null;
  grade_percent: number | null;
  /** 1 when the row is a course the student is taking now. */
  in_progress: number;
  /** The printed row a transcript course was read from. */
  evidence_id: string | null;
  document_id: string | null;
  import_id: string | null;
  grading_total_percent: number | null;
  grading_total_consistent: number | null;
  document_type: string | null;
  created_at: string;
  updated_at: string;
  /** Present on `listCourses` only: how many grading rows the course has. */
  grading_count?: number;
}

export interface GradingRow {
  id: string;
  name: string;
  weight: number;
  evidence_id: string | null;
}

export interface MeetingRow {
  id: string;
  days: string;
  start_time: string | null;
  end_time: string | null;
  room: string | null;
  alternatives: number;
  evidence_id: string | null;
}

export interface PolicyRow {
  id: string;
  field: string;
  value: string;
  evidence_id: string | null;
}

export interface EvidenceRow {
  id: string;
  field: string;
  page: number | null;
  text: string | null;
  start: number | null;
  end: number | null;
  bbox: string | null;
  model_confidence: number | null;
  confidence_source: string;
  extraction_method: string;
}

export interface ImportRow {
  id: string;
  document_id: string;
  status: 'committed' | 'undone';
  created_at: string;
  committed_at: string | null;
  undone_at: string | null;
  course_id: string | null;
  summary: string;
  document_type: string | null;
  course_title: string | null;
}

export interface CourseDetail {
  course: CourseRow;
  grading: GradingRow[];
  meetings: MeetingRow[];
  policies: PolicyRow[];
  evidence: EvidenceRow[];
  document: {
    id: string;
    document_type: string;
    source_uri: string | null;
    created_at: string;
    ocr_text: string;
  } | null;
}

export async function commitImport(db: SQLiteDatabase, input: CommitInput): Promise<void> {
  const now = new Date().toISOString();
  // The gate is the product's safety promise, so it is enforced here as well as in
  // the review screen: a caller that skips review still cannot turn a receipt into
  // a course.
  if (!input.document.academic) {
    throw new Error('This document was not read as academic, so there is nothing to import.');
  }
  // A transcript import names no single course: its courses are one per row.
  const courseId = input.transcriptCourses?.length
    ? null
    : input.linkToCourseId ?? input.courseId;
  await db.withTransactionAsync(async () => {
    // Re-importing the same page after an undo must not trip the primary key.
    // An upsert keeps the document's identity stable instead of REPLACE, which
    // would delete the row and cascade away the earlier import record.
    await db.runAsync(
      `INSERT INTO documents (
        id, document_type, academic, document_type_confidence, source_kind, source_uri,
        ocr_text, engine_version, schema_version, character_count, warnings, timings, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        document_type = excluded.document_type,
        academic = excluded.academic,
        document_type_confidence = excluded.document_type_confidence,
        source_kind = excluded.source_kind,
        source_uri = excluded.source_uri,
        ocr_text = excluded.ocr_text,
        engine_version = excluded.engine_version,
        schema_version = excluded.schema_version,
        character_count = excluded.character_count,
        warnings = excluded.warnings,
        timings = excluded.timings`,
      [
        input.document.id,
        input.document.documentType,
        input.document.academic ? 1 : 0,
        input.document.documentTypeConfidence,
        input.document.sourceKind,
        input.document.sourceUri,
        input.document.ocrText,
        input.document.engineVersion,
        input.document.schemaVersion,
        input.document.characterCount,
        JSON.stringify(input.document.warnings),
        JSON.stringify(input.document.timings),
        now,
      ],
    );

    // The import row comes *before* the course: `courses.import_id` references it
    // and SQLite checks that reference immediately, so the other order fails with
    // `FOREIGN KEY constraint failed` on a connection with foreign keys on (which
    // this app turns on). `imports.course_id` deliberately has no constraint, so a
    // row may name a course that is inserted a statement later.
    await db.runAsync(
      `INSERT INTO imports (id, document_id, status, created_at, committed_at, course_id, summary)
       VALUES (?, ?, 'committed', ?, ?, ?, ?)`,
      [input.importId, input.document.id, now, now, courseId, JSON.stringify(input.summary)],
    );

    // Evidence ids are content-derived, so a re-import reproduces the same rows.
    // Clearing first keeps a second import from failing on those ids.
    await db.runAsync('DELETE FROM evidence WHERE document_id = ?', [input.document.id]);

    if (input.linkToCourseId) {
      // A link never rewrites the existing course's identity; it fills blanks and
      // attaches what the new document knows about.
      await db.runAsync(
        `UPDATE courses SET
           instructor = COALESCE(instructor, ?),
           credits = COALESCE(credits, ?),
           updated_at = ?
         WHERE id = ?`,
        [input.course?.instructor ?? null, input.course?.credits ?? null, now, input.linkToCourseId],
      );
    } else if (input.transcriptCourses?.length) {
      // A transcript is many courses. Every row created by this import carries
      // the import id, which is what Undo removes as one unit.
      for (const course of input.transcriptCourses) {
        await db.runAsync(
          `INSERT INTO courses (
            id, title, code, credits, term, grade, grade_percent, evidence_id,
            document_id, import_id, grading_total_percent, grading_total_consistent,
            document_type, in_progress, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)`,
          [
            course.id,
            course.title,
            course.code,
            course.credits,
            course.term,
            course.grade,
            course.gradePercent,
            course.evidenceId,
            input.document.id,
            input.importId,
            input.document.documentType,
            course.inProgress ? 1 : 0,
            now,
            now,
          ],
        );
      }
    } else {
      const course = input.course;
      await db.runAsync(
        `INSERT INTO courses (
          id, title, code, instructor, credits, document_id, import_id,
          grading_total_percent, grading_total_consistent, document_type, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          input.courseId,
          course?.title ?? null,
          course?.code ?? null,
          course?.instructor ?? null,
          course?.credits ?? null,
          input.document.id,
          input.importId,
          course?.gradingTotalPercent ?? null,
          course?.gradingTotalConsistent == null ? null : course.gradingTotalConsistent ? 1 : 0,
          course?.documentType ?? input.document.documentType,
          now,
          now,
        ],
      );
    }

    for (const row of input.grading) {
      await db.runAsync(
        `INSERT INTO grading_categories (id, course_id, name, weight, position, evidence_id)
         VALUES (?, ?, ?, ?, ?, ?)`,
        [row.id, courseId, row.name, row.weight, row.position, row.evidenceId],
      );
    }
    for (const row of input.meetings) {
      await db.runAsync(
        `INSERT INTO meetings (id, course_id, days, start_time, end_time, room, alternatives, evidence_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          row.id,
          courseId,
          JSON.stringify(row.days),
          row.startTime,
          row.endTime,
          row.room,
          row.alternatives ? 1 : 0,
          row.evidenceId,
        ],
      );
    }
    for (const row of input.policies) {
      await db.runAsync(
        `INSERT INTO policies (id, course_id, field, value, evidence_id) VALUES (?, ?, ?, ?, ?)`,
        [row.id, courseId, row.field, row.value, row.evidenceId],
      );
    }
    for (const row of input.evidence) {
      await db.runAsync(
        `INSERT INTO evidence (
          id, document_id, field, page, text, start, end, bbox, ocr_confidence,
          model_confidence, confidence_source, extraction_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          row.id,
          input.document.id,
          row.field,
          row.page,
          row.text,
          row.start,
          row.end,
          row.bbox ? JSON.stringify(row.bbox) : null,
          row.ocrConfidence,
          row.modelConfidence,
          row.confidenceSource,
          row.extractionMethod,
        ],
      );
    }

    if (input.transcript) await saveTranscript(db, input.transcript);
  });
}

export async function listCourses(db: SQLiteDatabase): Promise<CourseRow[]> {
  return db.getAllAsync<CourseRow>(
    `SELECT c.*,
            (SELECT COUNT(*) FROM grading_categories g WHERE g.course_id = c.id) AS grading_count
     FROM courses c ORDER BY c.created_at DESC`,
  );
}

export async function courseCount(db: SQLiteDatabase): Promise<number> {
  const row = await db.getFirstAsync<{ count: number }>('SELECT COUNT(*) AS count FROM courses');
  return row?.count ?? 0;
}

export async function getCourse(db: SQLiteDatabase, id: string): Promise<CourseDetail | null> {
  const course = await db.getFirstAsync<CourseRow>('SELECT * FROM courses WHERE id = ?', [id]);
  if (!course) return null;
  const grading = await db.getAllAsync<GradingRow>(
    'SELECT id, name, weight, evidence_id FROM grading_categories WHERE course_id = ? ORDER BY position ASC',
    [id],
  );
  const meetings = await db.getAllAsync<MeetingRow>(
    'SELECT id, days, start_time, end_time, room, alternatives, evidence_id FROM meetings WHERE course_id = ?',
    [id],
  );
  const policies = await db.getAllAsync<PolicyRow>(
    'SELECT id, field, value, evidence_id FROM policies WHERE course_id = ? ORDER BY field ASC',
    [id],
  );
  // Evidence for this course's own document, plus any evidence a *linked* import
  // attached to it. Without the second half, rows imported from a schedule into a
  // syllabus's course are tappable and lead nowhere.
  const evidence = await db.getAllAsync<EvidenceRow>(
    `SELECT id, field, page, text, start, end, bbox, model_confidence, confidence_source, extraction_method
     FROM evidence
     WHERE document_id = ?
        OR id IN (
          SELECT evidence_id FROM grading_categories WHERE course_id = ? AND evidence_id IS NOT NULL
          UNION SELECT evidence_id FROM meetings WHERE course_id = ? AND evidence_id IS NOT NULL
          UNION SELECT evidence_id FROM policies WHERE course_id = ? AND evidence_id IS NOT NULL
          UNION SELECT evidence_id FROM courses WHERE id = ? AND evidence_id IS NOT NULL
        )`,
    [course.document_id ?? '', id, id, id, id],
  );
  const document = course.document_id
    ? await db.getFirstAsync<{
        id: string;
        document_type: string;
        source_uri: string | null;
        created_at: string;
        ocr_text: string;
      }>('SELECT id, document_type, source_uri, created_at, ocr_text FROM documents WHERE id = ?', [
        course.document_id,
      ])
    : null;
  return { course, grading, meetings, policies, evidence, document };
}

/** The course name recorded in `imports.summary`, for rows whose course is gone. */
function summaryCourse(summary: string): string | null {
  try {
    const parsed = JSON.parse(summary) as { course?: string | null };
    return typeof parsed.course === 'string' && parsed.course.trim() ? parsed.course : null;
  } catch {
    return null;
  }
}

export async function listImports(db: SQLiteDatabase, limit = 10): Promise<ImportRow[]> {
  const rows = await db.getAllAsync<ImportRow>(
    `SELECT i.*, d.document_type AS document_type, c.title AS course_title
     FROM imports i
     LEFT JOIN documents d ON d.id = i.document_id
     LEFT JOIN courses c ON c.id = i.course_id
     ORDER BY i.created_at DESC LIMIT ?`,
    [limit],
  );
  // An undone import's course row is gone, but the summary still remembers the
  // name: history should read "MATH 1A · undone", not "Unnamed document".
  return rows.map((row) =>
    row.course_title ? row : { ...row, course_title: summaryCourse(row.summary) },
  );
}

/** The import that created a course, for "Undo import". */
export async function importForCourse(
  db: SQLiteDatabase,
  courseId: string,
): Promise<{ id: string; status: string } | null> {
  // The import recorded on the course itself is the one Undo must remove: after a
  // linked import is undone, "newest committed" could name a link instead and the
  // button would leave the creating import alive. The fallback keeps older rows
  // (imported before the column existed) undoable.
  const course = await db.getFirstAsync<{ import_id: string | null }>(
    'SELECT import_id FROM courses WHERE id = ?',
    [courseId],
  );
  if (course?.import_id) {
    const created = await db.getFirstAsync<{ id: string; status: string }>(
      'SELECT id, status FROM imports WHERE id = ?',
      [course.import_id],
    );
    if (created?.status === 'committed') return created;
  }
  return db.getFirstAsync<{ id: string; status: string }>(
    `SELECT id, status FROM imports WHERE course_id = ? AND status = 'committed'
     ORDER BY created_at DESC LIMIT 1`,
    [courseId],
  );
}

/** The import that already claimed this exact document, if there is one. */
export async function committedImportForDocument(
  db: SQLiteDatabase,
  documentId: string,
): Promise<{ id: string; course_id: string | null } | null> {
  return db.getFirstAsync<{ id: string; course_id: string | null }>(
    `SELECT id, course_id FROM imports WHERE document_id = ? AND status = 'committed' ORDER BY created_at DESC LIMIT 1`,
    [documentId],
  );
}

export async function getEvidence(db: SQLiteDatabase, id: string): Promise<EvidenceRow | null> {
  return db.getFirstAsync<EvidenceRow>(
    `SELECT id, field, page, text, start, end, bbox, model_confidence, confidence_source, extraction_method
     FROM evidence WHERE id = ?`,
    [id],
  );
}

export interface MatchCandidate {
  id: string;
  title: string | null;
  code: string | null;
}

export async function matchCandidates(db: SQLiteDatabase): Promise<MatchCandidate[]> {
  return db.getAllAsync<MatchCandidate>('SELECT id, title, code FROM courses ORDER BY created_at DESC');
}

/**
 * Undo one import.
 *
 * A course created by the import is deleted (its categories, meetings and
 * policies cascade). A course the import was *linked* to survives, and only the
 * rows this document produced are removed, identified through their evidence.
 * The document itself stays: the import history is a record of what was read,
 * and discarding source text is a separate, explicit action.
 */
export async function undoImport(db: SQLiteDatabase, importId: string): Promise<void> {
  const row = await db.getFirstAsync<{ document_id: string; course_id: string | null }>(
    'SELECT document_id, course_id FROM imports WHERE id = ?',
    [importId],
  );
  if (!row) return;
  const now = new Date().toISOString();
  // Every course this import created, not just the one its `course_id` names: a
  // transcript import creates one course per row and its import row names none.
  const created = await db.getAllAsync<{ id: string }>(
    'SELECT id FROM courses WHERE import_id = ?',
    [importId],
  );

  await db.withTransactionAsync(async () => {
    if (created.length) {
      for (const course of created) {
        await db.runAsync('DELETE FROM courses WHERE id = ?', [course.id]);
      }
    } else if (row.course_id) {
      // Two identifiers, because one is not enough: rows this import wrote carry
      // ids derived from the import id, while a value the reviewer typed has no
      // evidence at all. Keying on evidence alone left those rows on the course
      // after the app said the import was undone.
      const prefix = `${importId}-%`;
      for (const table of ['grading_categories', 'meetings', 'policies']) {
        await db.runAsync(`DELETE FROM ${table} WHERE course_id = ? AND id LIKE ?`, [row.course_id, prefix]);
        await db.runAsync(
          `DELETE FROM ${table} WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)`,
          [row.document_id],
        );
      }
      await refreshGradingTotal(db, row.course_id, now);
    }
    await db.runAsync(`UPDATE imports SET status = 'undone', undone_at = ? WHERE id = ?`, [now, importId]);
  });
}

/** Recompute the denormalised total after rows were added or removed. */
async function refreshGradingTotal(db: SQLiteDatabase, courseId: string, now: string): Promise<void> {
  const totals = await db.getFirstAsync<{ total: number | null; n: number }>(
    'SELECT SUM(weight) AS total, COUNT(*) AS n FROM grading_categories WHERE course_id = ?',
    [courseId],
  );
  const total = totals?.n ? Math.round((totals.total ?? 0) * 100) / 100 : null;
  await db.runAsync(
    `UPDATE courses SET grading_total_percent = ?, grading_total_consistent = ?, updated_at = ? WHERE id = ?`,
    [total, total == null ? null : Math.abs(total - 100) <= 5 ? 1 : 0, now, courseId],
  );
}

/**
 * Remove a stored document and its evidence. Courses keep their identity.
 *
 * The rows extracted from that document go too: their `evidence_id` would
 * otherwise point at a row that no longer exists, and "view source" would fail
 * silently instead of saying the source is gone.
 */
export async function deleteDocument(db: SQLiteDatabase, documentId: string): Promise<void> {
  const affected = await db.getAllAsync<{ id: string }>(
    `SELECT DISTINCT course_id AS id FROM grading_categories
       WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)
     UNION SELECT DISTINCT course_id FROM meetings
       WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)
     UNION SELECT DISTINCT course_id FROM policies
       WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)`,
    [documentId, documentId, documentId],
  );
  const now = new Date().toISOString();
  await db.withTransactionAsync(async () => {
    for (const table of ['grading_categories', 'meetings', 'policies']) {
      await db.runAsync(
        `DELETE FROM ${table} WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)`,
        [documentId],
      );
    }
    // A transcript course keeps its identity but loses the row it was read from.
    await db.runAsync(
      'UPDATE courses SET evidence_id = NULL WHERE evidence_id IN (SELECT id FROM evidence WHERE document_id = ?)',
      [documentId],
    );
    await db.runAsync('DELETE FROM documents WHERE id = ?', [documentId]);
    for (const course of affected) await refreshGradingTotal(db, course.id, now);
  });
}

/** The stored page or PDF a document points at, for deleting it from disk. */
export async function sourceUriForImport(db: SQLiteDatabase, importId: string): Promise<string | null> {
  const row = await db.getFirstAsync<{ source_uri: string | null }>(
    `SELECT d.source_uri FROM imports i JOIN documents d ON d.id = i.document_id WHERE i.id = ?`,
    [importId],
  );
  return row?.source_uri ?? null;
}

/** Every stored page or PDF, for "Delete all data". */
export async function allSourceUris(db: SQLiteDatabase): Promise<string[]> {
  const rows = await db.getAllAsync<{ source_uri: string | null }>(
    'SELECT source_uri FROM documents WHERE source_uri IS NOT NULL',
  );
  return rows.map((row) => row.source_uri).filter((uri): uri is string => Boolean(uri));
}

/**
 * Wipe everything Pace has stored. Only reachable from Settings, after a confirm.
 *
 * The `settings` row survives on purpose: deleting academic data should not reset
 * the user's storage preference. Callers delete the page and PDF files themselves
 * (`allSourceUris`) and clear those directories.
 */
export async function deleteAllData(db: SQLiteDatabase): Promise<void> {
  await db.withTransactionAsync(async () => {
    for (const table of [
      'policies',
      'meetings',
      'grading_categories',
      'courses',
      'planned_courses',
      'transcripts',
      'imports',
      'evidence',
      'documents',
    ]) {
      await db.runAsync(`DELETE FROM ${table}`);
    }
  });
}

/** A stored preference. The fallback is returned when the key has never been set. */
export async function getSetting(db: SQLiteDatabase, key: string, fallback: string): Promise<string> {
  const row = await db.getFirstAsync<{ value: string }>('SELECT value FROM settings WHERE key = ?', [key]);
  return row?.value ?? fallback;
}

export async function setSetting(db: SQLiteDatabase, key: string, value: string): Promise<void> {
  await db.runAsync(
    'INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
    [key, value],
  );
}

/** The storage policy the scanner follows. `true` keeps the page file for provenance. */
export const KEEP_SOURCE_KEY = 'keep_source';

export async function keepSourceEnabled(db: SQLiteDatabase): Promise<boolean> {
  return (await getSetting(db, KEEP_SOURCE_KEY, 'true')) !== 'false';
}

export async function storedDocumentCount(db: SQLiteDatabase): Promise<number> {
  const row = await db.getFirstAsync<{ count: number }>('SELECT COUNT(*) AS count FROM documents');
  return row?.count ?? 0;
}

/* ---------------------------------------------------------------------------
 * v5 — the transcript as an object, and the courses a person plans
 * ------------------------------------------------------------------------- */

/** What the imported transcript page stated, as the page stated it. */
export interface TranscriptRow {
  document_id: string;
  import_id: string | null;
  student_name: string | null;
  school: string | null;
  reported_gpa: number | null;
  earned_credits: number;
  in_progress_credits: number;
  course_count: number;
  /** JSON: term label → the GPA that term printed. Null when no term printed one. */
  term_gpas: string | null;
  imported_at: string;
}

export interface TranscriptInput {
  documentId: string;
  importId: string;
  student: string | null;
  school: string | null;
  reportedGpa: number | null;
  earnedCredits: number;
  inProgressCredits: number;
  courseCount: number;
  termGpas: Record<string, number> | null;
}

/** The term GPAs a transcript printed, keyed by term label. Never recomputed. */
export function termGpasOf(row: TranscriptRow | null | undefined): Record<string, number> {
  if (!row?.term_gpas) return {};
  try {
    const parsed: unknown = JSON.parse(row.term_gpas);
    if (!parsed || typeof parsed !== 'object') return {};
    const out: Record<string, number> = {};
    for (const [term, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === 'number' && Number.isFinite(value)) out[term] = value;
    }
    return out;
  } catch {
    return {};
  }
}

/** Written in the import's own transaction, so a transcript is never half-recorded. */
export async function saveTranscript(db: SQLiteDatabase, input: TranscriptInput): Promise<void> {
  await db.runAsync(
    `INSERT INTO transcripts (
      document_id, import_id, student_name, school, reported_gpa,
      earned_credits, in_progress_credits, course_count, term_gpas, imported_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(document_id) DO UPDATE SET
      import_id = excluded.import_id,
      student_name = excluded.student_name,
      school = excluded.school,
      reported_gpa = excluded.reported_gpa,
      earned_credits = excluded.earned_credits,
      in_progress_credits = excluded.in_progress_credits,
      course_count = excluded.course_count,
      term_gpas = excluded.term_gpas,
      imported_at = excluded.imported_at`,
    [
      input.documentId,
      input.importId,
      input.student,
      input.school,
      input.reportedGpa,
      input.earnedCredits,
      input.inProgressCredits,
      input.courseCount,
      input.termGpas && Object.keys(input.termGpas).length ? JSON.stringify(input.termGpas) : null,
      new Date().toISOString(),
    ],
  );
}

/** The most recently imported transcript, which is the student's current record. */
export async function latestTranscript(db: SQLiteDatabase): Promise<TranscriptRow | null> {
  return db.getFirstAsync<TranscriptRow>('SELECT * FROM transcripts ORDER BY imported_at DESC LIMIT 1');
}

/** A course the user is planning. It has no document and no evidence behind it. */
export interface PlannedCourseRow {
  id: string;
  name: string;
  credits: number;
  level: 'regular' | 'honors' | 'ap';
  expected_grade: string;
  term: string | null;
  created_at: string;
  updated_at: string;
}

export interface PlannedCourseInput {
  id: string;
  name: string;
  credits: number;
  level: 'regular' | 'honors' | 'ap';
  expectedGrade: string;
  term: string | null;
}

export async function listPlannedCourses(db: SQLiteDatabase): Promise<PlannedCourseRow[]> {
  return db.getAllAsync<PlannedCourseRow>('SELECT * FROM planned_courses ORDER BY created_at ASC');
}

export async function upsertPlannedCourse(db: SQLiteDatabase, input: PlannedCourseInput): Promise<void> {
  const now = new Date().toISOString();
  await db.runAsync(
    `INSERT INTO planned_courses (id, name, credits, level, expected_grade, term, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       name = excluded.name,
       credits = excluded.credits,
       level = excluded.level,
       expected_grade = excluded.expected_grade,
       term = excluded.term,
       updated_at = excluded.updated_at`,
    [input.id, input.name, input.credits, input.level, input.expectedGrade, input.term, now, now],
  );
}

export async function deletePlannedCourse(db: SQLiteDatabase, id: string): Promise<void> {
  await db.runAsync('DELETE FROM planned_courses WHERE id = ?', [id]);
}

/** The GPA scale, once the user has set one. `null` means no projection is shown. */
export const GPA_SCALE_KEY = 'gpa_scale';

export async function readGpaScaleValue(db: SQLiteDatabase): Promise<string | null> {
  const value = await getSetting(db, GPA_SCALE_KEY, '');
  return value ? value : null;
}

export async function writeGpaScaleValue(db: SQLiteDatabase, json: string): Promise<void> {
  await setSetting(db, GPA_SCALE_KEY, json);
}
