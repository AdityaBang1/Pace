/**
 * The data layer, against a real SQLite engine.
 *
 * `tests/mobile/app_logic.ts` tests the draft → rows mapping with plain objects,
 * which cannot catch a schema problem: foreign keys, cascades and `ON DELETE`
 * only exist inside a database. This harness runs the *actual* `schema.ts`,
 * `repo.ts` and `commit.ts` code against `node:sqlite`, so an import either lands
 * as one transaction or fails here the way it would on the phone.
 *
 * `expo-sqlite` is only type-imported by these modules, so nothing native is
 * involved. Run with `npm run test:db`.
 */
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';

import { commitImport, deleteAllData, getCourse, importForCourse, listImports, undoImport } from '../../src/database/repo';
import type { CommitInput } from '../../src/database/repo';
import { migrate } from '../../src/database/schema';

/**
 * The subset of `SQLiteDatabase` the app uses, over `node:sqlite`.
 *
 * Deliberately synchronous underneath: the ordering of statements is what this
 * harness is checking, and the app's own `await` points still apply.
 */
class TestDatabase {
  readonly raw = new DatabaseSync(':memory:');

  async execAsync(sql: string): Promise<void> {
    this.raw.exec(sql);
  }

  async runAsync(sql: string, params: unknown[] = []): Promise<void> {
    this.raw.prepare(sql).run(...(params as never[]));
  }

  async getAllAsync<T>(sql: string, params: unknown[] = []): Promise<T[]> {
    return this.raw.prepare(sql).all(...(params as never[])) as T[];
  }

  async getFirstAsync<T>(sql: string, params: unknown[] = []): Promise<T | null> {
    const row = this.raw.prepare(sql).get(...(params as never[]));
    return (row ?? null) as T | null;
  }

  async withTransactionAsync(task: () => Promise<void>): Promise<void> {
    this.raw.exec('BEGIN');
    try {
      await task();
      this.raw.exec('COMMIT');
    } catch (error) {
      this.raw.exec('ROLLBACK');
      throw error;
    }
  }

  count(table: string): number {
    const row = this.raw.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as { n: number };
    return row.n;
  }
}

type Db = Parameters<typeof commitImport>[0];

function input(overrides: Partial<CommitInput> = {}): CommitInput {
  return {
    importId: 'draft-abc-1',
    linkToCourseId: null,
    courseId: 'course-draft-abc-1',
    course: {
      title: 'MATH 1A Calculus',
      code: 'MATH 1A',
      instructor: 'Dr. Reyes',
      credits: 4,
      gradingTotalPercent: 100,
      gradingTotalConsistent: true,
      documentType: 'syllabus',
    },
    document: {
      id: 'doc-abc',
      documentType: 'syllabus',
      academic: true,
      documentTypeConfidence: 0.95,
      sourceKind: 'photo',
      sourceUri: '/pages/doc.jpg',
      ocrText: 'MATH 1A Calculus\nParticipation 10%\nFinal 90%',
      engineVersion: 'pace-engine/1',
      schemaVersion: 'pace.academic-ir/0.3',
      characterCount: 42,
      warnings: [],
      timings: { totalMs: 900 },
    },
    grading: [
      { id: 'draft-abc-1-g0', name: 'Participation', weight: 10, position: 0, evidenceId: 'ev-1' },
      // Typed by the reviewer: no evidence row, which is exactly the row an undo
      // keyed on evidence would miss.
      { id: 'draft-abc-1-g1', name: 'Final', weight: 90, position: 1, evidenceId: null },
    ],
    meetings: [
      {
        id: 'draft-abc-1-m0',
        days: ['MO', 'WE'],
        startTime: '13:15',
        endTime: '14:30',
        room: '214',
        alternatives: false,
        evidenceId: 'ev-1',
      },
    ],
    policies: [{ id: 'draft-abc-1-p0', field: 'attendance', value: 'Required', evidenceId: 'ev-1' }],
    evidence: [
      {
        id: 'ev-1',
        field: 'grading',
        page: 0,
        text: 'Participation 10%',
        start: 12,
        end: 29,
        bbox: { x: 0.1, y: 0.2, width: 0.4, height: 0.03 },
        ocrConfidence: 0.9,
        modelConfidence: null,
        confidenceSource: 'rule',
        extractionMethod: 'deterministic',
      },
    ],
    summary: { course: 'MATH 1A Calculus', grading: 2, meetings: 1, policies: 1, warnings: 0, linked: false },
    ...overrides,
  };
}

async function fresh(): Promise<TestDatabase> {
  const db = new TestDatabase();
  await migrate(db as unknown as Db);
  return db;
}

function check(condition: boolean, label: string): void {
  assert.ok(condition, label);
  console.log(`  ok  ${label}`);
}

async function main(): Promise<void> {
  // 1. An import lands as one unit.
  const db = await fresh();
  await commitImport(db as unknown as Db, input());
  check(db.count('courses') === 1, 'import writes one course');
  check(db.count('imports') === 1, 'import writes one import record');
  check(db.count('grading_categories') === 2, 'import writes every grading row, including a typed one');
  check(db.count('evidence') === 1, 'import writes evidence');

  // 2. Foreign keys are actually enforced on this connection.
  const fk = db.raw.prepare('PRAGMA foreign_keys').get() as { foreign_keys: number };
  check(fk.foreign_keys === 1, 'foreign keys are on, so the schema is really enforcing references');

  // 3. Undo removes exactly this import's rows.
  await undoImport(db as unknown as Db, 'draft-abc-1');
  check(db.count('courses') === 0, 'undo removes the course');
  check(db.count('grading_categories') === 0, 'undo removes grading rows that have no evidence');
  check(db.count('meetings') === 0, 'undo removes meetings');
  check(db.count('policies') === 0, 'undo removes policies');
  check(db.count('documents') === 1, 'undo keeps the document in recent imports');
  check(db.count('evidence') === 1, 'undo keeps the document’s evidence');

  // 4. A linked import: undo must leave the created course alive, and that course
  //    must still offer "Undo" for the import that created it.
  const linked = await fresh();
  await commitImport(linked as unknown as Db, input());
  await commitImport(
    linked as unknown as Db,
    input({
      importId: 'draft-def-2',
      courseId: 'course-draft-def-2',
      linkToCourseId: 'course-draft-abc-1',
      document: { ...input().document, id: 'doc-def', sourceKind: 'file', sourceUri: '/files/schedule.pdf' },
      grading: [
        { id: 'draft-def-2-g0', name: 'Quizzes', weight: 20, position: 0, evidenceId: 'ev-2' },
      ],
      meetings: [],
      policies: [],
      evidence: [
        {
          id: 'ev-2',
          field: 'grading',
          page: 0,
          text: 'Quizzes 20%',
          start: 0,
          end: 11,
          bbox: null,
          ocrConfidence: null,
          modelConfidence: null,
          confidenceSource: 'rule',
          extractionMethod: 'deterministic',
        },
      ],
      summary: { course: 'MATH 1A Calculus', grading: 1, meetings: 0, policies: 0, warnings: 0, linked: true },
    }),
  );
  check(linked.count('courses') === 1, 'a linked import does not create a second course');
  check(linked.count('grading_categories') === 3, 'a linked import adds its rows to the existing course');

  const linkedDetail = await getCourse(linked as unknown as Db, 'course-draft-abc-1');
  check(
    linkedDetail?.evidence.some((row) => row.id === 'ev-2') === true,
    'a linked import’s evidence is reachable from the course it was attached to',
  );

  await undoImport(linked as unknown as Db, 'draft-def-2');
  check(linked.count('grading_categories') === 2, 'undoing a link removes only the linked rows');
  check(linked.count('courses') === 1, 'undoing a link keeps the course');

  const undoable = await importForCourse(linked as unknown as Db, 'course-draft-abc-1');
  check(undoable?.id === 'draft-abc-1', 'the creating import is still the undo target after a link is undone');

  const detail = await getCourse(linked as unknown as Db, 'course-draft-abc-1');
  check(detail != null, 'the course still loads');
  check(
    detail?.evidence.some((row) => row.id === 'ev-1') === true,
    'the course’s own evidence is reachable',
  );

  // 5. Recent imports survive an undo with a readable name.
  const history = await listImports(linked as unknown as Db);
  check(history.length === 2, 'both imports are in history');
  check(
    history.every((row) => Boolean(row.course_title) || row.summary.length > 0),
    'an import row always has a name to show',
  );

  // 6. Delete-all really empties the store.
  await deleteAllData(linked as unknown as Db);
  for (const table of ['documents', 'imports', 'courses', 'grading_categories', 'meetings', 'policies', 'evidence']) {
    check(linked.count(table) === 0, `delete-all empties ${table}`);
  }

  console.log('\nDB OK: commit, undo, links and delete-all behave against real SQLite');
}

main().catch((error) => {
  console.error('\nDB FAILED:', error instanceof Error ? error.message : error);
  process.exit(1);
});
