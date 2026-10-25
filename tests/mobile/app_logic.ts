/**
 * The app logic that sits between the engine and the database, checked without a
 * phone.
 *
 * `buildDraft` and `draftToCommitInput` are pure functions over an AcademicRecord,
 * so they can run here against the same fixtures the parity harness uses. What is
 * *not* covered here — SQLite transactions, the review screen's rendering, Vision
 * OCR — needs a device or simulator, and the report says so.
 */
import { PaceEngine } from '../../src/engine/PaceEngine';
import { buildDraft, gradingTotal, importBlocker, reviewCounts } from '../../src/features/import/draft';
import { draftToCommitInput } from '../../src/features/import/commit';
import { assets, createSession, readFixture } from './nodeAssets';
import { layoutPages, locate } from '../../src/features/scan/geometry';
import type { PaceVisionPage } from '../../modules/pace-vision';

let failures = 0;
let checks = 0;

/** One line-level box per text line, the shape the OCR module returns. */
function blocksFor(text: string, page: number): PaceVisionPage['blocks'] {
  return text.split('\n').map((line, index) => ({
    id: `p${page}-line-${index}`,
    text: line,
    page,
    x: 0.1,
    y: 0.05 + index * 0.05,
    width: 0.8,
    height: 0.03,
    confidence: 0.9,
  }));
}

function check(condition: boolean, name: string, detail?: unknown) {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}${detail === undefined ? '' : ` — ${JSON.stringify(detail)}`}`);
  }
}

async function draftFor(engine: PaceEngine, fixture: string) {
  const text = await readFixture(fixture);
  const { record, text: normalized, timings } = await engine.understand(text);
  const draft = buildDraft(record, {
    source: { kind: 'file', uri: null, label: fixture },
    text: normalized,
    timings: timings as unknown as Record<string, number>,
    evidenceMeta: {},
    image: null,
    match: null,
  });
  return { record, draft, text: normalized };
}

async function main() {
  const engine = await PaceEngine.create(assets, createSession);

  // ---- a syllabus produces a complete, importable draft -------------------
  {
    const { record, draft } = await draftFor(engine, 'syllabus_math.txt');
    check(draft.academic, 'syllabus is academic');
    check(Boolean(draft.title || draft.code), 'syllabus has a course name');
    check(draft.grading.length >= 2, 'syllabus has grading categories', draft.grading.length);
    check(gradingTotal(draft) === 100, 'syllabus weights total 100', gradingTotal(draft));
    check(importBlocker(draft) === null, 'syllabus is importable', importBlocker(draft));
    check(draft.meetings.length > 0, 'syllabus has a meeting', draft.meetings.length);

    const commit = draftToCommitInput(draft);
    check(commit.document.id === record.document.id, 'document id is stable through the commit');
    check(commit.evidence.length === record.evidence.length, 'every evidence row is committed');
    check(commit.grading.length === draft.grading.length, 'every included category is committed');
    check(
      new Set(commit.grading.map((row) => row.id)).size === commit.grading.length,
      'grading row ids are unique',
    );
    check(
      commit.grading.every((row) => row.evidenceId === null || record.evidence.some((e) => e.id === row.evidenceId)),
      'grading rows point at evidence that exists',
    );
    check(commit.course?.gradingTotalPercent === 100, 'stored total is the reviewed total');
    check(commit.course?.gradingTotalConsistent === true, 'stored total is marked consistent');

    const counts = reviewCounts(draft);
    check(counts.total > 0 && counts.total >= counts.needsReview, 'review counts add up', counts);
  }

  // ---- a receipt never becomes an academic record -------------------------
  {
    const { record, draft } = await draftFor(engine, 'receipt.txt');
    check(!draft.academic, 'receipt is not academic');
    check(record.document.gate === 'blocked', 'receipt is gated, not merely unlabelled');
    check(Object.keys(record.course).length === 0, 'receipt has no course fields');
    check(record.grading.length === 0, 'receipt has no grading categories');
    check(record.meetings.length === 0, 'receipt has no meetings');
    check(draft.grading.length === 0, 'receipt draft has nothing to import');
    check(importBlocker(draft) !== null, 'receipt cannot be imported');
    const commit = draftToCommitInput(draft);
    check(commit.course?.title === null && commit.course?.code === null, 'receipt would not create a named course');
    check(commit.grading.length === 0 && commit.meetings.length === 0, 'receipt would commit no academic rows');
  }

  // ---- the reviewer can reopen the gate -----------------------------------
  {
    const text = await readFixture('receipt.txt');
    const { record } = await engine.understand(text, { assumeDocumentType: 'syllabus' });
    check(record.document.gate === 'open', 'override opens the gate');
    check(record.document.documentType?.academic === true, 'override marks the type academic');
    check(
      record.warnings.some((warning) => warning.includes('reviewer')),
      'override is recorded as the reviewer’s decision',
    );
    check(record.grading.length === 0, 'override does not invent grading on a receipt');
  }

  // ---- excluding a row changes what is committed --------------------------
  {
    const { draft } = await draftFor(engine, 'syllabus_chem.txt');
    const dropped = draft.grading[0];
    const edited = {
      ...draft,
      grading: draft.grading.map((row, index) => (index === 0 ? { ...row, included: false } : row)),
    };
    const commit = draftToCommitInput(edited);
    check(commit.grading.length === draft.grading.length - 1, 'excluded row is not committed');
    check(!commit.grading.some((row) => row.name === dropped.value.category && row.weight === dropped.value.weight),
      'excluded row is absent by value');
    check(
      commit.course?.gradingTotalPercent !== 100 && commit.course?.gradingTotalConsistent === false,
      'removing a category makes the stored total inconsistent',
    );
  }

  // ---- stable identity ----------------------------------------------------
  {
    const { draft: first } = await draftFor(engine, 'syllabus_math.txt');
    const { draft: second } = await draftFor(engine, 'syllabus_math.txt');
    check(first.documentId === second.documentId, 'same text gives the same document id');
    check(
      first.grading.map((row) => row.evidenceId).join('|') === second.grading.map((row) => row.evidenceId).join('|'),
      'same text gives the same evidence ids',
    );
  }

  // ---- multi-page offsets, the part a PDF depends on ----------------------
  // The reader sees one joined string, so a value on page 3 must still resolve to
  // page 3. This is pure arithmetic and is worth pinning without a device.
  {
    const pages: PaceVisionPage[] = [
      { page: 0, method: 'ocr', text: 'Alpha\nBeta', blocks: blocksFor('Alpha\nBeta', 0), imageWidth: 100, imageHeight: 140, durationMs: 1 },
      { page: 1, method: 'ocr', text: 'Gamma', blocks: blocksFor('Gamma', 1), imageWidth: 100, imageHeight: 140, durationMs: 1 },
      { page: 2, method: 'ocr', text: 'Delta\nEpsilon', blocks: blocksFor('Delta\nEpsilon', 2), imageWidth: 100, imageHeight: 140, durationMs: 1 },
    ];
    const laid = layoutPages(pages);
    check(
      laid.text === 'Alpha\nBeta\n\nGamma\n\nDelta\nEpsilon',
      'pages are joined with a blank line',
      laid.text,
    );
    check(laid.lines.length === 5, 'every line gets geometry', laid.lines.length);
    check(locate(laid.text.indexOf('Gamma'), laid.lines)?.page === 1, 'a line on page 2 resolves to page 2');
    check(locate(laid.text.indexOf('Epsilon'), laid.lines)?.page === 2, 'a line on page 3 resolves to page 3');
    check(locate(laid.text.indexOf('Alpha'), laid.lines)?.page === 0, 'a line on page 1 resolves to page 1');
    check(
      laid.lines.every((line) =>
        laid.text
          .slice(line.start, line.end)
          .trim()
          .split('\n')
          .every((part) => laid.text.split('\n').includes(part)),
      ),
      'every line span sits inside the line it names',
      laid.lines.map((line) => laid.text.slice(line.start, line.end)),
    );

    // A page of prose is left exactly as it came. Only a page that arrived one
    // text run per cell is laid out again, because that is the page whose own
    // lines are not its printed lines.
    {
      const prose = [
        'MATH 1A: Calculus I',
        'Instructor: Dr. Lena Ortiz',
        'Meeting Times: TU 11:00-12:15 in Room 204',
        'Grading: Essays 30%, Quizzes 20%, Midterm 20%, Final 30%',
        'Attendance is required, and late work loses ten percent a day.',
        'Office hours: Thursday 2:00-3:00, or by appointment.',
      ].join('\n');
      const prosePage: PaceVisionPage = {
        page: 0,
        method: 'ocr',
        text: prose,
        blocks: prose.split('\n').map((line, index) => ({
          id: `p0-${index}`,
          text: line,
          page: 0,
          x: 0.08,
          y: 0.05 + index * 0.045,
          width: 0.8,
          height: 0.03,
          confidence: 0.9,
        })),
        imageWidth: 1000,
        imageHeight: 1400,
        durationMs: 1,
      };
      check(layoutPages([prosePage]).text === prose, 'a page of prose keeps its own lines');
      check(
        layoutPages([prosePage]).lines.length === 6,
        'and one geometry entry per line',
        layoutPages([prosePage]).lines.length,
      );
    }

    // A page whose boxes are missing still contributes its text; it only loses the
    // highlight, because a box we do not have cannot be pointed at.
    const broken: PaceVisionPage = { ...pages[1], blocks: [] };
    const withoutBoxes = layoutPages([pages[0], broken]);
    check(withoutBoxes.text.includes('Gamma'), 'a page without boxes keeps its text', withoutBoxes.text);
    check(
      locate(withoutBoxes.text.indexOf('Gamma'), withoutBoxes.lines)?.bbox === null,
      'and its lines carry no box instead of a wrong one',
      locate(withoutBoxes.text.indexOf('Gamma'), withoutBoxes.lines),
    );
  }

  // ---- cross-document matching is a proposal, not a merge -----------------
  {
    const results = engine.matchCandidates('MATH 1A: Calculus', ['MATH 1A Calculus I', 'CHEM 1L']);
    check(results.length === 2, 'matcher scores every candidate');
    check(results[0].probability > results[1].probability, 'the right candidate ranks first', results.map((r) => r.probability));
    check(results[0].decision !== 'different', 'a near-identical title is not rejected', results[0].decision);
  }

  // ---- dot-leader tables, which no fixture uses --------------------------
  // The Python rule and this port both changed to accept "Essays ...... 30%".
  // The fixtures do not contain that layout, so the parity harness cannot see a
  // divergence here; this is the check that can.
  {
    const text = [
      'Course Syllabus',
      'Instructor: Ms. Priya Raman',
      'Course: Honors World Literature (ENG 210H)',
      '',
      'Grading Policy',
      'Participation ............ 15%',
      'Close Reading Essays ..... 30%',
      'Comparative Paper ........ 25%',
      'Final Portfolio .......... 30%',
      '',
      'Office Hours',
      'Tuesdays 3:00-4:00 PM in the library annex.',
      '',
      'Academic Integrity',
      'All submitted work must be your own.',
    ].join('\n');
    const { record } = await engine.understand(text);
    check(record.grading.length === 4, 'dot-leader weights are read', record.grading.map((row) => row.category));
    check(record.gradingTotalPercent === 100, 'dot-leader weights total 100', record.gradingTotalPercent);
    check(
      record.grading.every((row) => record.evidence.some((e) => e.field === 'grading' && e.start === row.evidence.start)),
      'dot-leader rows carry evidence',
    );
  }

  // ---- the known gap: the gate abstains on this schedule ------------------
  // This fixture is a real limitation of the frozen doc-type model, not a bug in
  // the app: it proposes "other" at 0.70 and suppresses extraction. The test pins
  // the behaviour so a better model has to be a deliberate change, and checks that
  // the reviewer's override is the supported path in the meantime.
  {
    const { draft } = await draftFor(engine, 'schedule.txt');
    check(!draft.academic, 'the gate still abstains on this schedule', draft.documentType?.label);
    check(draft.documentType?.abstained === true, 'the abstention is visible to the UI');
    check(draftToCommitInput(draft).meetings.length === 0, 'a blocked schedule commits nothing');

    const text = await readFixture('schedule.txt');
    const { record } = await engine.understand(text, { assumeDocumentType: 'schedule' });
    check(record.document.gate === 'open', 'override opens the schedule gate');
    console.log(
      `schedule override → meetings ${record.meetings.length}, course ${JSON.stringify(record.course)}`,
    );
  }

  console.log(`${failures ? 'FAILED' : 'APP LOGIC OK'}: ${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
