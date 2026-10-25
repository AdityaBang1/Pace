/**
 * The transcript path, end to end without a phone.
 *
 * A syllabus is one course with grading weights; a transcript is twenty courses
 * with a mark and a credit each. Running the syllabus readers over a transcript
 * produced exactly the wrong record — the printed column header became the course
 * title — so this harness pins the shape that replaced it:
 *
 *   1. the reader finds the rows the page prints (terms, marks, credits, totals)
 *   2. the engine reports the document as a transcript and runs no syllabus reader
 *   3. the draft and the commit keep one course per row, with its term and mark
 *
 * Run with `npm run test:transcript`.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';

import type { PaceVisionBlock, PaceVisionPage } from '../../modules/pace-vision';
import { PaceEngine } from '../../src/engine/PaceEngine';
import { readTranscript } from '../../src/engine/transcript';
import { buildDraft, importBlocker, includedTranscript, reviewCounts } from '../../src/features/import/draft';
import { draftToCommitInput } from '../../src/features/import/commit';
import { layoutPages } from '../../src/features/scan/geometry';
import { assets, createSession } from './nodeAssets';

const ROOT = path.resolve(__dirname, '../..');
const SAMPLES = path.join(ROOT, 'data', 'samples');

let failures = 0;
let checks = 0;

function check(condition: boolean, name: string, detail?: unknown): void {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}${detail === undefined ? '' : ` — ${JSON.stringify(detail)}`}`);
  }
}

function sample(name: string): string {
  return readFileSync(path.join(SAMPLES, `${name}.txt`), 'utf8').replace(/\r\n/g, '\n');
}

async function main(): Promise<void> {
  const engine = await PaceEngine.create(assets, createSession);

  // ---- the term-block layout: course / mark / weight / credit ---------------
  {
    const text = sample('transcript_term_marks');
    const reading = readTranscript(text);
    check(reading.detected, 'a term table is detected as a transcript');
    check(reading.layout === 'course_mark_weight_credit', 'its layout is the numeric one', reading.layout);
    check(reading.courses.length === 8, 'every printed row is read', reading.courses.length);
    check(
      reading.courses[0].code === '220725y' && reading.courses[0].title === 'Accel Adv Alg:Concept&Connect',
      'a code and its wrapped title are one row',
      reading.courses[0],
    );
    check(reading.courses[1].title === 'Accel Geo B:Concept&Connect', 'the second wrapped title is joined', reading.courses[1].title);
    check(reading.courses[4].title === 'Intro to Business and Tech', 'a title wrapped after "and" is still joined', reading.courses[4].title);
    check(reading.courses.every((row) => row.mark !== null), 'every row has a mark');
    check(reading.courses.every((row) => row.credit !== null), 'every row has a credit');
    check(reading.totalCredits === 7.5, 'the rows add up to the printed 7.5 credits', reading.totalCredits);
    check(reading.reportedCredits === 7.5, 'the printed footer total is read', reading.reportedCredits);
    check(reading.reportedGpa === 4.13333, 'the printed GPA is read', reading.reportedGpa);
    check(reading.checks.creditSumMatchesReported === true, 'the credit check agrees with the page');
    check(
      reading.courses.every((row) => text.slice(row.span.start, row.span.end).includes(row.title.split(' ')[0])),
      'every row span points at its own printed line',
    );
    check(
      reading.terms.map((term) => term.value).every((value) => value === '2025-2026 Grade 09 Term 2'),
      'the term banner is named once per printed banner',
      reading.terms.map((term) => term.value),
    );

    const { record, timings } = await engine.understand(text);
    check(record.document.documentType?.label === 'transcript', 'the record is typed as a transcript');
    check(record.document.documentType?.proposedLabel === 'course_catalog', 'the gate proposal is kept beside it', record.document.documentType?.proposedLabel);
    check(record.document.gate === 'open', 'the gate is open for a transcript');
    check(record.transcript !== null, 'the record carries a transcript block');
    check(record.transcript?.courses.length === 8, 'the record carries every row', record.transcript?.courses.length);
    check(Object.keys(record.course).length === 0, 'no syllabus course title is invented', record.course);
    check(record.grading.length === 0, 'no grading categories are invented');
    check(record.meetings.length === 0, 'no meetings are invented');
    check(record.sections.length === 0, 'the section model does not run');
    check(timings.sectionModelMs === 0, 'the section model takes no time on a transcript', timings.sectionModelMs);
    check(
      record.warnings.some((warning) => warning.startsWith('read as a transcript: 8 courses')),
      'the record says it was read as a transcript',
      record.warnings,
    );
    check(
      record.warnings.some((warning) => warning.includes('has no transcript class')),
      'the record says the gate had no transcript class',
      record.warnings,
    );
    check(record.evidence.length === 8, 'one evidence row per course', record.evidence.length);
    check(record.evidence.every((row) => row.groundedInSource), 'every transcript evidence row is grounded');
    check(
      record.evidence.every((row) => text.slice(row.start ?? 0, row.end ?? 0).trim().length > 0),
      'every transcript evidence row has a span',
    );
    check(record.transcript?.terms.length === 1, 'the duplicated banner is one term', record.transcript?.terms.length);
    check(record.transcript?.terms[0].courses === 8, 'the term counts its rows', record.transcript?.terms[0]);

    const draft = buildDraft(record, {
      source: { kind: 'file', uri: null, label: 'transcript_term_marks.txt' },
      text,
      timings: timings as unknown as Record<string, number>,
      evidenceMeta: {},
      image: null,
      match: null,
    });
    check(draft.transcript !== null, 'the draft is a transcript draft');
    check(draft.transcript?.rows.length === 8, 'one proposal per row', draft.transcript?.rows.length);
    check(draft.title === null, 'a transcript draft has no single course title', draft.title);
    check(draft.academic, 'a transcript draft is academic');
    check(importBlocker(draft) === null, 'a transcript is importable', importBlocker(draft));
    check(reviewCounts(draft).total === 8, 'the review counts eight rows', reviewCounts(draft));
    check(Boolean(draft.transcript?.rows.every((row) => row.included)), 'every row starts included');
    check(
      Boolean(
        draft.transcript?.rows.every(
          (row) => row.start != null && text.slice(row.start, row.end ?? 0).includes(row.value.title.split(' ')[0]),
        ),
      ),
      'every proposal resolves to its own source span',
    );

    const commit = draftToCommitInput(draft);
    check(commit.transcriptCourses?.length === 8, 'the commit writes one course per row', commit.transcriptCourses?.length);
    check(commit.course === null, 'the commit has no single course');
    check(commit.grading.length === 0 && commit.meetings.length === 0, 'the commit writes no syllabus rows');
    check(
      Boolean(commit.transcriptCourses?.every((course) => course.term === '2025-2026 Grade 09 Term 2')),
      'every committed course keeps its term',
    );
    check(
      commit.transcriptCourses?.[0].grade === '94' && commit.transcriptCourses?.[0].gradePercent === 94,
      'a numeric mark is kept as printed and as a percentage',
      commit.transcriptCourses?.[0],
    );
    check(
      new Set(commit.transcriptCourses?.map((course) => course.id)).size === 8,
      'committed course ids are unique',
    );
    check(
      Boolean(commit.transcriptCourses?.every(
        (course) => !course.evidenceId || record.evidence.some((row) => row.id === course.evidenceId),
      )),
      'every committed course points at evidence that exists',
    );
    check(commit.summary.transcript === true && commit.summary.courses === 8, 'the import summary counts the courses', commit.summary);
  }

  // ---- one page printed in two panels, one table per school --------------
  // The shape a real registrar PDF has: two panels on one page, a school banner
  // and a column header above every table, each table under its own `Credit:`
  // footer, and a term whose rows run over the panel break under a second header.
  // Three things went wrong here and each one lost courses: a title that wrapped
  // onto the line below took the next course's title with it (so two classes came
  // out as one), a title printed right up to the mark column lost its row, and a
  // term split over the break was checked against a footer that covers both
  // halves, so a complete term read as a missing row.
  {
    const text = sample('transcript_two_panels');
    const reading = readTranscript(text);
    check(reading.detected, 'a two-panel transcript is detected');
    check(reading.layout === 'course_mark_weight_credit', 'its layout is the numeric one', reading.layout);
    check(reading.student === 'Doe, Jordan', 'the student name beside the title is read', reading.student);
    check(reading.school === 'Northgate High School', 'the school is read from the title line', reading.school);
    check(reading.courses.length === 14, 'every course on the page is read', reading.courses.length);
    check(
      reading.courses.every((row) => row.code !== null),
      'every row keeps its course number',
      reading.courses.filter((row) => row.code === null).map((row) => row.title),
    );
    check(
      reading.courses[0].code === '240118g' && reading.courses[0].title === 'Accel Geometry:Concepts',
      'a code and the title wrapped under it are one row',
      reading.courses[0],
    );
    // A title printed with a single space before its mark used to fail the row.
    const software = reading.courses.find((row) => row.code === '650441h');
    check(
      software?.title === 'Intro to Software Tech' && software.mark === '99' && software.credit === 1,
      'a title that runs up to the mark column keeps its row',
      software,
    );
    // The row above it must not have swallowed it.
    const business = reading.courses.find((row) => row.code === '650440h');
    check(
      business?.title === 'Intro to Business and Tech',
      'a wrapped title stops at its own last word instead of taking the next course',
      business,
    );
    check(
      reading.courses.some((row) => row.title === 'Literature & Composition I Hnr'),
      'a title wrapped after an ampersand is joined',
      reading.courses.map((row) => row.title),
    );
    check(reading.totalCredits === 11.5, 'the credits add up across both panels', reading.totalCredits);
    check(
      reading.checks.tablesWhoseCreditsDisagree === 0,
      'a term split over the panel break is checked against the footer of the whole term',
      reading.checks,
    );
    check(
      reading.tableTotals.length === 4 &&
        reading.tableTotals[3].rows === 8 &&
        reading.tableTotals[3].credits === 7.5 &&
        reading.tableTotals[3].reportedCredits === 7.5,
      'the split term is one table of eight rows worth 7.5 credits',
      reading.tableTotals,
    );
    check(
      reading.courses.every((row) =>
        text.slice(row.span.start, row.span.end).includes(row.title.split(' ')[0]),
      ),
      'every row span points at its own printed line',
    );

    const { record } = await engine.understand(text);
    check(record.transcript?.courses.length === 14, 'the record carries every row', record.transcript?.courses.length);
    check(record.transcript?.terms.length === 4, 'the repeated banner is one term', record.transcript?.terms);
    check(
      record.transcript?.terms.find((term) => term.value === '2025-2026 Grade 10 Term 2')?.courses === 8,
      'the split term counts both halves',
      record.transcript?.terms,
    );
    check(
      !record.warnings.some((warning) => warning.includes('a row may have been missed')),
      'a complete transcript raises no missing-row warning',
      record.warnings,
    );
  }

  // ---- the same page reported one text run per cell -------------------------
  // A PDF text layer reports the runs the author drew, one per line: the column
  // header arrives one word per line, a row's title and each of its numbers on
  // lines of their own, and a course still in progress as a title with its credit
  // printed underneath. This is the shape the phone sees for a registrar PDF.
  {
    const text = [
      'South Forsyth High School Transcript',
      'Bang, Aditya Prashant',
      'Course',
      'Mark',
      'Weight',
      'Credit',
      '2025-2026 Grade 09   Term 2',
      '220725y Accel Adv',
      'Alg:Concept&Connect',
      '94',
      '1.0000',
      '1',
      '42075y AP Human Geography',
      '90',
      '1.0000',
      '1',
      'Credit: 2.000  GPA: 4.13333',
      'In-Progress Courses',
      '700695y  AP Business',
      '1.000',
      '50725y Spanish III',
      '1.000',
      'Credit Summary',
      'High School Diploma   Attempted   Earned',
      'Total   2.000   2.000',
      '',
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.detected, 'a page reported one run per cell is still a transcript');
    check(reading.courses.length === 4, 'every printed row is read, finished or in progress', reading.courses.length);
    check(
      reading.courses[0].code === '220725y' && reading.courses[0].title === 'Accel Adv Alg:Concept&Connect',
      'a title split over two cells and a row split over five lines is one course',
      reading.courses[0],
    );
    check(reading.courses[0].mark === '94' && reading.courses[0].credit === 1, 'its numbers are its own', reading.courses[0]);
    check(reading.totalCredits === 2, 'the finished courses are the printed total', reading.totalCredits);
    const inProgress = reading.courses.filter((row) => row.inProgress);
    check(inProgress.length === 2, 'both in-progress rows are read', inProgress.map((row) => row.title));
    check(
      inProgress[0].title === 'AP Business' && inProgress[0].code === '700695y' && inProgress[0].credit === 1,
      'a title with its credit on the line below is one course',
      inProgress[0],
    );
    check(inProgress.every((row) => row.mark === null && row.term === null), 'an in-progress row has no mark and no term');
    check(reading.inProgressCredits === 2, 'their credits are counted apart from the total', reading.inProgressCredits);
    check(
      text.slice(inProgress[0].span.start, inProgress[0].span.end).includes('AP Business'),
      'an in-progress row span names its own line',
      inProgress[0].span,
    );

    const { record } = await engine.understand(text);
    check(record.transcript?.courses.length === 4, 'the record carries all four rows', record.transcript?.courses.length);
    check(
      record.transcript?.inProgressCredits === 2,
      'the record separates the credits in progress from the total',
      record.transcript?.inProgressCredits,
    );
    const draft = buildDraft(record, {
      source: { kind: 'file', uri: null, label: 'transcript.pdf' },
      text,
      timings: {},
      evidenceMeta: {},
      image: null,
      match: null,
    });
    check(draft.transcript?.rows.length === 4, 'the draft proposes every row', draft.transcript?.rows.length);
    check(
      draft.transcript?.rows.filter((row) => row.value.inProgress).length === 2,
      'the draft carries which rows are in progress',
    );
    const commit = draftToCommitInput(draft);
    check(commit.transcriptCourses?.length === 4, 'all four rows are imported', commit.transcriptCourses?.length);
    check(
      commit.transcriptCourses?.filter((course) => course.inProgress).length === 2,
      'the in-progress rows are imported as in progress',
      commit.transcriptCourses,
    );
    check(commit.summary.inProgress === 2, 'the summary counts them separately', commit.summary);
    check(commit.summary.credits === 2, 'the summary total counts finished courses only', commit.summary);
  }

  // ---- the dated-row layout, with subject headings and a state assessment ---
  {
    const text = sample('mock_transcript_lepid');
    const reading = readTranscript(text);
    check(reading.detected, 'a dated transcript is detected');
    check(reading.layout === 'dated_rows', 'its layout is the dated one', reading.layout);
    check(reading.courses.length === 38, 'every dated row is read', reading.courses.length);
    check(reading.courses[0].title === 'Elective Intensive', 'the first course is read', reading.courses[0].title);
    check(reading.courses[0].mark === 'A', 'a letter mark is kept as printed', reading.courses[0].mark);
    check(reading.courses[0].credit === 0.25, 'its credit is read', reading.courses[0].credit);
    check(
      reading.courses.find((row) => row.title === 'Algebra Prep')?.section === 'MATH',
      'a subject heading owns the rows below it',
    );
    check(reading.courses.filter((row) => row.section === 'LANGUAGE ARTS').length === 11, 'every subject area is assigned', reading.courses.filter((row) => row.section === 'LANGUAGE ARTS').length);
    check(reading.student === 'Lepid, Arthur', 'the student name is read', reading.student);
    check(reading.documentTitle === 'ACADEMIC HISTORY', 'the document title is read', reading.documentTitle);
    check(reading.school === 'Pacific High School', 'the school is read', reading.school);
    check(reading.totalCredits === 13.25, 'the credits add up', reading.totalCredits);
    check(reading.reportedCredits === 13.25, 'the printed total credits are read', reading.reportedCredits);
    check(reading.assessments.length === 3, 'the state assessments are read', reading.assessments.length);
    check(
      reading.assessments[0].level === 3 && reading.assessments[0].label === 'Proficient',
      'an assessment keeps its performance level',
      reading.assessments[0],
    );
    check(reading.courses.every((row) => row.term === null), 'no term banner means no term, not a guessed one');

    const { record } = await engine.understand(text);
    check(record.document.documentType?.label === 'transcript', 'the dated transcript is typed the same way');
    check(record.transcript?.student === 'Lepid, Arthur', 'the record keeps the student name');
    check(record.transcript?.assessments.length === 3, 'the record keeps the assessments', record.transcript?.assessments.length);
    check(
      record.warnings.some((warning) => warning.includes('no term heading was printed')),
      'the record says the rows are not grouped by term',
      record.warnings,
    );
    check(record.course.course_title === undefined, 'no syllabus title is invented from the school banner');
  }

  // ---- a row-wise OCR of a two-column page ---------------------------------
  {
    const text = sample('mock_transcript_lepid_rowwise');
    const reading = readTranscript(text);
    check(reading.detected, 'a row-wise two-column page is detected');
    check(reading.checks.columnInterleavingDetected === true, 'the page column seam is found');
    check(reading.courses.length === 38, 'both page columns are read', reading.courses.length);
    // Both page columns must survive: one title printed only in the left column
    // and one printed only in the right one.
    check(
      reading.courses.some((row) => row.title === 'Readers Workshop') &&
        reading.courses.some((row) => row.title === 'Crew'),
      'both page columns are read, not just the first',
      reading.courses.map((row) => row.title).slice(0, 8),
    );
    const published = reading.courses.find((row) => row.title === 'AK Studies');
    check(published?.section === 'SOCIAL STUDIES', 'a right-column row keeps its own heading', published?.section);
  }

  // ---- a PDF text layer, which hands over one text run per printed cell ----
  // Real geometry: the runs of data/samples/transcript_term_marks.pdf. A text
  // layer reports the runs the author drew, so the page arrives as `Course`,
  // `Mark`, `Weight`, `Credit` and then each course's code, title, wrapped title
  // and numbers on lines of their own. The reader puts those runs back into rows;
  // without that this page read as no transcript at all.
  {
    const runs = JSON.parse(
      readFileSync(path.join(SAMPLES, 'transcript_term_marks_runs.json'), 'utf8'),
    ) as PaceVisionBlock[];
    const text = runs.map((block) => block.text).join('\n');
    const reading = readTranscript(text);
    check(reading.detected, 'a transcript read from a PDF text layer is detected', reading.checks);
    check(reading.layout === 'course_mark_weight_credit', 'its layout is the numeric one', reading.layout);
    check(reading.courses.length === 8, 'every course the PDF prints is read', reading.courses.length);
    check(
      reading.courses[0].code === '220725y' && reading.courses[0].title === 'Accel Adv Alg:Concept&Connect',
      'a code and the title that wrapped under it are one row',
      reading.courses[0],
    );
    check(reading.courses.every((row) => row.mark !== null), 'every row keeps its mark');
    check(reading.totalCredits === 7.5, 'the rows add up to the printed 7.5', reading.totalCredits);
    check(reading.reportedCredits === 7.5, 'the printed footer is read', reading.reportedCredits);
  }

  // ---- two term tables printed side by side on one page --------------------
  // This is the shape that folded two courses into one: both columns share every
  // baseline, so a reader that matches numbers from the right edge of a line took
  // the left column's title for the right column's. One page, two tables: each
  // course keeps its own title, mark and credit, and each table is checked against
  // its own footer.
  {
    const runs = JSON.parse(
      readFileSync(path.join(SAMPLES, 'transcript_term_marks_runs.json'), 'utf8'),
    ) as PaceVisionBlock[];
    const text = [
      ...runs.map((block) => block.text),
      ...runs.map((block) => block.text.replace('Term 2', 'Term 1')),
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.courses.length === 16, 'both page columns are read in full', reading.courses.length);
    check(reading.terms.length === 2, 'both term banners are read', reading.terms.map((term) => term.value));
    check(
      reading.courses.filter((row) => row.title.includes('Accel Adv')).length === 2,
      'a wrapped title is not lost in either column',
      reading.courses.map((row) => row.title),
    );
    check(
      reading.courses.filter((row) => row.title === 'Intro to Business and Tech').length === 2,
      'a title wrapped before a word break is joined in both columns',
    );
    check(reading.totalCredits === 15, 'the credits of both terms are summed', reading.totalCredits);
    check(
      reading.tableTotals.length === 2 &&
        reading.tableTotals.every((table) => table.reportedCredits === 7.5 && table.credits === 7.5),
      'each table is checked against its own printed footer, not one against the whole',
      reading.tableTotals,
    );
    check(reading.reportedCredits === null, 'no document total is claimed when the page never printed one', reading.reportedCredits);
    check(
      reading.checks.tablesWhoseCreditsDisagree === 0,
      'the per-table check finds no missing row',
      reading.checks,
    );
  }

  // ---- a second term printed under the header above it --------------------
  // A school that prints its column header once, then a term banner and that
  // term's rows with their own footer, is printing two tables. Stopping at the
  // first footer read the first term only and lost the course under the second
  // banner, which is a whole credit missing from the page's own total.
  {
    const text = [
      'Course                                      Mark   Weight   Credit',
      '2025-2026 Grade 09   Term 1',
      '03391 GAVS AP Psychology A   99   0.5000   0.5',
      'Credit: .500   GPA: 5.00000',
      '2025-2026 Grade 09   Term 2',
      '03392 GAVS AP Psychology B   99   0.5000   0.5',
      'Credit: .500   GPA: 5.00000',
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.courses.length === 2, 'the rows under the second banner are read', reading.courses.length);
    check(
      reading.courses.map((row) => row.code).join(',') === '03391,03392',
      'each term keeps its own course',
      reading.courses.map((row) => row.code),
    );
    check(
      reading.courses.map((row) => row.term).join(' / ') ===
        '2025-2026 Grade 09 Term 1 / 2025-2026 Grade 09 Term 2',
      'each row keeps the term printed above it',
      reading.courses.map((row) => row.term),
    );
    check(reading.totalCredits === 1, 'the two terms add up', reading.totalCredits);
    check(
      reading.tableTotals.length === 2 &&
        reading.tableTotals.every((table) => table.rows === 1 && table.credits === 0.5 && table.reportedCredits === 0.5),
      'each term is checked against its own footer',
      reading.tableTotals,
    );
    check(reading.checks.tablesWhoseCreditsDisagree === 0, 'and neither is called short', reading.checks);
  }

  // ---- a header printed once, rows continuing on the pages below it --------
  // A printed transcript often labels its columns once and then repeats only the
  // term banner and the footer on the next page. Reading a table only up to its
  // own footer read the first page and dropped the rest of the year.
  {
    const text = [
      '2025-2026 Grade 09 Term 2',
      'Course                                        Mark         Weight       Credit',
      '220725y     Accel Adv                                        94          1.0000             1',
      'Alg:Concept&Connect',
      'Credit: 7.500   GPA: 4.13333',
      '',
      '2025-2026 Grade 09 Term 2',
      '200715b     Accel Geo                                        94          0.5000           0.5',
      'B:Concept&Connect',
      'Credit: 7.500   GPA: 4.13333',
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.courses.length === 2, 'the page under the header is still the same table', reading.courses.length);
    check(
      reading.courses[1].code === '200715b' && reading.courses[1].title === 'Accel Geo B:Concept&Connect',
      'and keeps its own code and wrapped title',
      reading.courses[1],
    );
    check(reading.courses[1].credit === 0.5 && reading.courses[1].mark === '94', 'and its own numbers', reading.courses[1]);
  }

  // ---- two rows that landed on one text line, with no page gutter ----------
  {
    const text = [
      '2025-2026 Grade 09 Term 2',
      'Course            Mark   Weight   Credit',
      '220725y Accel Adv   94   1.0000   1     200715b Accel Geo   94  0.5000  0.5',
      'Credit: 1.500   GPA: 4.0',
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.courses.length === 2, 'each row of a shared line is read', reading.courses.length);
    check(
      reading.courses[0].title === 'Accel Adv' && reading.courses[1].title === 'Accel Geo',
      'the two courses do not become one',
      reading.courses.map((row) => row.title),
    );
    check(
      reading.courses[0].credit === 1 && reading.courses[1].credit === 0.5,
      'each row keeps its own credit',
      reading.courses.map((row) => row.credit),
    );
  }

  // ---- a row whose title and numbers were printed on separate lines -------
  // The shape a PDF text layer actually arrives in on a device: the title run and
  // its wrapped tail, then the numbers on a line of their own, because the tail
  // changed the baseline between them. Reading the numbers as a line with no
  // title dropped every row whose title wrapped — seven courses of a real
  // transcript — while the single-line rows read fine.
  {
    const text = [
      '2025-2026 Grade 09 Term 2',
      'Course            Mark   Weight   Credit',
      '220725y Accel Adv',
      'Alg:Concept&Connect',
      '94   1.0000   1',
      '42075y AP Human Geography',
      '90   1.0000   1',
      'Credit: 2.000   GPA: 4.0',
    ].join('\n');
    const reading = readTranscript(text);
    check(reading.courses.length === 2, 'a row split over three lines is read', reading.courses.length);
    check(
      reading.courses[0].code === '220725y' && reading.courses[0].title === 'Accel Adv Alg:Concept&Connect',
      'its title and the tail under it are one row',
      reading.courses[0],
    );
    check(
      reading.courses[0].mark === '94' && reading.courses[0].weight === 1 && reading.courses[0].credit === 1,
      'and its numbers land in their own columns',
      reading.courses[0],
    );
    check(reading.courses[1].title === 'AP Human Geography', 'the next row keeps its own title', reading.courses[1].title);
  }

  // ---- columns a course does not need, and columns a row left empty -------
  // A transcript prints numbers that are not a mark, a weight or a credit -- grade
  // points, earned credit -- and leaves the weight of an unweighted course blank.
  // Both used to cost the whole row: an unrecognised column rejected the table,
  // and a short row failed to line up with its columns.
  {
    const extra = [
      '2025-2026 Grade 09 Term 2',
      'Course                        Mark  Weight  Credit  Grade    Points',
      '220725y Accel Adv              94   1.0000       1     4.0       12',
      '200715b Accel Geo              94   0.5000     0.5     2.0        6',
      'Credit: 1.500   GPA: 4.0',
    ].join('\n');
    const reading = readTranscript(extra);
    check(reading.courses.length === 2, 'an extra numeric column does not reject the table', reading.courses.length);
    check(
      reading.courses[0].mark === '94' && reading.courses[0].credit === 1,
      'and the columns we keep are still read',
      reading.courses[0],
    );

    const blank = [
      '2025-2026 Grade 09 Term 2',
      'Course                        Mark  Weight  Credit',
      '220725y Accel Adv              94   1.0000       1',
      '200715b Accel Geo              94                0.5',
      '32115y Biology I Honors        93   1.0000       1',
      'Credit: 2.500   GPA: 4.0',
    ].join('\n');
    const short = readTranscript(blank);
    check(short.courses.length === 3, 'a row with a blank column is kept', short.courses.length);
    const unweighted = short.courses.find((row) => row.title === 'Accel Geo');
    check(
      unweighted?.mark === '94' && unweighted.weight === null && unweighted.credit === 0.5,
      'the blank stays blank and its neighbours do not shift',
      unweighted,
    );
    check(short.totalCredits === 2.5 && short.reportedCredits === 2.5, 'the credits still add up', short.totalCredits);
  }

  // ---- a syllabus is not a transcript --------------------------------------
  for (const name of ['mock_syllabus', 'example_syllabus']) {
    const text = sample(name);
    const reading = readTranscript(text);
    check(!reading.detected, `${name} is not detected as a transcript`);
    const { record } = await engine.understand(text);
    check(record.transcript === null, `${name} has no transcript block`);
    check(
      Object.keys(record.course).length > 0 || record.grading.length > 0,
      `${name} still goes through the syllabus reader`,
    );
  }

  // ---- the reviewer's choice is honoured both ways -------------------------
  {
    const text = sample('transcript_term_marks');
    const asSyllabus = await engine.understand(text, { assumeDocumentType: 'syllabus' });
    check(asSyllabus.record.transcript === null, 'choosing syllabus turns the transcript reader off');
    check(
      asSyllabus.record.document.documentType?.label === 'syllabus',
      'the record says the reviewer set the type',
      asSyllabus.record.document.documentType,
    );

    const syllabus = sample('mock_syllabus');
    const asTranscript = await engine.understand(syllabus, { assumeDocumentType: 'transcript' });
    check(asTranscript.record.transcript !== null, 'choosing transcript produces a transcript record');
    check(asTranscript.record.transcript?.courses.length === 0, 'with no rows, because the page prints none');
    check(
      asTranscript.record.warnings.some((warning) => warning.includes('no transcript row was recognised')),
      'and says so instead of pretending',
      asTranscript.record.warnings,
    );
  }

  if (failures) {
    console.error(`\nTRANSCRIPT FAILED: ${failures} of ${checks} checks`);
    process.exitCode = 1;
    return;
  }
  console.log(`TRANSCRIPT OK: ${checks} checks passed`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
