/**
 * The academic arithmetic, without a phone.
 *
 * Two numbers reach the user: credits, which come from rows, and a GPA, which is
 * either what the page printed or arithmetic on a scale the user set. This harness
 * pins both — in particular that a course in progress never counts as earned, that
 * no projection is produced without a scale, and that a printed GPA is never
 * re-derived from the courses.
 *
 * Run with `npm run test:domain`.
 */
import type { CourseRow } from '../../src/database/repo';
import { creditsValue, gpaValue, summarise, termLabel } from '../../src/domain/academics';
import {
  DEFAULT_SCALE,
  levelOf,
  letterFor,
  parseScale,
  projectGpa,
  weightedGpa,
} from '../../src/domain/gpa';

let failures = 0;
let checks = 0;

function check(condition: boolean, name: string, detail?: unknown): void {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}${detail === undefined ? '' : ` — ${JSON.stringify(detail)}`}`);
  }
}

function course(overrides: Partial<CourseRow>): CourseRow {
  return {
    id: 'c',
    title: 'Course',
    code: null,
    instructor: null,
    credits: 1,
    term: '2025-2026 Grade 09 Term 2',
    grade: '94',
    grade_percent: 94,
    in_progress: 0,
    evidence_id: null,
    document_id: null,
    import_id: null,
    grading_total_percent: null,
    grading_total_consistent: null,
    document_type: 'transcript',
    created_at: '',
    updated_at: '',
    ...overrides,
  };
}

// The transcript this was written against: 13.0 earned, 7.0 in progress.
const rows: CourseRow[] = [
  course({ id: 'a', title: 'Accel Adv Alg:Concept&Connect', credits: 1, term: '2025-2026 Grade 09 Term 2' }),
  course({ id: 'b', title: 'AP Human Geography', credits: 1, grade: '90' }),
  course({ id: 'c', title: 'Spanish II', credits: 1, grade: '97' }),
  course({ id: 'd', title: 'Health', credits: 0.5, term: '2024-2025 Grade 09 Term 7', grade: '99' }),
  course({ id: 'e', title: 'AP Business', credits: 1, term: null, grade: null, grade_percent: null, in_progress: 1 }),
  course({ id: 'f', title: 'Spanish III', credits: 1, term: null, grade: null, grade_percent: null, in_progress: 1 }),
];

const summary = summarise(rows);
check(summary.count === 6, 'every course is counted', summary.count);
check(summary.earned === 3.5, 'earned credits are the finished courses', summary.earned);
check(summary.inProgress === 2, 'in-progress credits are kept apart', summary.inProgress);
check(summary.current.length === 2, 'the current courses are the in-progress ones', summary.current);
check(summary.terms.length === 2, 'the finished courses group by term', summary.terms);
check(summary.terms[0].label === '2025–26 · Term 2', 'a term reads as its year and term', summary.terms[0].label);
check(summary.terms[1].term === '2024-2025 Grade 09 Term 7', 'the older term is kept as printed', summary.terms[1].term);
check(summary.terms[0].credits === 3, 'a term sums its own credits', summary.terms[0].credits);

check(termLabel('2025-2026 Grade 09 Term 1') === '2025–26 · Term 1', 'a term label drops the grade level');
check(termLabel('Fall 2025') === 'Fall 2025', 'an unfamiliar term is passed through');
check(termLabel(null) === 'No term printed', 'a missing term says so');

check(creditsValue(13) === '13', 'a whole credit has no decimal');
check(creditsValue(7.5) === '7.5', 'a half credit keeps one decimal');
check(creditsValue(null) === '—', 'an unknown credit is a dash');
check(gpaValue(4.23529) === '4.24', 'a printed GPA is rounded for display only');

check(letterFor('94') === null, 'a numeric mark is not a letter');
check(letterFor('A-') === 'A', 'a letter mark is its head letter');
check(levelOf('AP Human Geography', true) === 'ap', 'AP is read from the title');
check(levelOf('Biology I Honors', true) === 'honors', 'Honors is read from the title');
check(levelOf('Accel Adv Alg', true) === 'honors', 'Accelerated is honors when the page says so');
check(levelOf('Accel Adv Alg', false) === 'regular', 'and regular when it does not');

// Only courses with a letter can be scaled: a percentage mark is a printed value,
// not something Pace converts into a letter on a scale it was never given.
const scaled = summary.scaled;
check(scaled.length === 0, 'a transcript of percentage marks yields no GPA baseline', scaled);

const letters = [
  { credits: 1, level: 'regular' as const, grade: 'A' },
  { credits: 1, level: 'ap' as const, grade: 'A' },
];
check(weightedGpa(DEFAULT_SCALE, letters) === 4.5, 'the scale weights by level', weightedGpa(DEFAULT_SCALE, letters));
check(projectGpa(null, letters, [{ credits: 1, level: 'ap', expectedGrade: 'A' }]).reason === 'no-scale', 'no scale, no projection');
check(projectGpa(DEFAULT_SCALE, letters, []).reason === 'no-courses', 'nothing planned, no projection');
check(projectGpa(DEFAULT_SCALE, [], [{ credits: 1, level: 'ap', expectedGrade: 'A' }]).reason === 'no-history', 'nothing graded, nothing to project from');
const projection = projectGpa(DEFAULT_SCALE, letters, [{ credits: 1, level: 'ap', expectedGrade: 'B' }]);
check(projection.reason === 'ok' && projection.gpa === 4.333, 'a planned B pulls the average down', projection);
check(projection.credits === 3, 'the projection states its basis', projection.credits);

check(parseScale('') === null, 'an unset scale is null');
check(parseScale('{"regular":{}}') === null, 'an incomplete scale is refused');
const stored = parseScale(JSON.stringify(DEFAULT_SCALE));
check(stored?.ap.A === 5, 'a stored scale round-trips', stored?.ap);

if (failures) {
  console.error(`\nDOMAIN FAILED: ${failures} of ${checks} checks`);
  process.exit(1);
}
console.log(`DOMAIN OK: ${checks} checks passed`);
