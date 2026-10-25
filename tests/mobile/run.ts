/**
 * Python <-> mobile parity.
 *
 * Four checks, because they fail in different places:
 *   1. token ids          -- exact; a tokeniser off by one token is silent
 *   2. section decisions  -- label exact, probability within 1e-4
 *   3. the record         -- values, spans, ids, warnings, gate
 *   4. course matching    -- probability within 1e-3, decision exact
 *
 * Run with `npm run test:engine`. Exit code is the result.
 */
import { PaceEngine } from '../../src/engine/PaceEngine';
import { WordPieceTokenizer } from '../../src/engine/tokenizer';
import { assets, createSession, readFixture, readJson } from './nodeAssets';

interface ParityFile {
  max_length: number;
  temperature: number;
  labels: string[];
  tokens: { text: string; ids: number[] }[];
  sections: { fixture: string; detected: number; headings: { heading: string; label: string; confidence: number }[] }[];
}

type Failure = { where: string; expected: unknown; actual: unknown };

const failures: Failure[] = [];
let checks = 0;

function equal(where: string, expected: unknown, actual: unknown): void {
  checks += 1;
  if (JSON.stringify(expected) !== JSON.stringify(actual)) {
    failures.push({ where, expected, actual });
  }
}

function close(where: string, expected: number, actual: number, tolerance: number): void {
  checks += 1;
  if (!(Math.abs(expected - actual) <= tolerance)) {
    failures.push({ where, expected, actual });
  }
}

const PY_TO_TS_COURSE: Record<string, string> = {
  course_title: 'course_title',
  course_code: 'course_code',
  credits: 'credits',
  instructor: 'instructor',
};

async function main(): Promise<void> {
  const parity = await readJson<ParityFile>('fixtures', 'parity.json');
  const engine = await PaceEngine.create(assets, createSession);

  // 1. tokeniser --------------------------------------------------------- //
  const vocab = await assets.readText('tokenizer/vocab.model');
  const tokenizer = new WordPieceTokenizer(vocab, {
    doLowerCase: engine.manifest.tokenizer.config.do_lower_case,
    tokenizeChineseChars: engine.manifest.tokenizer.config.tokenize_chinese_chars ?? true,
    stripAccents:
      engine.manifest.tokenizer.config.strip_accents ??
      engine.manifest.tokenizer.config.do_lower_case,
  });
  for (const row of parity.tokens) {
    equal(`tokenizer(${JSON.stringify(row.text.slice(0, 24))})`, row.ids,
      tokenizer.encodeIds(row.text, parity.max_length));
  }

  // 2. sections + record -------------------------------------------------- //
  const fixturePairs: [string, string][] = [
    ['syllabus_math.txt', 'syllabus_math.json'],
    ['syllabus_chem.txt', 'syllabus_chem.json'],
    ['schedule.txt', 'schedule.json'],
    ['catalog.txt', 'catalog.json'],
    ['receipt.txt', 'receipt.json'],
  ];

  for (const [fixture, expectedFile] of fixturePairs) {
    const text = await readFixture(fixture);
    const { record } = await engine.understand(text);
    const expected = await readJson<Record<string, any>>('fixtures', 'expected', expectedFile);
    const name = fixture.replace('.txt', '');

    equal(`${name}.schemaVersion`, expected.schema_version, record.schemaVersion);
    equal(`${name}.gate`, expected.document.gate, record.document.gate);
    equal(`${name}.documentType.label`, expected.document.document_type?.label ?? null,
      record.document.documentType?.label ?? null);
    if (expected.document.document_type) {
      close(`${name}.documentType.confidence`, expected.document.document_type.confidence,
        record.document.documentType!.confidence, 1e-4);
    }

    equal(`${name}.course`, expected.course, record.course);
    equal(`${name}.grading`, expected.grading.map((row: any) => [row.category, row.weight, row.confidence]),
      record.grading.map((row) => [row.category, row.weight, row.confidence]));
    equal(`${name}.meetings`, expected.meetings.map((row: any) =>
      [row.days, row.start_time, row.end_time, row.room, row.confidence]),
      record.meetings.map((row) => [row.days, row.startTime, row.endTime, row.room, row.confidence]));
    equal(`${name}.policies`, Object.fromEntries(Object.entries(expected.policies).map(
      ([field, entry]: [string, any]) => [field, [entry.value, entry.confidence]])),
      Object.fromEntries(Object.entries(record.policies).map(([field, entry]) => [field, [entry.value, entry.confidence]])));
    equal(`${name}.gradingTotal`, [expected.grading_total_percent, expected.grading_total_consistent],
      [record.gradingTotalPercent, record.gradingTotalConsistent]);

    equal(`${name}.evidence`, expected.evidence.map((item: any) =>
      [item.field, item.value, item.start, item.end, item.confidence, item.grounded_in_source]),
      record.evidence.map((item) => [item.field, item.value, item.start, item.end, item.confidence, item.groundedInSource]));
    equal(`${name}.evidenceIds`, expected.evidence.map((item: any) => item.id),
      record.evidence.map((item) => item.id));
    equal(`${name}.entities`, expected.entities.map((item: any) => [item.kind, item.field, item.value, item.evidence_id]),
      record.entities.map((item) => [item.kind, item.field, item.value, item.evidenceId]));
    equal(`${name}.relations`, expected.relations.map((item: any) => [item.type, item.from, item.to]),
      record.relations.map((item) => [item.type, item.from, item.to]));
    equal(`${name}.warnings`, expected.warnings, record.warnings);
    equal(`${name}.sections`, expected.sections.map((item: any) => [item.heading, item.label]),
      record.sections.map((item) => [item.heading, item.label]));
    equal(`${name}.documentId`, expected.document.id, record.document.id);
  }

  // 3. section predictions (heading-only and heading+value targets) ------- //
  const sectionTargets = parity.sections;
  const byFixture = new Map(sectionTargets.map((row) => [row.fixture, row]));
  for (const [fixture] of fixturePairs) {
    const target = byFixture.get(fixture);
    if (!target) continue;
    const text = await readFixture(fixture);
    const { record } = await engine.understand(text);
    equal(`${fixture}.sectionCount`, target.headings.length, record.sections.length);
    target.headings.forEach((heading, index) => {
      const actual = record.sections[index];
      if (!actual) return;
      equal(`${fixture}.section[${index}].label`, heading.label, actual.label);
      close(`${fixture}.section[${index}].confidence`, heading.confidence, actual.confidence, 1e-4);
    });
  }

  // 4. course matching ---------------------------------------------------- //
  // Matching is batch-shaped: the character-similarity feature is fitted over the
  // pair set, so a probe's number depends on what else is in the batch. The
  // manifest's probes were produced one call at a time, which is how the app
  // asks about a single candidate, so the harness does the same.
  const manifest = await readJson<any>('assets', 'models', 'manifest.model');
  const probes = manifest.models.entityResolution.probes as
    { left: string; right: string; probability: number; decision: string }[];
  const results = probes.map((probe) => engine.matchCourses(probe.left, probe.right));
  probes.forEach((probe, index) => {
    close(`match(${probe.left}|${probe.right}).probability`, probe.probability,
      results[index].probability, 1e-3);
    equal(`match(${probe.left}|${probe.right}).decision`, probe.decision, results[index].decision);
  });

  report();
}

function report(): void {
  if (!failures.length) {
    console.log(`PARITY OK: ${checks} checks passed`);
    return;
  }
  console.log(`PARITY FAILED: ${failures.length} of ${checks} checks`);
  for (const failure of failures.slice(0, 40)) {
    console.log(`\n- ${failure.where}`);
    console.log(`  python: ${JSON.stringify(failure.expected)}`);
    console.log(`  engine: ${JSON.stringify(failure.actual)}`);
  }
  if (failures.length > 40) console.log(`\n... ${failures.length - 40} more`);
  process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
