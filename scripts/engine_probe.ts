/**
 * Run the shipped reader over any text file, on this machine.
 *
 * This is the developer-side instrument for the app's engine: the same
 * TypeScript, the same artifacts, the same ONNX graph, printed as a summary of
 * what the review screen would show. It exists so a document can be checked
 * without building or installing the app.
 *
 *   npx tsx scripts/engine_probe.ts path/to/document.txt [--assume=syllabus]
 *
 * `--assume` is the reviewer override the review screen offers, so a document the
 * gate rejected can be checked against the same reader without editing anything.
 */
import { readFileSync } from 'node:fs';

import { PaceEngine } from '../src/engine/PaceEngine';
import { buildDraft, gradingTotal, importBlocker } from '../src/features/import/draft';
import { assets, createSession } from '../tests/mobile/nodeAssets';

async function main() {
  const path = process.argv[2];
  if (!path) {
    console.error('usage: npx tsx scripts/engine_probe.ts <file.txt>');
    process.exit(2);
  }
  const text = readFileSync(path, 'utf8');
  const assume = process.argv.slice(3).find((arg) => arg.startsWith('--assume='))?.split('=')[1];
  const engine = await PaceEngine.create(assets, createSession);
  const { record, timings } = await engine.understand(text, assume ? { assumeDocumentType: assume } : {});
  const draft = buildDraft(record, {
    source: { kind: 'file', uri: null, label: path },
    text,
    timings: timings as unknown as Record<string, number>,
    evidenceMeta: {},
    image: null,
    match: null,
  });

  const type = record.document.documentType;
  console.log(`document      ${type?.label ?? 'unknown'} (${((type?.confidence ?? 0) * 100).toFixed(1)}%)`);
  console.log(`gate          ${record.document.gate}${type?.abstained ? ' (the classifier abstained)' : ''}`);
  console.log(`characters    ${record.document.characters}`);
  console.log(`sections      ${record.sections.length}`);
  for (const section of record.sections) {
    console.log(`  - ${section.label.padEnd(22)} ${section.heading.slice(0, 48)} (${Math.round(section.confidence * 100)}%)`);
  }
  if (record.transcript) {
    const transcript = record.transcript;
    console.log(
      `transcript    ${transcript.courses.length} courses · ${transcript.terms.length} term${transcript.terms.length === 1 ? '' : 's'} · ${transcript.totalCredits} credits` +
        `${transcript.reportedCredits != null ? ` (printed ${transcript.reportedCredits})` : ''}` +
        `${transcript.reportedGpa != null ? ` · GPA ${transcript.reportedGpa}` : ''}`,
    );
    for (const term of transcript.terms) {
      console.log(`  term        ${term.value} (${term.courses} courses, ${term.credits} credits)`);
    }
    for (const row of transcript.courses) {
      console.log(`  row         ${[row.mark, row.credit, row.code, row.title].filter((value) => value != null).join('  ')}`);
    }
    if (transcript.assessments.length) {
      console.log(
        `  assessments ${transcript.assessments
          .map((row) => `${row.title} ${row.score}${row.label ? ` (${row.level} ${row.label})` : ''}`)
          .join(' | ')}`,
      );
    }
  }
  console.log(`course        ${JSON.stringify(record.course)}`);
  console.log(`grading       ${record.grading.map((row) => `${row.category} ${row.weight}%`).join(', ') || '—'}`);
  console.log(`grading total ${gradingTotal(draft) ?? '—'} (${record.gradingTotalConsistent ?? 'n/a'})`);
  console.log(
    `meetings      ${record.meetings.map((row) => `${row.days.join(' ')} ${row.startTime ?? '?'}-${row.endTime ?? '?'} ${row.room ?? ''}`).join(' | ') || '—'}`,
  );
  console.log(`policies      ${Object.keys(record.policies).join(', ') || '—'}`);
  console.log(`evidence      ${record.evidence.length} rows, ${record.evidence.filter((e) => e.groundedInSource).length} grounded`);
  for (const warning of record.warnings) console.log(`warning       ${warning}`);
  console.log(
    `import        ${importBlocker(draft) ??
      (draft.transcript
        ? `ready (${draft.transcript.rows.length} course rows)`
        : `ready (${draft.grading.length} categories)`)}`,
  );
  console.log(
    `timings       type ${timings.documentTypeMs}ms · sections ${timings.sectionsMs}ms · model ${timings.sectionModelMs}ms · rules ${timings.extractionMs}ms · total ${timings.totalMs}ms`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
