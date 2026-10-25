/**
 * The record builder: values in, `pace.academic-ir/0.3` out.
 *
 * This mirrors `pace.api.read` statement for statement, including the id
 * derivation, because the app and the Python reference must agree on identity as
 * well as on values. A UI that stored `ent_…` from one engine and `ent_…` from
 * the other as different rows would duplicate a course after a re-scan.
 */

import { documentId, stableId } from './sha256';
import type { RuleTables } from './rules/sections';
import type { GradingRow, MeetingValue, PolicyValue, Value } from './rules/extract';
import type {
  AcademicRecord,
  ConfidenceBand,
  DocumentType,
  Entity,
  Evidence,
  Relation,
  Section,
  TranscriptRecord,
} from './types';
import { SCHEMA_VERSION } from './types';

export const BANDS: [number, ConfidenceBand][] = [
  [0.97, 'high'],
  [0.85, 'medium'],
  [0, 'low'],
];

export function confidenceBand(probability: number): ConfidenceBand {
  for (const [floor, name] of BANDS) if (probability >= floor) return name;
  return 'low';
}

export interface TypedSection {
  heading: string;
  label: string;
  task: string;
  confidence: number;
  start: number;
  end: number;
}

export interface RecordInput {
  text: string;
  cleaned: string;
  docType: DocumentType | null;
  sections: TypedSection[];
  course: Record<string, Value<string | number>>;
  grading: GradingRow[];
  meetings: MeetingValue[];
  policies: Record<string, PolicyValue>;
  tables: RuleTables;
  model: AcademicRecord['model'];
  gradingTotal: { percent: number | null; consistent: boolean | null };
  /** Set when the document was read as a transcript instead of a syllabus. */
  transcript?: { record: TranscriptRecord; evidence: Evidence[] } | null;
}

interface Entry {
  field: string;
  item: Evidence;
  source: Value<string | number>;
}

const COURSE_KINDS: Record<string, Entity['kind']> = {
  course_title: 'course',
  course_code: 'course_code',
  instructor: 'instructor',
  credits: 'credit',
};

export const CALIBRATION = {
  sections_in_distribution_ece: 0.0016,
  sections_unseen_university_ece: 0.1665,
  sections_unseen_university_accuracy: 0.783,
  rule_values_development_precision: 0.917,
  rule_values_holdout_precision: 0.955,
  holdout_documents: 12,
  bands: { high: '>=0.97', medium: '0.85-0.97', low: '<0.85' },
  bands_apply_to: 'section labels only (model probabilities)',
  value_confidences:
    'set by the rule that read the value (0.60-0.97), not a probability; their measured precision is the rule number above',
  note:
    'A high band is a preselect-and-confirm signal, not an assertion. On unseen universities 0.98+ model confidence measured 0.828 precision. Every value this record returns requires a human confirmation.',
};

function pyValue(field: string, value: string | number | null | undefined): string {
  if (value === null || value === undefined) return '';
  // The Python reader stores credits as a float, and str(4.0) is "4.0"; ids must
  // hash the same text, so the formatting is explicit rather than incidental.
  if (field === 'credits' && typeof value === 'number') return value.toFixed(1);
  return String(value);
}

export function buildRecord(input: RecordInput): AcademicRecord {
  const { cleaned, text, docType } = input;
  const docId = documentId(cleaned);
  const blocked = docType !== null && !docType.academic;
  const transcript = blocked ? null : input.transcript?.record ?? null;

  const course = blocked || transcript ? {} : input.course;
  const grading = blocked || transcript ? [] : input.grading;
  const meetings = blocked || transcript ? [] : input.meetings;
  const policies = blocked || transcript ? {} : input.policies;

  const entries: Entry[] = [];
  const push = (field: string, source: Value<string | number>) => {
    const sourceText = source.evidence.text;
    const start = Number.isInteger(source.evidence.start) ? source.evidence.start : null;
    const end = Number.isInteger(source.evidence.end) ? source.evidence.end : null;
    const written = sourceText?.trim();
    let grounded = false;
    if (start !== null && end !== null && start >= 0 && end > start && end <= cleaned.length) {
      grounded = written ? cleaned.slice(start, end).includes(written) : true;
    }
    const value = source.value;
    const evidence: Evidence = {
      id: stableId('evidence', docId, [field, start, end, pyValue(field, value)]),
      field,
      value,
      confidence: source.confidence,
      confidenceSource: 'rule',
      band: null,
      sourceText: sourceText ?? null,
      start,
      end,
      groundedInSource: grounded,
      requiresConfirmation: true,
    };
    entries.push({ field, item: evidence, source });
  };

  for (const field of ['course_title', 'course_code', 'credits', 'instructor']) {
    const value = course[field];
    if (value) push(field, value);
  }
  for (const row of grading) {
    push('grading', {
      value: `${row.category} ${row.weight}%`,
      confidence: row.confidence,
      rule: row.rule,
      evidence: row.evidence,
    });
  }
  for (const meeting of meetings) {
    const label = meeting.days.filter(Boolean).join(' ');
    const clock = `${meeting.startTime ?? '?'}-${meeting.endTime ?? '?'}`;
    push('meetings', {
      value: `${label} ${clock}`.trim(),
      confidence: meeting.confidence,
      rule: 'meeting_line',
      evidence: meeting.evidence,
    });
  }
  for (const [field, policy] of Object.entries(policies)) {
    if (policy.value === null) continue;
    push(field, {
      value: policy.value,
      confidence: policy.confidence,
      rule: 'policy_body',
      evidence: policy.evidence,
    });
  }
  // Transcript rows are evidence of their own: each one points at the printed
  // line it was read from, and the committed course keeps that link.
  for (const row of input.transcript?.evidence ?? []) {
    entries.push({ field: row.field, item: row, source: {
      value: row.value ?? '',
      confidence: row.confidence,
      rule: 'transcript_row',
      evidence: { text: row.sourceText ?? '', start: row.start ?? 0, end: row.end ?? 0 },
    } });
  }

  const sections: Section[] = blocked || transcript
    ? []
    : input.sections.map((section, index) => ({
        id: stableId('section', docId, [index, section.start, section.label]),
        heading: section.heading,
        label: section.label,
        task: section.task,
        confidence: Math.round(section.confidence * 10000) / 10000,
        band: confidenceBand(section.confidence),
        confidenceSource: 'model',
        start: section.start,
        end: section.end,
        sourceText: cleaned.slice(section.start, section.end),
      }));

  const { entities, relations } = entitiesAndRelations(entries, grading, meetings, docId);

  const suppressed = blocked
    ? gateSuppressed(input)
    : [];
  const warnings = blocked
    ? [
        `this does not look like an academic document (${docType?.label} ${(docType?.confidence ?? 0).toFixed(2)}); extraction was suppressed and no academic fields are returned`,
      ]
    : transcript
      ? transcriptWarnings({ transcript, docType, cleaned, entries })
      : warningsFor({ course, grading, meetings, sections, cleaned, entries, gradingTotal: input.gradingTotal });

  return {
    schemaVersion: SCHEMA_VERSION,
    document: {
      id: docId,
      characters: text.length,
      cleanedCharacters: cleaned.length,
      sectionsDetected: sections.length,
      sectionsSuppressed: blocked ? input.sections.length : 0,
      gate: blocked ? 'blocked' : 'open',
      documentType: docType,
      documentTypeUnavailableBecause: docType === null ? 'document gate artifact not loaded' : null,
    },
    course: Object.fromEntries(Object.entries(course).map(([key, value]) => [key, value.value])),
    transcript,
    grading: grading.map((row) => ({
      id: stableId('grading', docId, [row.category, row.weight, row.evidence.start]),
      category: row.category,
      weight: row.weight,
      confidence: row.confidence,
      confidenceSource: 'rule',
      evidence: row.evidence,
    })),
    gradingTotalPercent: input.gradingTotal.percent,
    gradingTotalConsistent: input.gradingTotal.consistent,
    meetings: meetings.map((meeting) => ({
      id: stableId('meeting', docId, [meeting.days.join(','), meeting.startTime, meeting.endTime, meeting.evidence.start]),
      days: meeting.days,
      startTime: meeting.startTime,
      endTime: meeting.endTime,
      room: meeting.room,
      alternatives: meeting.alternatives,
      confidence: meeting.confidence,
      confidenceSource: 'rule',
      evidence: meeting.evidence,
    })),
    policies: Object.fromEntries(
      Object.entries(policies)
        .filter(([, policy]) => policy.value !== null)
        .map(([field, policy]) => [
          field,
          {
            id: stableId('policy', docId, [field, policy.evidence.start]),
            value: policy.value,
            confidence: policy.confidence,
            confidenceSource: 'rule',
            reason: policy.reason,
            evidence: policy.evidence,
          },
        ]),
    ),
    sections,
    entities,
    relations,
    evidence: entries.map((entry) => entry.item),
    suppressed,
    confidence: CALIBRATION,
    model: input.model,
    warnings,
    requiresConfirmation: true,
    commitPolicy: 'proposals_only_user_confirmation_required',
  };
}

function entitiesAndRelations(
  entries: Entry[],
  grading: GradingRow[],
  meetings: MeetingValue[],
  docId: string,
): { entities: Entity[]; relations: Relation[] } {
  const entities: Entity[] = [];
  const relations: Relation[] = [];
  const byField = new Map<string, Entity>();

  entries.forEach((entry) => {
    const { field, item } = entry;
    // A transcript row is evidence for a course, not an entity of its own: the
    // committed course is built from the row, and an entity per row would be a
    // second, parallel identity for the same thing.
    if (field === 'transcript_course') return;
    const kind: Entity['kind'] =
      field === 'grading' ? 'grade_category' : field === 'meetings' ? 'meeting' : COURSE_KINDS[field] ?? 'policy';
    const entity: Entity = {
      id: stableId('entity', docId, [kind, field, item.start, pyValue(field, item.value)]),
      kind,
      field,
      value: item.value,
      confidence: item.confidence,
      confidenceSource: 'rule',
      band: null,
      evidenceId: item.id,
      requiresConfirmation: true,
    };
    if (field === 'grading') {
      const row = grading.find((candidate) => `${candidate.category} ${candidate.weight}%` === item.value);
      entity.weight = row?.weight ?? null;
    }
    entities.push(entity);
    if (!byField.has(field)) byField.set(field, entity);

    if (field === 'grading') {
      relations.push({
        id: stableId('relation', docId, ['WEIGHT_OF', entity.id]),
        type: 'WEIGHT_OF',
        from: entity.id,
        to: `${entity.weight}%`,
        weight: entity.weight ?? undefined,
        evidenceId: item.id,
        predicted: false,
        requiresConfirmation: true,
      });
    } else if (field === 'meetings') {
      const meeting = meetings.find((candidate) => candidate.evidence.start === item.start);
      const clock = `${meeting?.startTime ?? ''}-${meeting?.endTime ?? ''}`;
      relations.push({
        id: stableId('relation', docId, ['MEETS_ON', entity.id]),
        type: 'MEETS_ON',
        from: entity.id,
        to: `${(meeting?.days ?? []).join(' ')} ${clock}`.trim(),
        evidenceId: item.id,
        predicted: false,
        requiresConfirmation: true,
      });
    }
  });

  const course = byField.get('course_title') ?? byField.get('course_code');
  const instructor = byField.get('instructor');
  if (course && instructor) {
    relations.push({
      id: stableId('relation', docId, ['TAUGHT_BY', course.id, instructor.id]),
      type: 'TAUGHT_BY',
      from: course.id,
      to: instructor.id,
      evidenceId: instructor.evidenceId,
      predicted: false,
      requiresConfirmation: true,
    });
  }
  return { entities, relations };
}

function transcriptWarnings(args: {
  transcript: TranscriptRecord;
  docType: DocumentType | null;
  cleaned: string;
  entries: Entry[];
}): string[] {
  const { transcript } = args;
  const warnings: string[] = [];
  const rows = transcript.courses.length;
  const terms = transcript.terms.length;
  warnings.push(
    `read as a transcript: ${rows} course${rows === 1 ? '' : 's'}${
      terms ? ` across ${terms} term${terms === 1 ? '' : 's'}` : ''
    }, ${transcript.totalCredits} credit${transcript.totalCredits === 1 ? '' : 's'}; the syllabus readers were not run`,
  );
  if (transcript.reportedCredits !== null && Math.abs(transcript.totalCredits - transcript.reportedCredits) >= 0.001) {
    warnings.push(
      `the rows add up to ${transcript.totalCredits} credits but the transcript prints ${transcript.reportedCredits}: a row may have been missed`,
    );
  }
  // One table per term prints its own credits under itself. Comparing the whole
  // document against one term's footer would report a missing row that is not
  // missing, so each table is held to its own printed total instead.
  for (const table of transcript.tableTotals ?? []) {
    if (table.reportedCredits === null) continue;
    if (Math.abs(table.credits - table.reportedCredits) < 0.001) continue;
    const where = table.term ? `${table.term}: ` : '';
    warnings.push(
      `${where}${table.rows} row${table.rows === 1 ? '' : 's'} add up to ${table.credits} credits but the table prints ${table.reportedCredits}: a row may have been missed`,
    );
  }
  if (!terms) {
    warnings.push('no term heading was printed above the rows, so the courses are not grouped by term');
  }
  const noGrade = transcript.courses.filter((row) => row.mark === null).length;
  if (noGrade) {
    warnings.push(`${noGrade} course row${noGrade === 1 ? ' has' : 's have'} no mark: the printed mark column may not have been read`);
  }
  if (!rows) {
    warnings.push('no transcript row was recognised: the table may have lost its columns or its marks');
  }
  // The gate has four classes and none of them is a transcript, so it always
  // proposed something else. Saying so keeps a 0.85 "course_catalog" from
  // reading like a confident label for this page.
  if (args.docType && args.docType.proposedLabel !== 'transcript') {
    warnings.push(
      `the document gate has no transcript class: it proposed "${args.docType.proposedLabel}" (${args.docType.confidence.toFixed(2)}) and the transcript reader set the type`,
    );
  }
  const ungrounded = args.entries.filter((entry) => entry.item.start !== null && !entry.item.groundedInSource);
  if (ungrounded.length) {
    warnings.push(`${ungrounded.length} value(s) do not match their own source span`);
  }
  if (args.cleaned.length > 60000) {
    warnings.push(
      'very long transcript: every row the table reader can see is returned, nothing beyond it is silently summarised',
    );
  }
  return warnings;
}

function gateSuppressed(input: RecordInput): AcademicRecord['suppressed'] {  const out: AcademicRecord['suppressed'] = [];
  const add = (field: string, value: unknown, text: string | undefined) =>
    out.push({ field, value, reason: 'document_type_not_academic', evidence: (text ?? '').slice(0, 120) });
  for (const [field, value] of Object.entries(input.course)) add(field, value.value, value.evidence.text);
  for (const row of input.grading) add('grading', `${row.category} ${row.weight}%`, row.evidence.text);
  for (const meeting of input.meetings) add('meetings', meeting.startTime, meeting.evidence.text);
  for (const [field, policy] of Object.entries(input.policies)) add(field, policy.value, policy.evidence.text);
  return out;
}

function warningsFor(args: {
  course: Record<string, Value<string | number>>;
  grading: GradingRow[];
  meetings: MeetingValue[];
  sections: Section[];
  cleaned: string;
  entries: Entry[];
  gradingTotal: { percent: number | null; consistent: boolean | null };
}): string[] {
  const warnings: string[] = [];
  if (!args.sections.length) {
    warnings.push('no section heading was recognised; fields were read from whole lines');
  }
  if (args.grading.length && args.gradingTotal.consistent === false) {
    warnings.push(
      `grading weights sum to ${args.gradingTotal.percent}, not 100: the document may print nested sub-weights or the reader may have missed a row`,
    );
  }
  const ungrounded = args.entries.filter((entry) => entry.item.start !== null && !entry.item.groundedInSource);
  if (ungrounded.length) {
    warnings.push(
      `${ungrounded.length} value(s) do not match their own source span: ${JSON.stringify(
        Array.from(new Set(ungrounded.map((entry) => entry.field))).sort(),
      )}`,
    );
  }
  if (!Object.keys(args.course).length && !args.grading.length && !args.meetings.length) {
    warnings.push("nothing was recognised as this document's course, grading or meetings");
  }
  if (args.cleaned.length > 60000) {
    warnings.push(
      'very long document: sections are read up to the detector\'s limit, nothing beyond it is silently summarised',
    );
  }
  return warnings;
}
