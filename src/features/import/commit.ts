/**
 * The draft → rows mapping.
 *
 * Kept out of the context so it can be unit-tested without React or SQLite: give
 * it a draft and it returns exactly what `commitImport` will write. Rejected
 * proposals are absent here, which is how "Exclude" reaches the database.
 */
import type {
  CommitInput,
  EvidenceInput,
  GradingInput,
  MeetingInput,
  PolicyInput,
} from '../../database/repo';

import { includedGrading, type ImportDraft } from './draft';

export function draftToCommitInput(draft: ImportDraft): CommitInput {
  const grading: GradingInput[] = includedGrading(draft).map((row, index) => ({
    id: `${draft.id}-g${index}`,
    name: row.value.category,
    weight: row.value.weight,
    position: index,
    evidenceId: row.evidenceId,
  }));

  const meetings: MeetingInput[] = draft.meetings
    .filter((row) => row.included)
    .map((row, index) => ({
      id: `${draft.id}-m${index}`,
      days: row.value.days,
      startTime: row.value.startTime,
      endTime: row.value.endTime,
      room: row.value.room,
      alternatives: false,
      evidenceId: row.evidenceId,
    }));

  const policies: PolicyInput[] = draft.policies
    .filter((row) => row.included)
    .map((row, index) => ({
      id: `${draft.id}-p${index}`,
      field: row.key.replace(/^policy:/, ''),
      value: row.value,
      evidenceId: row.evidenceId,
    }));

  const evidence: EvidenceInput[] = draft.record.evidence.map((item) => {
    const meta = draft.evidenceMeta[item.id];
    return {
      id: item.id,
      field: item.field,
      page: meta?.page ?? null,
      text: item.sourceText,
      start: item.start,
      end: item.end,
      bbox: meta?.bbox ?? null,
      ocrConfidence: meta?.ocrConfidence ?? null,
      modelConfidence: item.confidenceSource === 'model' ? item.confidence : null,
      confidenceSource: item.confidenceSource,
      extractionMethod: item.confidenceSource === 'rule' ? 'deterministic' : 'model',
    };
  });

  const total = includedGrading(draft).reduce((sum, row) => sum + row.value.weight, 0);
  const consistent = grading.length ? Math.abs(total - 100) <= 5 : null;

  return {
    importId: draft.id,
    linkToCourseId: draft.linkToCourseId,
    courseId: `course-${draft.id}`,
    course: {
      title: draft.title?.included ? draft.title.value : null,
      code: draft.code?.included ? draft.code.value : null,
      instructor: draft.instructor?.included ? draft.instructor.value : null,
      credits: draft.credits?.included ? draft.credits.value : null,
      gradingTotalPercent: grading.length ? Math.round(total * 100) / 100 : null,
      gradingTotalConsistent: consistent,
      documentType: draft.documentType?.label ?? 'other',
    },
    document: {
      id: draft.documentId,
      documentType: draft.documentType?.label ?? 'other',
      academic: draft.academic,
      documentTypeConfidence: draft.documentType?.confidence ?? null,
      sourceKind: draft.source.kind,
      sourceUri: draft.source.uri,
      ocrText: draft.text,
      engineVersion: draft.record.model.engineVersion,
      schemaVersion: draft.record.schemaVersion,
      characterCount: draft.record.document.characters,
      warnings: draft.warnings,
      timings: draft.timings,
    },
    grading,
    meetings,
    policies,
    evidence,
    summary: {
      course: draft.title?.value ?? draft.code?.value ?? null,
      grading: grading.length,
      meetings: meetings.length,
      policies: policies.length,
      warnings: draft.warnings.length,
      linked: Boolean(draft.linkToCourseId),
    },
  };
}
