/**
 * The document → import draft step.
 *
 * The engine returns an AcademicRecord, which is a complete, evidence-linked
 * description of what it read. That is not yet something the user can consent to:
 * a person reviews *proposals*. This module turns one into the other, keeping the
 * evidence id of every value so the review screen can open the source line and
 * the committed rows can point back at it.
 *
 * Nothing here decides anything on the user's behalf: every proposal starts
 * `included: true` only because the reviewer is about to see all of them.
 */
import type { AcademicRecord, ConfidenceBand, DocumentType, Proposal, TranscriptRecord } from '../../engine/types';


export interface GradingValue {
  category: string;
  weight: number;
}

export interface MeetingValue {
  days: string[];
  startTime: string | null;
  endTime: string | null;
  room: string | null;
}

export interface EvidenceMeta {
  page: number | null;
  bbox: { x: number; y: number; width: number; height: number } | null;
  ocrConfidence: number | null;
}

export type SourceKind = 'photo' | 'library' | 'file' | 'text';

export interface DraftSource {
  kind: SourceKind;
  uri: string | null;
  label: string;
}

export interface TranscriptRowValue {
  title: string;
  code: string | null;
  mark: string | null;
  credit: number | null;
  term: string | null;
}

/**
 * A transcript draft: every printed row, as one proposal each.
 *
 * The unit of review is a course row, not a document, because that is what a
 * transcript is: eight rows, eight courses, each with its own mark and credit.
 */
export interface TranscriptDraft {
  student: string | null;
  school: string | null;
  rows: Proposal<TranscriptRowValue>[];
  terms: TranscriptRecord['terms'];
  assessments: TranscriptRecord['assessments'];
  totalCredits: number;
  reportedCredits: number | null;
  reportedGpa: number | null;
  layout: TranscriptRecord['layout'];
}

export interface MatchProposal {
  courseId: string;
  title: string;
  code: string | null;
  probability: number;
  decision: 'same' | 'uncertain' | 'different';
}

export interface ImportDraft {
  id: string;
  documentId: string;
  createdAt: string;
  record: AcademicRecord;
  /** The exact text the reader saw: kept so the source sheet can quote it. */
  text: string;
  timings: Record<string, number>;
  source: DraftSource;
  documentType: DocumentType | null;
  academic: boolean;
  title: Proposal<string> | null;
  code: Proposal<string> | null;
  instructor: Proposal<string> | null;
  credits: Proposal<number> | null;
  /** Present when the document was read as a transcript. */
  transcript: TranscriptDraft | null;
  grading: Proposal<GradingValue>[];
  meetings: Proposal<MeetingValue>[];
  policies: Proposal<string>[];
  warnings: string[];
  /** evidenceId → where it came from on the page, when OCR geometry exists. */
  evidenceMeta: Record<string, EvidenceMeta>;
  /** The photographed page, when there is one, for the source sheet. */
  image: { uri: string; width: number; height: number } | null;
  /** Pages read from the file (1 for a photograph). */
  pageCount: number;
  match: MatchProposal | null;
  linkToCourseId: string | null;
  /** Set when this exact page is already imported, so it cannot be silently doubled. */
  duplicate: { importId: string; courseId: string | null } | null;
}

function band(confidence: number): ConfidenceBand {
  // Bands, not percentages: these numbers are rule confidences, and the app never
  // claims a calibrated probability for a value it read with a pattern.
  if (confidence >= 0.9) return 'high';
  if (confidence >= 0.7) return 'medium';
  return 'low';
}

function evidenceFor(record: AcademicRecord, field: string, start: number | null | undefined) {
  if (start == null) return null;
  return record.evidence.find((item) => item.field === field && item.start === start) ?? null;
}

function entityFor(record: AcademicRecord, field: string) {
  return record.entities.find((entity) => entity.field === field) ?? null;
}

function entityForEvidence(record: AcademicRecord, evidenceId: string | null | undefined) {
  if (!evidenceId) return null;
  return record.entities.find((entity) => entity.evidenceId === evidenceId) ?? null;
}

function proposal<T>(args: {
  key: string;
  value: T;
  confidence: number;
  evidenceId: string | null;
  evidenceText: string | null;
  start: number | null;
  end: number | null;
  entityId: string | null;
  source?: 'rule' | 'model';
  warning?: string;
}): Proposal<T> {
  return {
    key: args.key,
    entityId: args.entityId,
    evidenceId: args.evidenceId,
    value: args.value,
    confidence: args.confidence,
    band: band(args.confidence),
    source: args.source ?? 'rule',
    evidenceText: args.evidenceText,
    start: args.start,
    end: args.end,
    included: true,
    edited: false,
    originalValue: args.value,
    ...(args.warning ? { warning: args.warning } : {}),
  };
}

function courseProposal<T>(
  record: AcademicRecord,
  field: 'course_title' | 'course_code' | 'instructor' | 'credits',
  value: T,
): Proposal<T> | null {
  const entity = entityFor(record, field);
  // Follow the entity's own evidence first. A document can carry two spans for the
  // same field (a page title and a catalogue header); taking the first by field
  // would highlight a line the value did not come from.
  const evidence =
    (entity?.evidenceId ? record.evidence.find((item) => item.id === entity.evidenceId) : null) ??
    record.evidence.find((item) => item.field === field) ??
    null;
  if (!entity && !evidence) return null;
  return proposal<T>({
    key: field,
    value,
    confidence: evidence?.confidence ?? entity?.confidence ?? 0.8,
    evidenceId: evidence?.id ?? entity?.evidenceId ?? null,
    evidenceText: evidence?.sourceText ?? null,
    start: evidence?.start ?? null,
    end: evidence?.end ?? null,
    entityId: entity?.id ?? null,
  });
}

export function buildDraft(
  record: AcademicRecord,
  meta: {
    source: DraftSource;
    text: string;
    timings: Record<string, number>;
    evidenceMeta: Record<string, EvidenceMeta>;
    image?: { uri: string; width: number; height: number } | null;
    pageCount?: number;
    match: MatchProposal | null;
    duplicate?: { importId: string; courseId: string | null } | null;
  },
): ImportDraft {
  const course = record.course as Record<string, string | number> | undefined;
  const title = course?.course_title != null ? courseProposal(record, 'course_title', String(course.course_title)) : null;
  const code = course?.course_code != null ? courseProposal(record, 'course_code', String(course.course_code)) : null;
  const instructor =
    course?.instructor != null ? courseProposal(record, 'instructor', String(course.instructor)) : null;
  const credits =
    course?.credits != null ? courseProposal(record, 'credits', Number(course.credits)) : null;

  const transcript = buildTranscriptDraft(record);

  const grading: Proposal<GradingValue>[] = record.grading.map((row) => {
    const evidence = evidenceFor(record, 'grading', row.evidence.start);
    return proposal<GradingValue>({
      key: `grading:${row.id}`,
      value: { category: row.category, weight: row.weight },
      confidence: row.confidence,
      evidenceId: evidence?.id ?? null,
      evidenceText: evidence?.sourceText ?? row.evidence.text,
      start: row.evidence.start,
      end: row.evidence.end,
      entityId: entityForEvidence(record, evidence?.id)?.id ?? null,
    });
  });

  const meetings: Proposal<MeetingValue>[] = record.meetings.map((meeting) => {
    const evidence = evidenceFor(record, 'meetings', meeting.evidence.start);
    return proposal<MeetingValue>({
      key: `meeting:${meeting.id}`,
      value: {
        days: meeting.days,
        startTime: meeting.startTime,
        endTime: meeting.endTime,
        room: meeting.room,
      },
      confidence: meeting.confidence,
      evidenceId: evidence?.id ?? null,
      evidenceText: evidence?.sourceText ?? meeting.evidence.text,
      start: meeting.evidence.start,
      end: meeting.evidence.end,
      entityId: entityForEvidence(record, evidence?.id)?.id ?? null,
    });
  });

  const policies: Proposal<string>[] = Object.entries(record.policies).map(([field, policy]) => {
    const evidence = evidenceFor(record, field, policy.evidence.start);
    return proposal<string>({
      key: `policy:${field}`,
      value: policy.value ?? '',
      confidence: policy.confidence,
      // A policy id and an evidence id are different namespaces. If the embedded
      // policy span cannot be resolved to a record.evidence row, keep the link
      // absent instead of storing a dangling provenance reference.
      evidenceId: evidence?.id ?? null,
      evidenceText: evidence?.sourceText ?? policy.evidence.text,
      start: policy.evidence.start,
      end: policy.evidence.end,
      entityId: entityForEvidence(record, evidence?.id)?.id ?? null,
      ...(policy.reason ? { warning: policy.reason } : {}),
    });
  });

  return {
    id: `draft-${record.document.id.slice(0, 16)}-${Date.now().toString(36)}`,
    documentId: record.document.id,
    createdAt: new Date().toISOString(),
    record,
    text: meta.text,
    timings: meta.timings,
    source: meta.source,
    documentType: record.document.documentType,
    academic: record.document.gate === 'open' && (record.document.documentType?.academic ?? true),
    title,
    code,
    instructor,
    credits,
    transcript,
    grading,
    meetings,
    policies,
    warnings: record.warnings,
    evidenceMeta: meta.evidenceMeta,
    image: meta.image ?? null,
    pageCount: meta.pageCount ?? 1,
    match: meta.match,
    linkToCourseId: null,
    duplicate: meta.duplicate ?? null,
  };
}

function buildTranscriptDraft(record: AcademicRecord): TranscriptDraft | null {
  const transcript = record.transcript;
  if (!transcript) return null;
  return {
    student: transcript.student,
    school: transcript.school,
    terms: transcript.terms,
    assessments: transcript.assessments,
    totalCredits: transcript.totalCredits,
    reportedCredits: transcript.reportedCredits,
    reportedGpa: transcript.reportedGpa,
    layout: transcript.layout,
    rows: transcript.courses.map((row) => {
      const evidence = record.evidence.find((item) => item.id === row.evidenceId) ?? null;
      return proposal<TranscriptRowValue>({
        key: `transcript:${row.id}`,
        value: { title: row.title, code: row.code, mark: row.mark, credit: row.credit, term: row.term },
        confidence: row.confidence,
        evidenceId: row.evidenceId || null,
        evidenceText: evidence?.sourceText ?? null,
        start: evidence?.start ?? null,
        end: evidence?.end ?? null,
        entityId: null,
      });
    }),
  };
}

/** Proposals that will be committed. Order is the order they are shown in. */
export function includedGrading(draft: ImportDraft): Proposal<GradingValue>[] {
  return draft.grading.filter((row) => row.included);
}

export function includedTranscript(draft: ImportDraft): Proposal<TranscriptRowValue>[] {
  return draft.transcript?.rows.filter((row) => row.included) ?? [];
}

export function gradingTotal(draft: ImportDraft): number | null {
  const rows = includedGrading(draft);
  if (!rows.length) return null;
  return Math.round(rows.reduce((sum, row) => sum + row.value.weight, 0) * 100) / 100;
}

/**
 * Why the Import button is disabled, if it is.
 *
 * A draft with no course name cannot become a course; everything else can be
 * imported as-is, because a course with only grading is still real information.
 */
export function importBlocker(draft: ImportDraft): string | null {
  if (draft.duplicate) return 'This page is already imported.';
  if (!draft.academic) return 'Pace did not find academic information in this document.';
  if (draft.transcript) {
    const rows = includedTranscript(draft);
    if (!rows.length) return 'Keep at least one course row to import.';
    return null;
  }
  const hasName = Boolean(draft.title?.included || draft.code?.included);
  if (!hasName && !draft.linkToCourseId) return 'Add a course title or code before importing.';
  const importedValues =
    Number(Boolean(draft.title?.included)) +
    Number(Boolean(draft.code?.included)) +
    Number(Boolean(draft.instructor?.included)) +
    Number(Boolean(draft.credits?.included)) +
    includedGrading(draft).length +
    draft.meetings.filter((meeting) => meeting.included).length +
    draft.policies.filter((policy) => policy.included).length;
  if (importedValues === 0) return 'Nothing is selected to import.';
  return null;
}

/** A one-line summary for the recent-imports list. */
export function draftHeadline(draft: ImportDraft): string {
  if (draft.transcript) {
    return draft.transcript.student ?? draft.transcript.school ?? 'Transcript';
  }
  const title = draft.title?.included ? draft.title.value : null;
  const code = draft.code?.included ? draft.code.value : null;
  return title ?? code ?? 'Untitled document';
}

/** The review screen's counts: how many values, how many need a decision. */
export function reviewCounts(draft: ImportDraft) {
  if (draft.transcript) {
    const rows = draft.transcript.rows.length;
    const needsReview = draft.transcript.rows.filter((row) => row.band === 'low' || row.band === 'medium').length;
    return { total: rows, needsReview };
  }
  const all = [
    ...(draft.title ? [draft.title] : []),
    ...(draft.code ? [draft.code] : []),
    ...(draft.instructor ? [draft.instructor] : []),
    ...(draft.credits ? [draft.credits] : []),
    ...draft.grading,
    ...draft.meetings,
    ...draft.policies,
  ];
  const needsReview = all.filter((item) => item.band === 'low' || item.band === 'medium').length;
  return { total: all.length, needsReview };
}