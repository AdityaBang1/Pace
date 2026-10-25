/**
 * The record the application consumes. Mirrors `pace.academic-ir/0.3`.
 *
 * Nothing here mentions tensors, logits or tokenisers: the UI is written against
 * these types, and the engine is free to change underneath them.
 */

export const SCHEMA_VERSION = 'pace.academic-ir/0.3';

export type ConfidenceBand = 'high' | 'medium' | 'low';

export interface Evidence {
  id: string;
  field: string;
  value: string | number | null;
  confidence: number;
  confidenceSource: 'rule' | 'model';
  band: ConfidenceBand | null;
  sourceText: string | null;
  start: number | null;
  end: number | null;
  groundedInSource: boolean;
  requiresConfirmation: true;
}

export interface Entity {
  id: string;
  kind:
    | 'course'
    | 'course_code'
    | 'instructor'
    | 'credit'
    | 'grade_category'
    | 'meeting'
    | 'policy';
  field: string;
  value: string | number | null;
  weight?: number | null;
  confidence: number;
  confidenceSource: 'rule' | 'model';
  band: ConfidenceBand | null;
  evidenceId: string;
  requiresConfirmation: true;
}

export interface Relation {
  id: string;
  type: 'WEIGHT_OF' | 'TAUGHT_BY' | 'MEETS_ON';
  from: string;
  to: string;
  weight?: number;
  evidenceId: string;
  predicted: false;
  requiresConfirmation: true;
}

export interface Section {
  id: string;
  heading: string;
  label: string;
  task: string;
  confidence: number;
  band: ConfidenceBand;
  confidenceSource: 'model' | 'heading-only';
  start: number;
  end: number;
  sourceText: string;
}

export interface GradingEntry {
  id: string;
  category: string;
  weight: number;
  confidence: number;
  confidenceSource: 'rule';
  evidence: { text: string; start: number; end: number };
}

export interface Meeting {
  id: string;
  days: string[];
  startTime: string | null;
  endTime: string | null;
  room: string | null;
  alternatives: boolean;
  confidence: number;
  confidenceSource: 'rule';
  evidence: { text: string; start: number; end: number };
}

export interface PolicyEntry {
  id: string;
  value: string | null;
  confidence: number;
  confidenceSource: 'rule';
  reason: string | null;
  evidence: { text: string; start: number; end: number };
}

export interface TranscriptCourseEntry {
  id: string;
  term: string | null;
  code: string | null;
  title: string;
  /** As printed: "94", "A-", "P". */
  mark: string | null;
  /** The mark as a percentage, when it was printed as one. */
  markPercent: number | null;
  credit: number | null;
  /** The subject area printed above the row, when there was one. */
  section: string | null;
  confidence: number;
  evidenceId: string;
  requiresConfirmation: true;
}

export interface TranscriptTerm {
  value: string;
  courses: number;
  credits: number;
}

export interface TranscriptAssessmentEntry {
  title: string;
  score: number;
  level: number | null;
  label: string | null;
}

/**
 * A transcript, as the app reviews it.
 *
 * One course per printed row, grouped by the term banner above it. This is the
 * shape an academic history actually has; the syllabus fields (`course`,
 * `grading`, `meetings`) are left empty for a transcript because none of them
 * describes it.
 */
export interface TranscriptRecord {
  layout: 'course_mark_weight_credit' | 'dated_rows';
  student: string | null;
  school: string | null;
  documentTitle: string | null;
  terms: TranscriptTerm[];
  courses: TranscriptCourseEntry[];
  assessments: TranscriptAssessmentEntry[];
  totalCredits: number;
  /** The total the page printed, when it printed one. */
  reportedCredits: number | null;
  reportedGpa: number | null;
  /**
   * One entry per table the page printed, with that table's own footer totals.
   * A term-per-table transcript has no document total to compare against, so each
   * table is checked against the credits it printed for itself.
   */
  tableTotals?: TranscriptTableTotal[];
  checks: Record<string, number | boolean | string | null>;
  requiresConfirmation: true;
}

export interface TranscriptTableTotal {
  term: string | null;
  rows: number;
  credits: number;
  reportedCredits: number | null;
  reportedGpa: number | null;
}

export interface DocumentType {
  label: string;
  proposedLabel: string;
  confidence: number;
  academic: boolean;
  abstained: boolean;
  top: { label: string; probability: number }[];
}

export interface AcademicRecord {
  schemaVersion: string;
  document: {
    id: string;
    characters: number;
    cleanedCharacters: number;
    sectionsDetected: number;
    sectionsSuppressed: number;
    gate: 'open' | 'blocked';
    documentType: DocumentType | null;
    documentTypeUnavailableBecause: string | null;
    pageCount?: number;
  };
  course: Record<string, string | number>;
  /** Present when the document was read as an academic transcript. */
  transcript: TranscriptRecord | null;
  grading: GradingEntry[];
  gradingTotalPercent: number | null;
  gradingTotalConsistent: boolean | null;
  meetings: Meeting[];
  policies: Record<string, PolicyEntry>;
  sections: Section[];
  entities: Entity[];
  relations: Relation[];
  evidence: Evidence[];
  suppressed: { field: string; value: unknown; reason: string; evidence: string }[];
  confidence: Record<string, unknown>;
  model: {
    engineVersion: string;
    checkpoint: string;
    family: string;
    parameters: number;
    maxStateTokens: number;
    sectionSource: string;
  };
  warnings: string[];
  requiresConfirmation: true;
  commitPolicy: 'proposals_only_user_confirmation_required';
}

/** One proposed value, as the review screen sees it. */
export interface Proposal<T> {
  key: string;
  entityId: string | null;
  evidenceId: string | null;
  value: T;
  confidence: number;
  band: ConfidenceBand | null;
  /** Rule values have a number, not a probability; the UI says so. */
  source: 'rule' | 'model';
  evidenceText: string | null;
  start: number | null;
  end: number | null;
  included: boolean;
  edited: boolean;
  originalValue: T;
  warning?: string;
}
