/**
 * The most recent run, in memory only.
 *
 * The debug screen is for the developer holding the phone, not the user, so this
 * is deliberately a module-level record that disappears when the app restarts —
 * nothing about timings or document types is written to the database or sent
 * anywhere.
 */
import type { AcademicRecord } from '../../engine/types';

export interface LastRun {
  at: string;
  characters: number;
  documentType: AcademicRecord['document']['documentType'];
  sections: { heading: string; label: string; task: string; confidence: number }[];
  warnings: string[];
  timings: Record<string, number>;
  counts: { evidence: number; grading: number; meetings: number; policies: number };
}

let last: LastRun | null = null;

export function recordRun(record: AcademicRecord, timings: Record<string, number>): void {
  last = {
    at: new Date().toISOString(),
    characters: record.document.characters,
    documentType: record.document.documentType,
    sections: record.sections.map((section) => ({
      heading: section.heading,
      label: section.label,
      task: section.task,
      confidence: section.confidence,
    })),
    warnings: record.warnings,
    timings,
    counts: {
      evidence: record.evidence.length,
      grading: record.grading.length,
      meetings: record.meetings.length,
      policies: Object.keys(record.policies).length,
    },
  };
}

export function getLastRun(): LastRun | null {
  return last;
}
