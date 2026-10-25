/**
 * The most recent run, in memory only.
 *
 * The debug screen is for the developer holding the phone, not the user, so this
 * is deliberately a module-level record that disappears when the app restarts —
 * nothing about timings or document types is written to the database or sent
 * anywhere.
 *
 * The extraction section records how each page was read (the PDF's own text or a
 * recognition of pixels) and keeps the text that was actually handed to the
 * reader. Without it, "the document read badly" cannot be told apart from "the
 * text was never extracted", which are different bugs.
 */
import type { AcademicRecord } from '../../engine/types';

export interface LastRunPage {
  page: number;
  method: 'text_layer' | 'ocr';
  characters: number;
  /** Mean of Vision's per-line confidences; null for a text layer or an empty page. */
  confidence: number | null;
}

export interface LastRunExtraction {
  pages: LastRunPage[];
  /** The text the reader was given, before the engine normalised it. */
  text: string;
}

export interface LastRun {
  at: string;
  characters: number;
  documentType: AcademicRecord['document']['documentType'];
  sections: { heading: string; label: string; task: string; confidence: number }[];
  warnings: string[];
  timings: Record<string, number>;
  counts: { evidence: number; grading: number; meetings: number; policies: number };
  extraction: LastRunExtraction | null;
}

/** How much of the page text is kept for inspection. A page of syllabus is ~2k. */
const TEXT_PREVIEW = 4000;

let last: LastRun | null = null;

export function recordRun(
  record: AcademicRecord,
  timings: Record<string, number>,
  extraction?: LastRunExtraction,
): void {
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
    extraction: extraction
      ? { pages: extraction.pages, text: extraction.text.slice(0, TEXT_PREVIEW) }
      : null,
  };
}

export function getLastRun(): LastRun | null {
  return last;
}
