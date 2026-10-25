/**
 * One scan, start to finish.
 *
 * The stages reported here are the stages that actually run, in order, with no
 * invented percentages: the review screen can only show "Syllabus detected" after
 * the document gate has produced a label, and only show "Text recognised" after
 * the OCR boundary returned lines.
 *
 * A file may be a photograph or a PDF. Both arrive as pages of OCR text with line
 * geometry; the reader itself never knows which it was.
 */
import { Directory, File, Paths } from 'expo-file-system';

import type { PaceEngine, UnderstandResult } from '../../engine/PaceEngine';
import { loadEngine } from '../../engine/native';
import { recognizeDocument, type PaceVisionDocument } from '../../../modules/pace-vision';
import type { DraftSource, EvidenceMeta } from '../import/draft';
import { evidenceMetaFrom, joinPages, lineGeometry } from './geometry';
import { recordRun } from './lastRun';

export type ScanStage = 'preparing' | 'reading' | 'understanding';

export interface ScanProgress {
  stage: ScanStage;
  detail?: string;
}

export interface ScanOutcome {
  engine: PaceEngine;
  result: UnderstandResult;
  ocr: PaceVisionDocument;
  source: DraftSource;
  evidenceMeta: Record<string, EvidenceMeta>;
  image: { uri: string; width: number; height: number } | null;
  pageCount: number;
  timings: Record<string, number>;
}

export class ScanError extends Error {
  readonly hint: string;
  constructor(message: string, hint: string) {
    super(message);
    this.name = 'ScanError';
    this.hint = hint;
  }
}

const MIN_TEXT = 40;

/**
 * Copy the source file into private storage.
 *
 * The picker hands back a cache file the system may purge, and the source sheet
 * has to still find the document a week later. A photograph is a few MB; a PDF is
 * kept for the same reason but is not re-rendered, so its pages are still quoted
 * from text rather than highlighted.
 */
async function persistSource(
  uri: string,
  documentHint: string,
  kind: 'image' | 'pdf',
  keep: boolean,
): Promise<string | null> {
  // When the policy is "don't keep", the picker's cache file is all we have, and
  // the system may purge it at any time. Writing that path into the document row
  // would leave provenance pointing at a file that no longer exists, so the row
  // stores no URI at all — the quoted line is what remains.
  if (!keep) return null;
  try {
    const directory = new Directory(Paths.document, kind === 'pdf' ? 'files' : 'pages');
    if (!directory.exists) directory.create({ intermediates: true, idempotent: true });
    const lower = uri.toLowerCase();
    const extension =
      kind === 'pdf' ? 'pdf' : lower.endsWith('.png') ? 'png' : lower.endsWith('.heic') ? 'heic' : 'jpg';
    const target = new File(directory, `${documentHint}.${extension}`);
    const source = new File(uri);
    // A zero-length or short file is the wreck of an interrupted copy; `exists`
    // alone would keep it forever and the source sheet would show a broken image.
    if (!target.exists || target.size === 0 || target.size !== source.size) {
      await source.copy(target, { overwrite: true });
    }
    if (target.size !== source.size) return null;
    return target.uri;
  } catch {
    // Losing the copy is not fatal: the text was already read, and the review
    // screen simply has no image to show.
    return null;
  }
}

/**
 * A plain-text document.
 *
 * No OCR, no geometry: the file already is the text the reader wants, so the same
 * engine runs with an empty geometry map and the source sheet shows the line
 * without a photograph behind it.
 */
export async function scanText(args: {
  text: string;
  label: string;
  onProgress?: (progress: ScanProgress) => void;
}): Promise<ScanOutcome> {
  const { text, label, onProgress } = args;
  const started = Date.now();
  if (text.trim().length < MIN_TEXT) {
    throw new ScanError(
      'That file has almost no text in it.',
      'Choose a document with text in it, or photograph the page instead.',
    );
  }

  onProgress?.({ stage: 'preparing', detail: 'Loading the reader' });
  const engine = await loadEngine();
  onProgress?.({ stage: 'understanding', detail: 'Reading the academic structure' });
  const { record, text: normalized, timings } = await engine.understand(text);
  const measured = { ...(timings as unknown as Record<string, number>), totalMs: Date.now() - started };
  recordRun(record, measured);

  return {
    engine,
    result: { record, text: normalized, timings },
    ocr: {
      kind: 'image',
      pages: [],
      text: normalized,
      pageCount: 1,
      truncated: false,
      durationMs: 0,
    },
    // The text itself is stored with the document; the picked file is a cache copy
    // that iOS is free to delete, so it is not recorded as provenance.
    source: { kind: 'file', uri: null, label },
    evidenceMeta: {},
    image: null,
    pageCount: 1,
    timings: measured,
  };
}

/**
 * Photograph or PDF, one path.
 *
 * `keepSource` is the storage policy from Settings: when it is off the picked file
 * is read and not copied, so provenance keeps its quoted line but loses the image.
 */
export async function scanFile(args: {
  uri: string;
  kind: 'photo' | 'library' | 'file';
  label: string;
  keepSource: boolean;
  onProgress?: (progress: ScanProgress) => void;
}): Promise<ScanOutcome> {
  const { uri, kind, label, keepSource, onProgress } = args;
  const started = Date.now();

  onProgress?.({ stage: 'preparing', detail: 'Loading the reader' });
  const engine = await loadEngine();

  onProgress?.({ stage: 'reading', detail: 'Recognising text' });
  const ocr = await recognizeDocument(uri);
  const { text, starts } = joinPages(ocr.pages);
  if (text.trim().length < MIN_TEXT) {
    throw new ScanError(
      ocr.kind === 'pdf' ? 'Pace could not read text on those pages.' : 'Pace could not read enough text on that page.',
      ocr.kind === 'pdf'
        ? 'Those pages may be scanned images with no readable text. Try a version with selectable text, or photograph the printed page.'
        : 'Move closer, hold the phone steady, and include the whole page — or import a PDF or a photo of the document.',
    );
  }

  onProgress?.({ stage: 'understanding', detail: 'Reading the academic structure' });
  const { record, text: normalized, timings } = await engine.understand(text);
  if (ocr.truncated) {
    record.warnings.push(
      `only the first ${ocr.pageCount} pages were read: the document is longer than the page limit`,
    );
  }

  const geometry = lineGeometry(ocr.pages, starts);
  const evidenceMeta = evidenceMetaFrom(record, geometry);
  const storedUri = await persistSource(uri, record.document.id.slice(0, 16), ocr.kind, keepSource);
  const firstPage = ocr.pages[0];

  const totalMs = Date.now() - started;
  const measured = {
    ...(timings as unknown as Record<string, number>),
    ocrMs: ocr.durationMs,
    pages: ocr.pageCount,
    totalMs,
  };
  recordRun(record, measured);

  return {
    engine,
    result: { record, text: normalized, timings },
    ocr,
    source: { kind, uri: storedUri, label },
    evidenceMeta,
    // A PDF's page image is not retained, and with the storage policy off there is
    // no durable copy of a photograph: in both cases the source sheet quotes the
    // line and names the page instead of highlighting a page we cannot show.
    image:
      ocr.kind === 'image' && firstPage && storedUri
        ? { uri: storedUri, width: firstPage.imageWidth, height: firstPage.imageHeight }
        : null,
    pageCount: ocr.pageCount,
    timings: measured,
  };
}
