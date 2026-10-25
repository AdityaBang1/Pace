/**
 * The one call the application makes.
 *
 * `PaceEngine.understand(text)` runs the whole reader locally: the document
 * gate, section detection, the ONNX section classifier, the deterministic value
 * readers, and the record builder. Callers get an `AcademicRecord`; they never
 * see a tensor, a token or a checkpoint.
 *
 * The engine is constructed with an asset reader and a session factory, so the
 * identical code runs in the Node parity harness and on the phone.
 */

import { DocumentTypeClassifier, type DocumentTypeModel } from './documentType';
import { buildRecord, type TypedSection } from './ir';
import { CourseMatcher, type MatcherModel } from './matcher';
import { SectionClassifier } from './sectionModel';
import { WordPieceTokenizer } from './tokenizer';
import type { SessionFactory } from './runtime';
import { cleanHtml } from './rules/text';
import { readTranscript, transcriptEvidence, buildTranscriptRecord } from './transcript';
import { documentId } from './sha256';
import type { Evidence } from './types';
import { detectSections, sectionBody, type DetectedSection, type RuleTables } from './rules/sections';
import {
  extractCourseCode,
  extractCourseTitle,
  extractCredits,
  extractGrading,
  extractInstructor,
  extractMeetings,
  extractPolicies,
  type GradingRow,
  type Value,
} from './rules/extract';
import type { AcademicRecord } from './types';

export interface EngineAssets {
  readText(name: string): Promise<string>;
  readBinary(name: string): Promise<Uint8Array>;
  /** A path or URI the ONNX runtime can open. */
  modelUri(name: string): Promise<string>;
}

export interface EngineManifest {
  engineVersion: string;
  schemaVersion: string;
  models: {
    section: {
      id: string;
      file: string;
      labels: string[];
      tasks: Record<string, string>;
      max_length: number;
      temperature: number;
      parameters: number;
      bytes: number;
      parity?: { rows: number; label_agreement: number };
    };
    context: {
      id: string;
      file: string;
      labels: string[];
      tasks: Record<string, string>;
      max_length: number;
      temperature: number;
      bytes: number;
    };
    documentType: {
      id: string;
      file: string;
      weights_file: string;
      classes: string[];
      threshold: number;
      bytes: number;
    };
    entityResolution: { id: string; file: string; threshold: number };
    rules: { id: string; file: string };
  };
  tokenizer: {
    dir: string;
    files: Record<string, string>;
    config: {
      do_lower_case: boolean;
      strip_accents: boolean | null;
      tokenize_chinese_chars?: boolean;
    };
  };
}

export interface EngineTimings {
  documentTypeMs: number;
  sectionsMs: number;
  sectionModelMs: number;
  extractionMs: number;
  totalMs: number;
  loadMs?: number;
}

export interface UnderstandResult {
  record: AcademicRecord;
  timings: EngineTimings;
  text: string;
}

export interface UnderstandOptions {
  /**
   * Read the document as this type even though the gate said otherwise.
   *
   * The gate is a proposal too: a photographed syllabus from an unfamiliar
   * institution can land on `other`, and the person holding the page knows
   * better. Supplying a type here re-opens extraction and records that the
   * choice was the user's, not the model's.
   */
  assumeDocumentType?: string;
}

function bestIndex(row: number[]): number {
  let best = 0;
  for (let position = 1; position < row.length; position += 1) {
    if (row[position] > row[best]) best = position;
  }
  return best;
}

export class PaceEngine {
  private constructor(
    readonly manifest: EngineManifest,
    private readonly docType: DocumentTypeClassifier,
    private readonly matcher: CourseMatcher,
    private readonly sectionModel: SectionClassifier,
    /**
     * The heading+value model is loaded on first use, not at startup.
     *
     * It only runs for sections whose heading the primary model is unsure about
     * (~7% of real sections), and it is the same size as the primary graph, so
     * opening it eagerly doubled app launch time for work most documents never
     * need. `null` until the first unfamiliar heading arrives.
     */
    private contextLoader: () => Promise<SectionClassifier>,
    private readonly rules: RuleTables,
    private readonly loadMs: number,
  ) {}

  static async create(assets: EngineAssets, createSession: SessionFactory): Promise<PaceEngine> {
    const started = Date.now();
    const manifest = JSON.parse(await assets.readText('manifest.model')) as EngineManifest;
    const rules = JSON.parse(await assets.readText('rules.model')) as RuleTables;
    const docModel = JSON.parse(await assets.readText('doc-type.model')) as DocumentTypeModel;
    const rawWeights = await assets.readBinary('doc-type.bin');
    // A Float32Array view needs a 4-byte-aligned buffer. Node's fs and the app's
    // file reader both normally hand one back, and the copy below only happens on
    // the platform that does not.
    const aligned = rawWeights.byteOffset % 4 === 0 ? rawWeights : rawWeights.slice();
    const docType = new DocumentTypeClassifier(docModel, {
      data: new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4),
      shape: docModel.weights.shape,
    });
    const matcherModel = JSON.parse(await assets.readText('matcher.model')) as MatcherModel;
    const vocab = await assets.readText('tokenizer/vocab.model');
    const { do_lower_case: doLowerCase, strip_accents: stripAccents } = manifest.tokenizer.config;
    const tokenizer = new WordPieceTokenizer(vocab, {
      doLowerCase,
      tokenizeChineseChars: manifest.tokenizer.config.tokenize_chinese_chars ?? true,
      // HuggingFace's BasicTokenizer defaults `strip_accents` to `lowercase` when
      // it is unset, so "Café" is encoded as "cafe". Passing `false` here made
      // every accented word an [UNK] while every fixture without accents passed.
      stripAccents: stripAccents ?? doLowerCase,
    });
    const session = await createSession(await assets.modelUri(manifest.models.section.file));
    const sectionModel = new SectionClassifier(session, tokenizer, {
      labels: manifest.models.section.labels,
      tasks: manifest.models.section.tasks,
      maxLength: manifest.models.section.max_length,
      temperature: manifest.models.section.temperature,
    });
    let contextModel: Promise<SectionClassifier> | null = null;
    const loadContext = () => {
      if (!contextModel) {
        contextModel = (async () => {
          const contextSession = await createSession(await assets.modelUri(manifest.models.context.file));
          return new SectionClassifier(contextSession, tokenizer, {
            labels: manifest.models.context.labels,
            tasks: manifest.models.context.tasks,
            maxLength: manifest.models.context.max_length,
            temperature: manifest.models.context.temperature,
          });
        })().catch((error) => {
          // A failed load stays retryable rather than wedging every later scan.
          contextModel = null;
          throw error;
        });
      }
      return contextModel;
    };
    return new PaceEngine(
      manifest,
      docType,
      new CourseMatcher(matcherModel),
      sectionModel,
      loadContext,
      rules,
      Date.now() - started,
    );
  }

  get sectionModelMeta() {
    return this.manifest.models.section;
  }

  async understand(text: string, options: UnderstandOptions = {}): Promise<UnderstandResult> {
    if (typeof text !== 'string') throw new TypeError('text must be a string');
    if (!text.trim()) throw new Error('text must not be empty');

    const started = Date.now();
    // One newline convention, the one Python's text mode reads, so offsets and
    // ids do not depend on whether the text came from a file, a PDF or OCR.
    const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const cleaned = cleanHtml(normalized);

    const typeStart = Date.now();
    const proposedType = this.docType.classify(cleaned);
    const assumed = options.assumeDocumentType?.trim();
    // The transcript reader runs before the syllabus reader because the two
    // shapes are incompatible: a transcript's column header would otherwise be
    // read as a course title. Its own evidence decides — a table header or a
    // column of dated rows — and a reviewer who says "read this as a syllabus"
    // skips it entirely.
    const transcriptForced = assumed === 'transcript';
    const transcriptReading = assumed && !transcriptForced ? null : readTranscript(cleaned);
    const transcriptMode = transcriptForced || Boolean(transcriptReading?.detected);
    const transcriptDocId = transcriptMode ? documentId(cleaned) : '';
    const transcriptEvidenceRows: Evidence[] = transcriptMode && transcriptReading
      ? transcriptEvidence(transcriptReading, transcriptDocId).courses
      : [];
    const transcriptRecord = transcriptMode && transcriptReading
      ? buildTranscriptRecord(transcriptReading, transcriptDocId, transcriptEvidenceRows)
      : null;
    const documentType = transcriptMode
      ? {
          ...proposedType,
          label: 'transcript',
          proposedLabel: proposedType.label,
          academic: true,
          abstained: false,
        }
      : assumed
        ? {
            ...proposedType,
            label: assumed,
            proposedLabel: assumed,
            academic: assumed !== 'other',
            abstained: false,
          }
        : proposedType;
    const documentTypeMs = Date.now() - typeStart;

    const sectionStart = Date.now();
    const detected: DetectedSection[] = transcriptMode ? [] : detectSections(cleaned, this.rules);
    const detectedMs = Date.now() - sectionStart;

    const modelStart = Date.now();
    let typed: TypedSection[] = [];
    if (!transcriptMode) {
      const predictions = await this.sectionModel.probabilities(detected.map((section) => section.text));
      // `classify_section`: the heading is asked first; the heading+value model only
      // takes over when the heading is unfamiliar (< 0.8), which is ~7% of real
      // sections. Two sections in the shipped fixtures need it ("Meeting Times",
      // "Attendance Policy"), which is why the second model is in the bundle.
      const fallbackIndexes: number[] = [];
      const fallbackInputs: string[] = [];
      detected.forEach((section, index) => {
        const best = bestIndex(predictions[index]);
        const confidence = predictions[index][best];
        if (confidence >= 0.8) return;
        const body = sectionBody(cleaned, detected, section, 300);
        if (!body) return;
        fallbackIndexes.push(index);
        fallbackInputs.push(`${section.text}: ${body.slice(0, 400)}`);
      });
      // No unfamiliar headings means the second graph is never opened at all.
      const contextPredictions = fallbackInputs.length
        ? await (await this.contextLoader()).probabilities(fallbackInputs)
        : [];
      typed = detected.map((section, index) => {
        const fallbackAt = fallbackIndexes.indexOf(index);
        const meta = fallbackAt === -1 ? this.manifest.models.section : this.manifest.models.context;
        const row = fallbackAt === -1 ? predictions[index] : contextPredictions[fallbackAt];
        const best = bestIndex(row);
        const label = meta.labels[best];
        return {
          heading: section.text,
          label,
          task: meta.tasks[label] ?? 'other',
          confidence: row[best],
          start: section.start,
          end: section.end,
        };
      });
    }
    const sectionModelMs = Date.now() - modelStart;

    const extractStart = Date.now();
    let course: Record<string, Value<string | number>> = {};
    let grading: GradingRow[] = [];
    let meetings: ReturnType<typeof extractMeetings> = [];
    let policies: ReturnType<typeof extractPolicies> = {};
    let total: number | null = null;
    if (!transcriptMode) {
      const title = extractCourseTitle(cleaned, detected);
      if (title) course.course_title = title;
      const code = extractCourseCode(cleaned);
      if (code) course.course_code = code;
      const credits = extractCredits(cleaned);
      if (credits) course.credits = credits;
      const instructor = extractInstructor(cleaned);
      if (instructor) course.instructor = instructor;

      grading = extractGrading(cleaned);
      meetings = extractMeetings(cleaned);
      policies = extractPolicies(
        cleaned,
        detected,
        typed.map((section) => section.label),
        this.rules,
      );
      total = grading.length ? grading.reduce((sum, row) => sum + row.weight, 0) : null;
    }
    const extractionMs = Date.now() - extractStart;

    const record = buildRecord({
      text: normalized,
      cleaned,
      docType: documentType,
      sections: typed,
      course,
      grading,
      meetings,
      policies,
      tables: this.rules,
      model: {
        engineVersion: this.manifest.engineVersion,
        checkpoint: this.manifest.models.section.id,
        family: 'encoder-classifier',
        parameters: this.manifest.models.section.parameters,
        maxStateTokens: this.manifest.models.section.max_length,
        sectionSource: 'heading-only',
      },
      gradingTotal: {
        percent: total,
        consistent: total === null ? null : Math.abs(total - 100) <= 5,
      },
      transcript: transcriptRecord ? { record: transcriptRecord, evidence: transcriptEvidenceRows } : null,
    });

    if (assumed) {
      record.warnings.push(
        `document type was set to "${assumed}" by the reviewer; the classifier proposed "${proposedType.label}" (${proposedType.confidence.toFixed(2)})`,
      );
    }

    return {
      record,
      text: normalized,
      timings: {
        documentTypeMs,
        sectionsMs: detectedMs,
        sectionModelMs,
        extractionMs,
        totalMs: Date.now() - started,
        loadMs: this.loadMs,
      },
    };
  }

  /** Propose whether two course names are the same course. Never merges. */
  matchCourses(left: string, right: string) {
    return this.matcher.decide(left, right);
  }

  /**
   * One course against every existing course, in one batch.
   *
   * The matcher's character-similarity feature is fitted across a batch (that is
   * how it was trained), so matching against a list is the honest use of it --
   * and the shape the review screen needs when a scan could belong to any of the
   * courses already imported.
   */
  matchCandidates(left: string, rights: string[]) {
    return this.matcher.decidePairs(rights.map((right) => [left, right] as [string, string]));
  }

  /** Pair-batch entry point, used by the parity harness. */
  matchPairs(pairs: [string, string][]) {
    return this.matcher.decidePairs(pairs);
  }
}
