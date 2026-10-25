/**
 * The in-memory import draft, shared by the scan, review and course screens.
 *
 * The draft is deliberately *not* persisted: a scan that is abandoned should not
 * leave a half-reviewed proposal in the database. The document and its evidence
 * only become rows when the user taps Import, in one transaction.
 */
import { useSQLiteContext, type SQLiteDatabase } from 'expo-sqlite';
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';

import type { PaceEngine, UnderstandResult } from '../../engine/PaceEngine';
import type { Proposal } from '../../engine/types';
import { commitImport, committedImportForDocument, matchCandidates } from '../../database/repo';
import { draftToCommitInput } from './commit';
import {
  buildDraft,
  type DraftSource,
  type EvidenceMeta,
  type GradingValue,
  type ImportDraft,
  type MatchProposal,
  type MeetingValue,
} from './draft';

export type ImportStatus = 'idle' | 'review' | 'committing' | 'committed' | 'failed';

interface ImportContextValue {
  draft: ImportDraft | null;
  status: ImportStatus;
  error: string | null;
  committedCourseId: string | null;
  engine: PaceEngine | null;
  begin: (args: {
    engine: PaceEngine;
    result: UnderstandResult;
    source: DraftSource;
    evidenceMeta: Record<string, EvidenceMeta>;
    image?: { uri: string; width: number; height: number } | null;
    pageCount?: number;
  }) => Promise<ImportDraft>;
  setIncluded: (key: string, included: boolean) => void;
  updateGrading: (key: string, value: GradingValue) => void;
  updateMeeting: (key: string, value: MeetingValue) => void;
  updateCourseText: (field: 'title' | 'code' | 'instructor', value: string) => void;
  updateCredits: (value: number | null) => void;
  setLink: (courseId: string | null) => void;
  commit: () => Promise<string>;
  reset: () => void;
  /** Re-read the stored text as a type the reviewer chose. */
  reread: (assumeDocumentType: string) => Promise<void>;
}

const ImportContext = createContext<ImportContextValue | null>(null);

function patchValue<T>(list: Proposal<T>[], key: string, value: T): Proposal<T>[] {
  return list.map((row) => (row.key === key ? { ...row, value, edited: true } : row));
}

type BeginInput = Parameters<ImportContextValue['begin']>[0];

/**
 * The work behind `begin`: duplicate detection, matching, then the draft.
 *
 * Kept out of the provider so a failure cannot leave React state half-updated —
 * a rejected duplicate query used to leave `review` set against a stale draft.
 */
async function prepareDraft(db: SQLiteDatabase, input: BeginInput): Promise<ImportDraft> {
  const { engine, result, source, evidenceMeta, image, pageCount } = input;
  const existing = await committedImportForDocument(db, result.record.document.id);
  const duplicate = existing ? { importId: existing.id, courseId: existing.course_id } : null;

  // Cross-document matching runs before review so a syllabus for a course already
  // imported can offer "link" instead of creating a duplicate. It is always a
  // proposal: nothing merges without the user's explicit tap.
  let match: MatchProposal | null = null;
  const left = [result.record.course.course_title, result.record.course.course_code]
    .filter(Boolean)
    .join(' ');
  if (left.trim()) {
    try {
      const candidates = await matchCandidates(db);
      const rights = candidates.map((candidate) => candidate.title ?? candidate.code ?? '');
      const usable = candidates
        .map((candidate, index) => ({ candidate, right: rights[index] }))
        .filter((entry) => entry.right.trim().length > 0);
      if (usable.length) {
        const results = engine.matchCandidates(left, usable.map((entry) => entry.right));
        let bestIndex = -1;
        for (let index = 0; index < results.length; index += 1) {
          if (bestIndex === -1 || results[index].probability > results[bestIndex].probability) {
            bestIndex = index;
          }
        }
        const best = bestIndex === -1 ? null : results[bestIndex];
        // The matcher was measured at its exported threshold (0.95 in the frozen
        // bundle). Values in the middle band are deliberately `uncertain`; showing
        // them as "likely the same" throws away that precision guarantee and can
        // encourage false links such as CHEM 1A ↔ CHEM 1L. Only the model's
        // explicit `same` decision is strong enough to reach the review UI.
        if (best?.decision === 'same') {
          const candidate = usable[bestIndex].candidate;
          match = {
            courseId: candidate.id,
            title: candidate.title ?? candidate.code ?? 'Existing course',
            code: candidate.code,
            probability: best.probability,
            decision: best.decision,
          };
        }
      }
    } catch {
      // Matching is an offer, not a requirement: a failure here must not stop the
      // document from being reviewed and imported.
      match = null;
    }
  }

  return buildDraft(result.record, {
    source,
    text: result.text,
    timings: result.timings as unknown as Record<string, number>,
    evidenceMeta,
    image: image ?? null,
    pageCount,
    match,
    duplicate,
  });
}

export function ImportProvider({ children }: { children: ReactNode }) {
  const db = useSQLiteContext();
  const [draft, setDraft] = useState<ImportDraft | null>(null);
  const [engine, setEngine] = useState<PaceEngine | null>(null);
  const [status, setStatus] = useState<ImportStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [committedCourseId, setCommittedCourseId] = useState<string | null>(null);
  // A ref, not the `status` state: two taps in the same frame both read the old
  // state, and a second transaction would collide on the course id.
  const commitInFlight = useRef(false);
  // A transcript import commits many courses and returns no course id, so the
  // idempotency check cannot be the id itself; the marker keeps a second tap
  // from writing the same rows twice.
  const committedRef = useRef<{ marker: string; courseId: string } | null>(null);

  const begin = useCallback<ImportContextValue['begin']>(
    async (input) => {
      setError(null);
      setCommittedCourseId(null);
      committedRef.current = null;
      try {
        const next = await prepareDraft(db, input);
        // Status moves only once the draft exists: a rejection from the duplicate
        // lookup used to leave `review` set with the *previous* document still in
        // hand, so a later visit to /review showed the wrong page.
        setEngine(input.engine);
        setDraft(next);
        setStatus('review');
        return next;
      } catch (cause) {
        setEngine(null);
        setDraft(null);
        setStatus('idle');
        setError(cause instanceof Error ? cause.message : 'The document could not be prepared.');
        throw cause;
      }
    },
    [db],
  );

  const setIncluded = useCallback((key: string, included: boolean) => {
    setDraft((current) => {
      if (!current) return current;
      const hit = (row: { key: string }) => row.key === key;
      const upd = <T,>(row: { key: string; included: boolean } & T) =>
        hit(row) ? { ...row, included } : row;
      return {
        ...current,
        title: current.title && hit(current.title) ? { ...current.title, included } : current.title,
        code: current.code && hit(current.code) ? { ...current.code, included } : current.code,
        instructor:
          current.instructor && hit(current.instructor)
            ? { ...current.instructor, included }
            : current.instructor,
        credits:
          current.credits && hit(current.credits) ? { ...current.credits, included } : current.credits,
        transcript: current.transcript
          ? { ...current.transcript, rows: current.transcript.rows.map(upd) }
          : null,
        grading: current.grading.map(upd),
        meetings: current.meetings.map(upd),
        policies: current.policies.map(upd),
      };
    });
  }, []);

  const updateGrading = useCallback((key: string, value: GradingValue) => {
    setDraft((current) =>
      current ? { ...current, grading: patchValue<GradingValue>(current.grading, key, value) } : current,
    );
  }, []);

  const updateMeeting = useCallback((key: string, value: MeetingValue) => {
    setDraft((current) =>
      current ? { ...current, meetings: patchValue<MeetingValue>(current.meetings, key, value) } : current,
    );
  }, []);

  const updateCourseText = useCallback(
    (field: 'title' | 'code' | 'instructor', value: string) => {
      setDraft((current) => {
        if (!current) return current;
        const target = current[field];
        const proposal = target
          ? { ...target, value, edited: true }
          : null;
        // A field the reader missed can still be typed in: the reviewer knows the
        // course name even when the photograph did not show it.
        const created = proposal ?? {
          key: field === 'title' ? 'course_title' : field === 'code' ? 'course_code' : 'instructor',
          entityId: null,
          evidenceId: null,
          value,
          confidence: 0.5,
          band: 'low' as const,
          source: 'rule' as const,
          evidenceText: null,
          start: null,
          end: null,
          included: true,
          edited: true,
          originalValue: value,
        };
        return { ...current, [field]: created };
      });
    },
    [],
  );

  const updateCredits = useCallback((value: number | null) => {
    setDraft((current) => {
      if (!current) return current;
      if (value === null) {
        return { ...current, credits: current.credits ? { ...current.credits, included: false } : null };
      }
      const base = current.credits;
      return {
        ...current,
        credits: base
          ? { ...base, value, edited: true, included: true }
          : {
              key: 'credits',
              entityId: null,
              evidenceId: null,
              value,
              confidence: 0.5,
              band: 'low',
              source: 'rule',
              evidenceText: null,
              start: null,
              end: null,
              included: true,
              edited: true,
              originalValue: value,
            },
      };
    });
  }, []);

  const setLink = useCallback((courseId: string | null) => {
    setDraft((current) => (current ? { ...current, linkToCourseId: courseId } : current));
  }, []);

  const commit = useCallback(async () => {
    if (!draft) throw new Error('there is no draft to import');
    // Idempotent: a double tap, a retried promise or a re-render must never write
    // the same draft twice. The second caller gets the first import's course.
    if (committedRef.current) return committedRef.current.courseId;
    if (commitInFlight.current) throw new Error('That import is still being saved.');
    commitInFlight.current = true;
    setStatus('committing');
    setError(null);
    try {
      const input = draftToCommitInput(draft);
      await commitImport(db, input);
      // A transcript writes one course per row and has no single course to open.
      const courseId = input.transcriptCourses?.length
        ? ''
        : input.linkToCourseId ?? input.courseId;
      committedRef.current = { marker: courseId || 'transcript', courseId };
      setStatus('committed');
      setCommittedCourseId(courseId);
      return courseId;
    } catch (cause) {
      setStatus('failed');
      setError(cause instanceof Error ? cause.message : 'the import could not be saved');
      throw cause;
    } finally {
      commitInFlight.current = false;
    }
  }, [db, draft]);

  const reset = useCallback(() => {
    setDraft(null);
    setStatus('idle');
    setError(null);
    setCommittedCourseId(null);
    committedRef.current = null;
  }, []);

  /** Re-read the same text as a different document type (reviewer override). */
  const rereadSeq = useRef(0);
  const reread = useCallback(
    async (assumeDocumentType: string) => {
      if (!draft || !engine) return;
      // Two quick picks race: the request that finishes last would win even if it
      // started first, leaving the picker showing a type the draft is not.
      const seq = (rereadSeq.current += 1);
      setError(null);
      try {
        const result = await engine.understand(draft.text, { assumeDocumentType });
        if (seq !== rereadSeq.current) return;
        const next = await begin({
          engine,
          result,
          source: draft.source,
          evidenceMeta: draft.evidenceMeta,
          image: draft.image,
          pageCount: draft.pageCount,
        });
        if (seq !== rereadSeq.current) return;
        setDraft(next);
      } catch (cause) {
        if (seq !== rereadSeq.current) return;
        setStatus('failed');
        setError(cause instanceof Error ? cause.message : 'That reading could not be produced.');
      }
    },
    [begin, draft, engine],
  );

  const value = useMemo<ImportContextValue>(
    () => ({
      draft,
      status,
      error,
      committedCourseId,
      engine,
      begin,
      setIncluded,
      updateGrading,
      updateMeeting,
      updateCourseText,
      updateCredits,
      setLink,
      commit,
      reset,
      reread,
    }),
    [
      draft,
      status,
      error,
      committedCourseId,
      engine,
      begin,
      setIncluded,
      updateGrading,
      updateMeeting,
      updateCourseText,
      updateCredits,
      setLink,
      commit,
      reset,
      reread,
    ],
  );

  return <ImportContext.Provider value={value}>{children}</ImportContext.Provider>;
}

export function useImportDraft(): ImportContextValue {
  const value = useContext(ImportContext);
  if (!value) throw new Error('useImportDraft must be used inside ImportProvider');
  return value;
}