import {
  Button as NativeButton,
  HStack,
  LabeledContent,
  Menu,
  Section,
  Stepper,
  Text as NativeText,
  TextField,
  Toggle,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import {
  disabled,
  font,
  foregroundStyle,
  labelsHidden,
  lineLimit,
  onLongPressGesture,
} from '@expo/ui/swift-ui/modifiers';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';

import { creditsValue, gpaValue } from '../src/domain/academics';
import type { Proposal } from '../src/engine/types';
import type { GradingValue, MeetingValue, TranscriptRowValue } from '../src/features/import/draft';
import { gradingTotal, importBlocker, reviewCounts } from '../src/features/import/draft';
import { useImportDraft } from '../src/features/import/ImportContext';
import type { SourceLine } from '../src/ui/source';
import { NativeScreen, Value } from '../src/ui/native';
import { SourceSection } from '../src/ui/source';
import {
  creditsLabel,
  daysLabel,
  documentTypeLabel,
  evidenceExcerpt,
  timeRangeLabel,
  weightLabel,
} from '../src/ui/format';
import { useTheme } from '../src/ui/useTheme';

function valueLabel(proposal: { key: string; value: unknown }): string {
  const value = proposal.value;
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return String(value);
  if (value && typeof value === 'object' && 'category' in value) {
    const grading = value as GradingValue;
    return `${grading.category} ${grading.weight}%`;
  }
  if (value && typeof value === 'object' && 'days' in value) {
    const meeting = value as MeetingValue;
    return [daysLabel(meeting.days), timeRangeLabel(meeting.startTime, meeting.endTime), meeting.room]
      .filter(Boolean)
      .join(' · ');
  }
  if (value && typeof value === 'object' && 'mark' in value) {
    const row = value as TranscriptRowValue;
    return [row.title, row.mark, row.credit != null ? `${row.credit} credits` : null]
      .filter(Boolean)
      .join(' · ');
  }
  return '';
}

const TYPE_OPTIONS = [
  { label: 'Syllabus', value: 'syllabus' },
  { label: 'Transcript', value: 'transcript' },
  { label: 'Schedule', value: 'schedule' },
  { label: 'Course catalog', value: 'course_catalog' },
];

/** Parse the way a person types days: "MWF", "Mon, Wed", "tue thu". */
function parseDays(text: string): string[] {
  const tokens = text.split(/[\s,;/·]+/).filter(Boolean);
  const names: Record<string, string> = {
    m: 'mon', mo: 'mon', mon: 'mon', monday: 'mon',
    t: 'tue', tu: 'tue', tue: 'tue', tues: 'tue', tuesday: 'tue',
    w: 'wed', we: 'wed', wed: 'wed', wednesday: 'wed',
    th: 'thu', thu: 'thu', thur: 'thu', thurs: 'thu', thursday: 'thu',
    f: 'fri', fr: 'fri', fri: 'fri', friday: 'fri',
    sa: 'sat', sat: 'sat', saturday: 'sat',
    su: 'sun', sun: 'sun', sunday: 'sun',
  };
  // "MWF" arrives as one token; expanding it only when every letter is a day
  // keeps "Mon" from being read as M, O, N.
  const expanded = tokens.flatMap((token) => {
    const lower = token.toLowerCase();
    if (lower.length > 1 && !names[lower] && [...lower].every((letter) => names[letter])) {
      return [...lower];
    }
    return [token];
  });
  const out: string[] = [];
  for (const token of expanded) {
    const day = names[token.toLowerCase()];
    if (day && !out.includes(day)) out.push(day);
  }
  return out;
}

/**
 * A text field that holds its own text, seeded once when its screen mounts.
 *
 * `@expo/ui`'s TextField keeps its own string, so a value read from the database
 * has to be pushed into it. Seeding on every keystroke would fight the person
 * typing, so the parent keys this by the draft id: a review of a new document is
 * a new field, and the same document keeps what was typed.
 */
function SeededField({
  label,
  initial,
  onChange,
}: {
  label: string;
  initial: string;
  onChange: (text: string) => void;
}) {
  const ref = useRef<TextFieldRef>(null);
  useEffect(() => {
    void ref.current?.setText(initial);
    // Mount only: the initial value is the value the document printed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return <TextField ref={ref} placeholder={label} onTextChange={onChange} />;
}

export default function Review() {
  const router = useRouter();
  const { palette } = useTheme();
  const {
    draft,
    status,
    error,
    engine,
    setIncluded,
    updateGrading,
    updateMeeting,
    updateCourseText,
    updateCredits,
    setLink,
    commit,
    reset,
    reread,
  } = useImportDraft();

  const [source, setSource] = useState<SourceLine | null>(null);
  const [busy, setBusy] = useState(false);
  // State is not enough here: a second tap can arrive before the re-render that
  // disables the button, and that would start a second commit.
  const importing = useRef(false);

  const openSource = (proposal: Proposal<unknown>, label: string) => {
    if (!draft) return;
    const meta = proposal.evidenceId ? draft.evidenceMeta[proposal.evidenceId] : undefined;
    setSource({
      label,
      quote: evidenceExcerpt(draft.text, proposal.start, proposal.end) || proposal.evidenceText || '',
      page: meta?.page ?? null,
      bbox: meta?.bbox ?? null,
      confidence: proposal.confidence,
      confidenceSource: proposal.source,
      method:
        proposal.source === 'rule'
          ? 'Read from the page by the deterministic reader'
          : 'Classified on this device by the section model',
      image: draft.image,
    });
  };

  if (!draft) {
    return (
      <NativeScreen title="Nothing to review" onBack={() => router.replace('/')}>
        <Section>
          <NativeText modifiers={[foregroundStyle(palette.inkMuted)]}>
            This review was closed. Scan a document to start another.
          </NativeText>
          <NativeButton
            label="Scan a document"
            systemImage="doc.viewfinder"
            onPress={() => router.replace('/scan')}
          />
        </Section>
      </NativeScreen>
    );
  }

  const counts = reviewCounts(draft);
  const total = gradingTotal(draft);
  const transcriptRows = draft.transcript?.rows ?? [];
  // Term banners group the rows the way the page printed them; a row printed
  // under no banner is kept, not dropped, and says so.
  const transcriptGroups: { label: string; rows: Proposal<TranscriptRowValue>[]; credits: number }[] = [];
  if (draft.transcript) {
    const known = draft.transcript.terms.map((term) => term.value);
    const byTerm = new Map<string, Proposal<TranscriptRowValue>[]>();
    for (const row of transcriptRows) {
      const key = row.value.inProgress
        ? 'In progress'
        : row.value.term && known.includes(row.value.term)
          ? row.value.term
          : 'No term printed';
      byTerm.set(key, [...(byTerm.get(key) ?? []), row]);
    }
    for (const [label, rows] of byTerm) {
      const credits = rows.reduce((sum, row) => sum + (row.value.credit ?? 0), 0);
      transcriptGroups.push({ label, rows, credits: Math.round(credits * 1000) / 1000 });
    }
  }
  // Review by exception: a transcript page is mostly right, so only the rows Pace
  // is unsure about are marked. An in-progress row has no mark by design and is
  // not one of them.
  const needsReviewRows = new Set(
    transcriptRows
      .filter(
        (row) =>
          row.confidence < 0.9 ||
          row.value.credit == null ||
          (row.value.mark == null && !row.value.inProgress),
      )
      .map((row) => row.key),
  );
  const includedCount = draft.transcript
    ? transcriptRows.filter((row) => row.included).length
    : (draft.title?.included ? 1 : 0) +
      (draft.code?.included ? 1 : 0) +
      (draft.instructor?.included ? 1 : 0) +
      (draft.credits?.included ? 1 : 0) +
      draft.grading.filter((row) => row.included).length +
      draft.meetings.filter((row) => row.included).length +
      draft.policies.filter((policy) => policy.included).length;
  const blocker = importBlocker(draft);
  const typeLabel = documentTypeLabel(draft.documentType?.label);
  const committing = busy || status === 'committing';

  const leave = () => {
    if (committing || importing.current) return;
    reset();
    router.replace('/');
  };

  const importNow = async () => {
    if (blocker || importing.current) return;
    importing.current = true;
    setBusy(true);
    const isTranscript = Boolean(draft.transcript);
    try {
      const courseId = await commit();
      // One haptic per import, at the moment it lands, paired with the navigation
      // that shows the course. A failure gets the error pattern instead. A
      // transcript has no single course to open, so it lands on the list.
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      reset();
      router.replace(isTranscript ? '/imported' : `/course/${courseId}`);
    } catch {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
      importing.current = false;
      setBusy(false);
    }
  };

  const note = [font({ textStyle: 'footnote' }), foregroundStyle(palette.inkMuted)];
  const warnNote = [font({ textStyle: 'footnote' }), foregroundStyle(palette.warn)];

  return (
    <NativeScreen
      title="Review"
      subtitle={`${typeLabel} · ${counts.total} ${
        draft.transcript ? `course${counts.total === 1 ? '' : 's'}` : `item${counts.total === 1 ? '' : 's'}`
      } found${counts.needsReview ? `, ${counts.needsReview} to check` : ''}`}
      onBack={leave}
    >
      {source ? <SourceSection source={source} onClose={() => setSource(null)} /> : null}

      <Section
        title="Document"
        footer={
          <NativeText>{`${draft.academic ? 'Academic document' : 'Not academic'}${
            draft.pageCount > 1 ? ` · ${draft.pageCount} pages` : ''
          }${needsReviewRows.size && draft.transcript ? ` · ${needsReviewRows.size} rows to check` : ''}`}</NativeText>
        }
      >
        <Menu label={`Read as · ${typeLabel}`} systemImage="doc.text.magnifyingglass">
          {TYPE_OPTIONS.map((option) => (
            <NativeButton
              key={option.value}
              label={option.label}
              systemImage={draft.documentType?.label === option.label ? 'checkmark' : undefined}
              onPress={() => {
                if (!engine || committing) return;
                void reread(option.value);
              }}
            />
          ))}
        </Menu>
        {draft.documentType && draft.documentType.confidence < 0.6 ? (
          <LabeledContent label="Type">
            <NativeText modifiers={warnNote}>uncertain</NativeText>
          </LabeledContent>
        ) : null}
      </Section>

      {!draft.academic ? (
        <Section
          title="Pace couldn't find academic information"
          footer={
            <NativeText>
              {`It read this as “${typeLabel}”. Nothing was extracted and nothing will be imported. If you know this is academic, pick what it is above: the extractor runs again on the same text.`}
            </NativeText>
          }
        >
          <NativeText modifiers={warnNote}>
            The record will say a person made that call.
          </NativeText>
        </Section>
      ) : null}

      {draft.duplicate ? (
        <Section
          title="This page is already imported"
          footer={
            <NativeText>
              Pace read the same text again. Importing it twice would create a second copy of the course, so
              nothing will be saved from this review.
            </NativeText>
          }
        >
          <NativeText modifiers={[foregroundStyle(palette.warn)]}>Nothing here can be imported.</NativeText>
        </Section>
      ) : null}

      {draft.match ? (
        <Section
          title="Possible existing course"
          footer={
            <NativeText>
              {`${Math.round(draft.match.probability * 100)}% likely the same course. Pace never merges on its own.`}
            </NativeText>
          }
        >
          <LabeledContent label={draft.match.title}>
            <Value>{draft.match.code ?? ''}</Value>
          </LabeledContent>
          <NativeButton
            label={draft.linkToCourseId ? 'Linked' : 'Link to it'}
            systemImage={draft.linkToCourseId ? 'link.badge.plus' : 'link'}
            onPress={() => setLink(draft.linkToCourseId ? null : (draft.match?.courseId ?? null))}
          />
          <NativeButton label="Keep separate" systemImage="square.split.2x1" onPress={() => setLink(null)} />
        </Section>
      ) : null}

      {draft.warnings.filter((warning) => !warning.startsWith('this does not look like an academic document')).length ? (
        <Section title="Warnings">
          {draft.warnings
            .filter((warning) => !warning.startsWith('this does not look like an academic document'))
            .slice(0, 3)
            .map((warning) => (
              <NativeText key={warning} modifiers={warnNote}>
                {warning}
              </NativeText>
            ))}
        </Section>
      ) : null}

      {draft.academic ? (
        draft.transcript ? (
          <>
            <Section
              title="Transcript"
              footer={
                draft.transcript.student || draft.transcript.school ? (
                  <NativeText>
                    {[draft.transcript.student, draft.transcript.school].filter(Boolean).join(' · ')}
                  </NativeText>
                ) : undefined
              }
            >
              <LabeledContent label="Courses">
                <Value>
                  {`${includedCount} of ${transcriptRows.length} kept`}
                </Value>
              </LabeledContent>
              <LabeledContent label="Earned">
                <Value>{creditsValue(draft.transcript.totalCredits)}</Value>
              </LabeledContent>
              <LabeledContent label="In progress">
                <Value>{creditsValue(draft.transcript.inProgressCredits)}</Value>
              </LabeledContent>
              {draft.transcript.reportedGpa != null ? (
                <LabeledContent label="Weighted GPA">
                  <Value>{gpaValue(draft.transcript.reportedGpa)}</Value>
                </LabeledContent>
              ) : null}
              {needsReviewRows.size ? (
                <LabeledContent label="Needs review">
                  <NativeText modifiers={warnNote}>{String(needsReviewRows.size)}</NativeText>
                </LabeledContent>
              ) : null}
            </Section>

            {transcriptGroups.map((group) => (
              <Section
                key={group.label}
                title={group.label}
                footer={
                  <NativeText>
                    {`${group.rows.length} course${group.rows.length === 1 ? '' : 's'} · ${group.credits} credits · hold a row to see its source`}
                  </NativeText>
                }
              >
                {group.rows.map((row) => (
                  <Toggle
                    key={row.key}
                    isOn={row.included}
                    onIsOnChange={(isOn) => setIncluded(row.key, isOn)}
                    modifiers={[onLongPressGesture(() => openSource(row, valueLabel(row)))]}
                  >
                    <NativeText
                      modifiers={[
                        foregroundStyle(row.included ? palette.ink : palette.inkFaint),
                      ]}
                    >
                      {`${row.value.title}${row.value.mark ? ` · ${row.value.mark}` : ''}`}
                    </NativeText>
                    <NativeText modifiers={note}>
                      {[
                        row.value.code,
                        row.value.credit != null
                          ? `${row.value.credit} credit${row.value.credit === 1 ? '' : 's'}`
                          : null,
                        needsReviewRows.has(row.key) ? 'needs review' : null,
                      ]
                        .filter(Boolean)
                        .join(' · ') || '—'}
                    </NativeText>
                  </Toggle>
                ))}
              </Section>
            ))}

            {draft.transcript.assessments.length ? (
              <Section title="State assessments">
                {draft.transcript.assessments.map((assessment, index) => (
                  <LabeledContent
                    key={`${assessment.title}-${index}`}
                    label={assessment.title}
                  >
                    <Value>
                      {assessment.label
                        ? `${assessment.score} · level ${assessment.level ?? '—'}`
                        : String(assessment.score)}
                    </Value>
                  </LabeledContent>
                ))}
              </Section>
            ) : null}
          </>
        ) : (
          <>
            <Section
              title="Course"
              footer={
                draft.title ? (
                  <NativeText>Switch a field off to leave it out of the import.</NativeText>
                ) : (
                  <NativeText>Type a title so this can be imported.</NativeText>
                )
              }
            >
              <HStack spacing={8}>
                <Toggle
                  isOn={draft.title?.included ?? true}
                  onIsOnChange={(isOn) => setIncluded('course_title', isOn)}
                  modifiers={[labelsHidden()]}
                />
                <SeededField
                  key={`${draft.id}-title`}
                  label="Title"
                  initial={draft.title?.value ?? ''}
                  onChange={(text) => updateCourseText('title', text)}
                />
              </HStack>
              <HStack spacing={8}>
                <Toggle
                  isOn={draft.code?.included ?? true}
                  onIsOnChange={(isOn) => setIncluded('course_code', isOn)}
                  modifiers={[labelsHidden()]}
                />
                <SeededField
                  key={`${draft.id}-code`}
                  label="Code"
                  initial={draft.code?.value ?? ''}
                  onChange={(text) => updateCourseText('code', text)}
                />
              </HStack>
              <HStack spacing={8}>
                <Toggle
                  isOn={draft.instructor?.included ?? true}
                  onIsOnChange={(isOn) => setIncluded('instructor', isOn)}
                  modifiers={[labelsHidden()]}
                />
                <SeededField
                  key={`${draft.id}-instructor`}
                  label="Instructor"
                  initial={draft.instructor?.value ?? ''}
                  onChange={(text) => updateCourseText('instructor', text)}
                />
              </HStack>
              <HStack spacing={8}>
                <Toggle
                  isOn={draft.credits?.included ?? true}
                  onIsOnChange={(isOn) => setIncluded('credits', isOn)}
                  modifiers={[labelsHidden()]}
                />
                <SeededField
                  key={`${draft.id}-credits`}
                  label="Credits"
                  initial={draft.credits != null ? String(draft.credits.value) : ''}
                  onChange={(text) => updateCredits(text.trim() ? Number(text) : null)}
                />
              </HStack>
            </Section>

            <Section
              title="Grading"
              footer={
                <NativeText>
                  {total != null && Math.abs(total - 100) > 5
                    ? 'The weights do not add up to 100%: Pace may have missed a category, or this syllabus uses nested weights. Nothing was normalised.'
                    : 'Weights are the document’s own; nothing is normalised.'}
                </NativeText>
              }
            >
              {draft.grading.length === 0 ? (
                <NativeText modifiers={note}>No grading breakdown was found in this document.</NativeText>
              ) : null}
              {draft.grading.map((row) => (
                <HStack key={row.key} spacing={8}>
                  <Toggle
                    isOn={row.included}
                    onIsOnChange={(isOn) => setIncluded(row.key, isOn)}
                    modifiers={[labelsHidden()]}
                  />
                  <NativeText
                    modifiers={[
                      foregroundStyle(row.included ? palette.ink : palette.inkFaint),
                      lineLimit(1),
                    ]}
                  >
                    {`${row.value.category} · ${weightLabel(row.value.weight)}`}
                  </NativeText>
                  <Stepper
                    label={row.value.category}
                    value={row.value.weight}
                    step={1}
                    min={0}
                    max={100}
                    modifiers={[labelsHidden()]}
                    onValueChange={(weight) =>
                      updateGrading(row.key, { category: row.value.category, weight })
                    }
                  />
                </HStack>
              ))}
              {draft.grading.length ? (
                <LabeledContent label="Total">
                  <NativeText
                    modifiers={[
                      font({ textStyle: 'body', weight: 'semibold' }),
                      foregroundStyle(
                        total != null && Math.abs(total - 100) <= 5 ? palette.high : palette.warn,
                      ),
                    ]}
                  >
                    {total == null ? '—' : weightLabel(total)}
                  </NativeText>
                </LabeledContent>
              ) : null}
            </Section>

            <Section title="Meetings">
              {draft.meetings.length === 0 ? (
                <NativeText modifiers={note}>No meeting time was found in this document.</NativeText>
              ) : null}
              {draft.meetings.map((meeting) => (
                <LabeledContent
                  key={meeting.key}
                  label={daysLabel(meeting.value.days) || 'Meeting'}
                >
                  <Value>
                    {[timeRangeLabel(meeting.value.startTime, meeting.value.endTime), meeting.value.room]
                      .filter(Boolean)
                      .join(' · ') || 'No time read'}
                  </Value>
                </LabeledContent>
              ))}
            </Section>

            {draft.meetings.map((meeting, index) => (
              <Section
                key={`edit-${meeting.key}`}
                title={`Meeting ${index + 1}`}
                footer={<NativeText>{meeting.evidenceText ?? undefined}</NativeText>}
              >
                <Toggle
                  label="Include this meeting"
                  isOn={meeting.included}
                  onIsOnChange={(isOn) => setIncluded(meeting.key, isOn)}
                />
                <HStack spacing={8}>
                  <NativeText modifiers={note}>Days</NativeText>
                  <SeededField
                    key={`${draft.id}-${meeting.key}-days`}
                    label="Mon, Wed"
                    initial={daysLabel(meeting.value.days)}
                    onChange={(text) =>
                      updateMeeting(meeting.key, { ...meeting.value, days: parseDays(text) })
                    }
                  />
                </HStack>
                <HStack spacing={8}>
                  <NativeText modifiers={note}>Starts</NativeText>
                  <SeededField
                    key={`${draft.id}-${meeting.key}-start`}
                    label="09:00"
                    initial={meeting.value.startTime ?? ''}
                    onChange={(text) =>
                      updateMeeting(meeting.key, {
                        ...meeting.value,
                        startTime: text.trim() || null,
                      })
                    }
                  />
                </HStack>
                <HStack spacing={8}>
                  <NativeText modifiers={note}>Ends</NativeText>
                  <SeededField
                    key={`${draft.id}-${meeting.key}-end`}
                    label="10:50"
                    initial={meeting.value.endTime ?? ''}
                    onChange={(text) =>
                      updateMeeting(meeting.key, { ...meeting.value, endTime: text.trim() || null })
                    }
                  />
                </HStack>
                <HStack spacing={8}>
                  <NativeText modifiers={note}>Room</NativeText>
                  <SeededField
                    key={`${draft.id}-${meeting.key}-room`}
                    label="Optional"
                    initial={meeting.value.room ?? ''}
                    onChange={(text) =>
                      updateMeeting(meeting.key, { ...meeting.value, room: text.trim() || null })
                    }
                  />
                </HStack>
              </Section>
            ))}

            {draft.policies.length ? (
              <Section title="Policies" footer={<NativeText>Hold a policy to see the line it came from.</NativeText>}>
                {draft.policies.map((policy) => (
                  <Toggle
                    key={policy.key}
                    isOn={policy.included}
                    onIsOnChange={(isOn) => setIncluded(policy.key, isOn)}
                    modifiers={[onLongPressGesture(() => openSource(policy, valueLabel(policy)))]}
                  >
                    <NativeText modifiers={[foregroundStyle(policy.included ? palette.ink : palette.inkFaint)]}>
                      {policy.key.replace('policy:', '').replace(/_/g, ' ')}
                    </NativeText>
                    <NativeText modifiers={note}>{policy.value || 'Not stated'}</NativeText>
                  </Toggle>
                ))}
              </Section>
            ) : null}
          </>
        )
      ) : null}

      {error ? (
        <Section title="That did not finish" footer={<NativeText>{error}</NativeText>}>
          <NativeText modifiers={[foregroundStyle(palette.danger)]}>
            Nothing was saved. You can try again.
          </NativeText>
        </Section>
      ) : null}

      <Section
        footer={
          <NativeText>
            {blocker ??
              (draft.transcript
                ? 'Nothing is saved until you import.'
                : 'Nothing is saved until you import. A value you edit keeps the printed text beside it.')}
          </NativeText>
        }
      >
        <NativeButton
          label={
            blocker
              ? 'Nothing to import'
              : committing
                ? 'Importing…'
                : `Import ${includedCount} item${includedCount === 1 ? '' : 's'}`
          }
          systemImage="tray.and.arrow.down"
          modifiers={[disabled(Boolean(blocker) || committing)]}
          onPress={() => void importNow()}
        />
      </Section>
    </NativeScreen>
  );
}
