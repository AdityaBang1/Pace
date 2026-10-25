import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, useReducedMotion } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import type { Proposal } from '../src/engine/types';
import type { GradingValue, ImportDraft, MeetingValue } from '../src/features/import/draft';
import { gradingTotal, importBlocker, reviewCounts } from '../src/features/import/draft';
import { useImportDraft } from '../src/features/import/ImportContext';
import { EditSheet, type EditTarget } from '../src/features/review/EditSheet';
import { PressableScale } from '../src/ui/press';
import { EvidenceSheet, type SourceLine } from '../src/features/review/EvidenceSheet';
import {
  Button,
  Card,
  Chip,
  ConfidenceMark,
  Divider,
  FieldRow,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../src/ui/components';
import { NativeDocumentTypePicker } from '../src/ui/NativeDocumentTypePicker';
import { creditsLabel, daysLabel, documentTypeLabel, evidenceExcerpt, timeRangeLabel, weightLabel } from '../src/ui/format';
import { EASE_OUT } from '../src/ui/motion';
import { hairline, radius, space, type } from '../src/ui/theme';
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
  return '';
}

const TYPE_OPTIONS = [
  { label: 'Syllabus', value: 'syllabus' },
  { label: 'Schedule', value: 'schedule' },
  { label: 'Course catalog', value: 'course_catalog' },
];

export default function Review() {
  const router = useRouter();
  const { palette } = useTheme();
  const insets = useSafeAreaInsets();
  const reduceMotion = useReducedMotion();
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

  const [target, setTarget] = useState<EditTarget | null>(null);
  const [source, setSource] = useState<SourceLine | null>(null);
  const [busy, setBusy] = useState(false);
  // State is not enough here: a second tap can arrive before the re-render that
  // disables the button, and that would start a second commit.
  const importing = useRef(false);
  const [editKey, setEditKey] = useState<string | null>(null);

  const openSource = useCallback(
    (proposal: Proposal<unknown>, label: string) => {
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
    },
    [draft],
  );

  if (!draft) {
    return (
      <Screen>
        <PageHeader title="Nothing to review" onBack={() => router.replace('/')} />
        <View style={{ padding: space.xl }}>
          <Text style={[type.body, { color: palette.inkMuted }]}>
            This review was closed. Scan a document to start another.
          </Text>
          <View style={{ marginTop: space.lg }}>
            <Button title="Scan a document" onPress={() => router.replace('/scan')} />
          </View>
        </View>
      </Screen>
    );
  }

  const counts = reviewCounts(draft);
  const total = gradingTotal(draft);
  const includedCount =
    (draft.title?.included ? 1 : 0) +
    (draft.code?.included ? 1 : 0) +
    (draft.instructor?.included ? 1 : 0) +
    (draft.credits?.included ? 1 : 0) +
    draft.grading.filter((row) => row.included).length +
    draft.meetings.filter((row) => row.included).length +
    draft.policies.filter((row) => row.included).length;
  const blocker = importBlocker(draft);
  const editingProposal = editKey
    ? ([
        draft.title,
        draft.code,
        draft.instructor,
        draft.credits,
        ...draft.grading,
        ...draft.meetings,
        ...draft.policies,
      ].find((item) => item?.key === editKey) ?? null)
    : null;
  const typeLabel = documentTypeLabel(draft.documentType?.label);
  const entering = reduceMotion ? undefined : FadeIn.duration(200).easing(EASE_OUT);

  const saveEdit = (next: EditTarget) => {
    if (next.kind === 'text') {
      const field = next.key === 'course_title' ? 'title' : next.key === 'course_code' ? 'code' : 'instructor';
      updateCourseText(field, next.value);
    } else if (next.kind === 'number') {
      updateCredits(Number(next.value));
    } else if (next.kind === 'weight') {
      updateGrading(next.key, { category: next.category, weight: Number(next.weight) });
    } else {
      updateMeeting(next.key, {
        days: next.days,
        startTime: next.startTime || null,
        endTime: next.endTime || null,
        room: next.room || null,
      });
    }
    setTarget(null);
  };

  // Leaving mid-commit would strand a saved course behind a screen that says the
  // review was closed, so the header controls are inert until the write settles.
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
    try {
      const courseId = await commit();
      // One haptic per import, at the moment it lands, paired with the navigation
      // that shows the course. A failure gets the error pattern instead.
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      reset();
      router.replace(`/course/${courseId}`);
    } catch {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
      importing.current = false;
      setBusy(false);
    }
  };

  const renderCourseRow = (label: string, proposal: Proposal<string> | Proposal<number> | null) => {
    if (!proposal) return null;
    const display =
      proposal.key === 'credits' ? creditsLabel(Number(proposal.value)) ?? String(proposal.value) : String(proposal.value);
    return (
      <FieldRow
        label={label}
        value={
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.sm }}>
            <Text
              style={[
                type.bodyStrong,
                { color: proposal.included ? palette.ink : palette.inkFaint },
                !proposal.included && { textDecorationLine: 'line-through' },
              ]}
              numberOfLines={2}
            >
              {display}
            </Text>
            <ConfidenceMark band={proposal.band} source={proposal.source} compact />
          </View>
        }
        onPress={() => {
          setEditKey(proposal.key);
          setTarget(
            proposal.key === 'credits'
              ? {
                  kind: 'number',
                  key: proposal.key,
                  title: 'Credits',
                  value: String(proposal.value),
                }
              : {
                  kind: 'text',
                  key: proposal.key,
                  title: label,
                  value: String(proposal.value),
                  maxLength: 120,
                },
          );
        }}
      />
    );
  };

  return (
    <View style={{ flex: 1, backgroundColor: palette.background }}>
      <ScrollView contentContainerStyle={{ paddingBottom: space.xxxl }} showsVerticalScrollIndicator={false}>
        <View style={{ paddingTop: insets.top }}>
          <PageHeader
            title="Review"
            subtitle={`${typeLabel} · ${counts.total} item${counts.total === 1 ? '' : 's'} found${
              counts.needsReview ? `, ${counts.needsReview} to check` : ''
            }`}
            onBack={leave}
            right={
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ disabled: committing }}
                disabled={committing}
                onPress={leave}
                hitSlop={10}
                style={({ pressed }) => ({
                  opacity: committing ? 0.3 : pressed ? 0.5 : 1,
                  padding: space.sm,
                })}
              >
                <Text style={[type.caption, { color: palette.inkMuted }]}>Close</Text>
              </Pressable>
            }
          />
        </View>

        <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
          <View style={{ flexDirection: 'row', gap: space.sm, flexWrap: 'wrap' }}>
            <Chip label={typeLabel} tone={draft.academic ? 'accent' : 'warn'} />
            {draft.documentType && draft.documentType.confidence < 0.6 ? (
              <Chip label="Type is uncertain" tone="warn" />
            ) : null}
            {draft.source.kind === 'photo' ? <Chip label="Photographed" /> : null}
            {draft.pageCount > 1 ? (
              <Chip label={`${draft.pageCount} pages`} />
            ) : null}
          </View>

          <Card padded={false}>
            <View style={{ paddingHorizontal: space.lg, paddingVertical: space.sm }}>
              <NativeDocumentTypePicker
                label="Read as"
                value={draft.documentType?.label ?? 'syllabus'}
                options={TYPE_OPTIONS}
                onChange={(value) => {
                  if (!engine || committing) return;
                  void reread(value);
                }}
              />
            </View>
          </Card>

          {!draft.academic ? (
            <View style={{ gap: space.md }}>
              <Notice
                tone="warn"
                title="Pace couldn't find academic information in this document."
                body={`It read this as "${typeLabel}". Nothing was extracted and nothing will be imported.`}
              />
              <Text style={[type.caption, { color: palette.inkMuted }]}>
                If you know this is academic, choose what it is above — the extractor runs again on the same text,
                and the record says a person made that call.
              </Text>
            </View>
          ) : null}

          {draft.duplicate ? (
            <Notice
              title="This page is already imported"
              body="Pace read the same text again. Importing it twice would create a second copy of the course, so nothing will be saved from this review."
              tone="warn"
            />
          ) : null}

          {draft.match ? (
            <Card>
              <Text style={[type.label, { color: palette.inkFaint }]}>POSSIBLE EXISTING COURSE</Text>
              <Text style={[type.body, { color: palette.ink, marginTop: space.sm }]}>{draft.match.title}</Text>
              {draft.match.code ? (
                <Text style={[type.caption, { color: palette.inkMuted }]}>{draft.match.code}</Text>
              ) : null}
              <Text style={[type.caption, { color: palette.inkMuted, marginTop: space.sm }]}>
                {Math.round(draft.match.probability * 100)}% likely the same course. Pace never merges on its own.
              </Text>
              <View style={{ flexDirection: 'row', gap: space.sm, marginTop: space.md }}>
                <View style={{ flex: 1 }}>
                  <Button
                    title={draft.linkToCourseId ? 'Linked' : 'Link to it'}
                    kind={draft.linkToCourseId ? 'secondary' : 'primary'}
                    compact
                    onPress={() => setLink(draft.linkToCourseId ? null : draft.match?.courseId ?? null)}
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <Button
                    title="Keep separate"
                    kind="quiet"
                    compact
                    onPress={() => setLink(null)}
                  />
                </View>
              </View>
            </Card>
          ) : null}

          {draft.warnings
            .filter((warning) => !warning.startsWith('this does not look like an academic document'))
            .slice(0, 3)
            .map((warning) => (
              <Notice key={warning} tone="warn" body={warning} />
            ))}
        </View>

        {draft.academic ? (
          <>
            <SectionHeader>COURSE</SectionHeader>
            <View style={{ paddingHorizontal: space.xl }}>
              <Animated.View entering={entering}>
                <Card>
                  {renderCourseRow('Title', draft.title)}
                  {draft.title && (draft.code || draft.instructor || draft.credits) ? <Divider /> : null}
                  {renderCourseRow('Code', draft.code)}
                  {draft.code && (draft.instructor || draft.credits) ? <Divider /> : null}
                  {renderCourseRow('Instructor', draft.instructor)}
                  {draft.instructor && draft.credits ? <Divider /> : null}
                  {renderCourseRow('Credits', draft.credits)}
                  {!draft.title && !draft.code && !draft.instructor && !draft.credits ? (
                    <Text style={[type.caption, { color: palette.inkMuted }]}>
                      No course details were recognised. Add a title so this can be imported.
                    </Text>
                  ) : null}
                </Card>
              </Animated.View>
              {!draft.title ? (
                <View style={{ marginTop: space.sm }}>
                  <Button
                    title="Add a course title"
                    kind="quiet"
                    compact
                    onPress={() =>
                      setTarget({ kind: 'text', key: 'course_title', title: 'Course title', value: '' })
                    }
                  />
                </View>
              ) : null}
            </View>

            <SectionHeader
              right={
                total != null ? (
                  <Text style={[type.caption, { color: palette.inkMuted }]}>Total {weightLabel(total)}</Text>
                ) : null
              }
            >
              GRADING
            </SectionHeader>
            <View style={{ paddingHorizontal: space.xl }}>
              <Card padded={false}>
                {draft.grading.length === 0 ? (
                  <View style={{ padding: space.lg }}>
                    <Text style={[type.caption, { color: palette.inkMuted }]}>
                      No grading breakdown was found in this document.
                    </Text>
                  </View>
                ) : null}
                {draft.grading.map((row, index) => {
                  const mark = row.band === 'low' ? 'low' : row.band === 'medium' ? 'medium' : 'high';
                  const color = mark === 'low' ? palette.low : mark === 'medium' ? palette.medium : palette.high;
                  return (
                    <View key={row.key}>
                      {index > 0 ? <Divider inset={space.lg} /> : null}
                      <PressableScale
                        accessibilityLabel={`${row.value.category}, ${weightLabel(row.value.weight)}. Edit or view source.`}
                        onPress={() => {
                          setEditKey(row.key);
                          setTarget({
                            kind: 'weight',
                            key: row.key,
                            title: row.value.category,
                            category: row.value.category,
                            weight: String(row.value.weight),
                          });
                        }}
                        onLongPress={() => openSource(row, valueLabel(row))}
                        style={{
                          paddingHorizontal: space.lg,
                          paddingVertical: space.md,
                          flexDirection: 'row',
                          alignItems: 'center',
                          gap: space.md,
                        }}
                      >
                        <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: color }} />
                        <Text
                          style={[
                            type.body,
                            { color: row.included ? palette.ink : palette.inkFaint, flex: 1 },
                            !row.included && { textDecorationLine: 'line-through' },
                          ]}
                          numberOfLines={1}
                        >
                          {row.value.category}
                        </Text>
                        <Text
                          style={[
                            type.number,
                            { color: row.included ? palette.ink : palette.inkFaint },
                          ]}
                        >
                          {weightLabel(row.value.weight)}
                        </Text>
                      </PressableScale>
                    </View>
                  );
                })}
                {draft.grading.length ? (
                  <>
                    <Divider inset={space.lg} />
                    <View
                      style={{
                        flexDirection: 'row',
                        paddingHorizontal: space.lg,
                        paddingVertical: space.md,
                        alignItems: 'center',
                      }}
                    >
                      <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]}>Total</Text>
                      <Text style={[type.number, { color: total != null && Math.abs(total - 100) <= 5 ? palette.high : palette.warn }]}>
                        {total == null ? '—' : weightLabel(total)}
                        {total != null && Math.abs(total - 100) <= 5 ? '  ✓' : ''}
                      </Text>
                    </View>
                  </>
                ) : null}
              </Card>

              {total != null && Math.abs(total - 100) > 5 ? (
                <View style={{ marginTop: space.sm }}>
                  <Notice
                    tone="warn"
                    title="Grading total is not 100%"
                    body="Pace may have missed a category, or this syllabus uses nested weights. Nothing was normalised — review before importing."
                  />
                </View>
              ) : null}
            </View>

            <SectionHeader>MEETING</SectionHeader>
            <View style={{ paddingHorizontal: space.xl }}>
              <Card padded={draft.meetings.length === 0}>
                {draft.meetings.length === 0 ? (
                  <Text style={[type.caption, { color: palette.inkMuted }]}>
                    No meeting time was found in this document.
                  </Text>
                ) : null}
                {draft.meetings.map((meeting, index) => {
                  const time = timeRangeLabel(meeting.value.startTime, meeting.value.endTime);
                  return (
                    <View key={meeting.key}>
                      {index > 0 ? <Divider inset={space.lg} /> : null}
                      <PressableScale
                        accessibilityLabel={`${daysLabel(meeting.value.days)} ${time ?? ''}. Edit meeting.`}
                        onPress={() => {
                          setEditKey(meeting.key);
                          setTarget({
                            kind: 'meeting',
                            key: meeting.key,
                            title: 'Meeting',
                            days: meeting.value.days,
                            startTime: meeting.value.startTime ?? '',
                            endTime: meeting.value.endTime ?? '',
                            room: meeting.value.room ?? '',
                          });
                        }}
                        onLongPress={() => openSource(meeting, valueLabel(meeting))}
                        style={{ padding: space.lg, gap: 2 }}
                      >
                        <Text style={[type.bodyStrong, { color: palette.ink }]}>{daysLabel(meeting.value.days)}</Text>
                        <Text style={[type.caption, { color: palette.inkMuted }]}>
                          {[time, meeting.value.room].filter(Boolean).join(' · ') || 'No time read'}
                        </Text>
                      </PressableScale>
                    </View>
                  );
                })}
              </Card>
            </View>

            {draft.policies.length ? (
              <>
                <SectionHeader>POLICIES</SectionHeader>
                <View style={{ paddingHorizontal: space.xl }}>
                  <Card padded={false}>
                    {draft.policies.map((policy, index) => (
                      <View key={policy.key}>
                        {index > 0 ? <Divider inset={space.lg} /> : null}
                        <PressableScale
                          onPress={() => openSource(policy, valueLabel(policy))}
                          style={{ padding: space.lg, gap: 3 }}
                        >
                          <Text style={[type.bodyStrong, { color: palette.ink }]}>
                            {policy.key.replace('policy:', '').replace(/_/g, ' ')}
                          </Text>
                          <Text style={[type.caption, { color: palette.inkMuted }]} numberOfLines={2}>
                            {policy.value}
                          </Text>
                        </PressableScale>
                      </View>
                    ))}
                  </Card>
                </View>
              </>
            ) : null}
          </>
        ) : null}

        {error ? (
          <View style={{ paddingHorizontal: space.xl, marginTop: space.lg }}>
            <Notice tone="danger" title="That did not finish" body={error} />
          </View>
        ) : null}
      </ScrollView>

      <View
        style={{
          borderTopWidth: hairline,
          borderTopColor: palette.line,
          backgroundColor: palette.background,
          paddingHorizontal: space.xl,
          paddingTop: space.md,
          paddingBottom: insets.bottom + space.md,
          gap: space.sm,
        }}
      >
        <Button
          title={
            blocker
              ? 'Nothing to import'
              : busy || status === 'committing'
                ? 'Importing…'
                : `Import ${includedCount} item${includedCount === 1 ? '' : 's'}`
          }
          onPress={importNow}
          disabled={Boolean(blocker) || busy || status === 'committing'}
          loading={busy || status === 'committing'}
        />
        <Text style={[type.caption, { color: palette.inkFaint, textAlign: 'center' }]}>
          {blocker ?? 'Nothing is saved until you import. Tap a value to edit it, hold to see its source.'}
        </Text>
      </View>

      <EvidenceSheet visible={source != null} source={source} onClose={() => setSource(null)} />

      <EditSheet
        target={target}
        visible={target != null}
        included={editingProposal?.included ?? true}
        canToggleIncluded={editingProposal != null}
        canViewSource={Boolean(editingProposal && (editingProposal.evidenceId || editingProposal.start != null))}
        onClose={() => {
          setTarget(null);
          setEditKey(null);
        }}
        onSave={saveEdit}
        onToggleIncluded={() => {
          if (!editKey || !editingProposal) return;
          setIncluded(editKey, !editingProposal.included);
          setTarget(null);
          setEditKey(null);
        }}
        onViewSource={() => {
          if (!editingProposal) return;
          setTarget(null);
          openSource(editingProposal as Proposal<unknown>, valueLabel(editingProposal));
        }}
      />
    </View>
  );
}
