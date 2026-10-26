import { Button as NativeButton, Section } from '@expo/ui/swift-ui';
import * as Haptics from 'expo-haptics';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Alert } from 'react-native';

import {
  getCourse,
  importForCourse,
  sourceUriForImport,
  undoImport,
  type CourseDetail,
  type EvidenceRow,
} from '../../src/database/repo';
import { creditsValue, termLabel } from '../../src/domain/academics';
import { LEVELS, levelLabel, levelOf } from '../../src/domain/gpa';
import { deleteStoredSources } from '../../src/features/scan/storage';
import { SourceSection, type SourceLine } from '../../src/ui/source';
import { LabeledContent, NativeScreen, NativeText, Value } from '../../src/ui/native';
import { daysLabel, documentTypeLabel, evidenceExcerpt, relativeDay, timeRangeLabel, weightLabel } from '../../src/ui/format';

function methodLabel(row: EvidenceRow): string {
  return row.confidence_source === 'model'
    ? 'Classified on this device by the section model'
    : 'Read from the page by the deterministic reader';
}

export default function CourseScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const [detail, setDetail] = useState<CourseDetail | null>(null);
  const [missing, setMissing] = useState(false);
  const [source, setSource] = useState<SourceLine | null>(null);
  const [importId, setImportId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const next = await getCourse(db, id);
      setDetail(next);
      setMissing(!next);
      setLoadError(null);
      const found = next ? await importForCourse(db, id) : null;
      setImportId(found?.status === 'committed' ? found.id : null);
    } catch (cause) {
      // A read error is not "this course is gone": saying so would invite the user
      // to believe an import was undone.
      setLoadError(cause instanceof Error ? cause.message : 'This course could not be read.');
    }
  }, [db, id]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  if (missing || !detail) {
    return (
      <NativeScreen title="Course" onBack={() => router.back()}>
        {loadError ? (
          <Section title="This course could not be read">
            <NativeText>{loadError}</NativeText>
            <NativeButton label="Try again" systemImage="arrow.clockwise" onPress={() => void load()} />
          </Section>
        ) : (
          <Section title={missing ? 'This course is gone' : 'Loading'}>
            <NativeText>{missing ? 'It was removed or its import was undone.' : 'Reading the course.'}</NativeText>
          </Section>
        )}
      </NativeScreen>
    );
  }

  const { course, grading, meetings, policies, document } = detail;
  const total = grading.reduce((sum, row) => sum + row.weight, 0);
  const evidenceById = new Map(detail.evidence.map((row) => [row.id, row]));
  const inProgress = course.in_progress === 1;
  const level = levelLabel(levelOf(course.title, true));

  const openSource = (label: string, evidenceId: string | null) => {
    if (!evidenceId) return;
    const row = evidenceById.get(evidenceId);
    if (!row) return;
    let bbox: SourceLine['bbox'] = null;
    if (row.bbox) {
      try {
        bbox = JSON.parse(row.bbox) as SourceLine['bbox'];
      } catch {
        bbox = null;
      }
    }
    setSource({
      label,
      quote: document?.ocr_text ? evidenceExcerpt(document.ocr_text, row.start, row.end) : row.text ?? '',
      page: row.page,
      bbox,
      confidence: row.model_confidence,
      confidenceSource: row.confidence_source === 'model' ? 'model' : 'rule',
      method: methodLabel(row),
      // A PDF is not an image React Native can draw, and a document imported with
      // the storage policy off has no URI at all; both fall back to the quoted line.
      image:
        document?.source_uri && !/\.pdf(\?|$)/i.test(document.source_uri)
          ? { uri: document.source_uri, width: 0, height: 0 }
          : null,
    });
  };

  const undo = () => {
    if (!importId) return;
    Alert.alert(
      'Undo this import?',
      'The course and everything imported with it will be removed. The document stays in recent imports.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Undo import',
          style: 'destructive',
          onPress: () => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
            void (async () => {
              // The page file is found before the rows go: after the undo nothing
              // remembers which file belonged to this document.
              const uri = await sourceUriForImport(db, importId);
              await undoImport(db, importId);
              await deleteStoredSources([uri]);
              router.back();
            })();
          },
        },
      ],
    );
  };

  const facts: { label: string; value: string }[] = [];
  if (course.term) facts.push({ label: 'Term', value: termLabel(course.term) });
  if (course.code) facts.push({ label: 'Code', value: course.code });
  if (course.instructor) facts.push({ label: 'Instructor', value: course.instructor });
  if (!inProgress && level !== 'Regular') facts.push({ label: 'Level', value: level });
  if (course.credits != null) facts.push({ label: 'Credit', value: creditsValue(course.credits) });

  return (
    <NativeScreen
      title={course.title ?? course.code ?? 'Course'}
      onBack={() => router.back()}
      actions={[
        {
          systemName: 'pencil',
          label: 'Edit this course',
          onPress: () => router.push(`/editors/course?id=${course.id}`),
        },
      ]}
    >
      {source ? <SourceSection source={source} onClose={() => setSource(null)} /> : null}

      <Section
        title="Mark"
        footer={
          course.edited === 1 && course.original_grade && course.original_grade !== course.grade ? (
            <NativeText>{`Edited · the transcript said ${course.original_grade}`}</NativeText>
          ) : undefined
        }
      >
          <LabeledContent label={inProgress ? 'Status' : 'Final'}>
            <Value>{inProgress ? 'In progress' : (course.grade ?? '—')}</Value>
          </LabeledContent>
          {course.expected_grade ? (
            <LabeledContent label="Expected">
              <Value>{course.expected_grade}</Value>
            </LabeledContent>
          ) : null}
        </Section>

        {facts.length ? (
          <Section title="Details">
            {facts.map((fact) => (
              <LabeledContent key={fact.label} label={fact.label}>
                <Value>{fact.value}</Value>
              </LabeledContent>
            ))}
            {!inProgress && level === 'Regular'
              ? LEVELS.filter((row) => row.value === 'regular').map((row) => (
                  <LabeledContent key={row.value} label="Level">
                    <Value>{levelLabel(row.value)}</Value>
                  </LabeledContent>
                ))
              : null}
          </Section>
        ) : null}

        {grading.length ? (
          <Section title="Grading" footer={<NativeText>{`${grading.length} categories · ${weightLabel(total)}`}</NativeText>}>
            {grading.map((row) => (
              <NativeButton
                key={row.id}
                label={`${row.name} · ${weightLabel(row.weight)}`}
                systemImage="doc.text.magnifyingglass"
                onPress={() => openSource(`${row.name} ${weightLabel(row.weight)}`, row.evidence_id)}
              />
            ))}
          </Section>
        ) : null}

        {meetings.length ? (
          <Section title="Meeting">
            {meetings.map((meeting) => {
              let days: string[] = [];
              try {
                days = JSON.parse(meeting.days) as string[];
              } catch {
                days = [];
              }
              const time = timeRangeLabel(meeting.start_time, meeting.end_time);
              return (
                <NativeButton
                  key={meeting.id}
                  label={[daysLabel(days), time, meeting.room].filter(Boolean).join(' · ')}
                  systemImage="calendar"
                  onPress={() => openSource(daysLabel(days), meeting.evidence_id)}
                />
              );
            })}
          </Section>
        ) : null}

        {policies.length ? (
          <Section title="Policies">
            {policies.map((policy) => (
              <NativeButton
                key={policy.id}
                label={`${policy.field.replace(/_/g, ' ')} · ${policy.value}`}
                systemImage="text.alignleft"
                onPress={() => openSource(policy.value, policy.evidence_id)}
              />
            ))}
          </Section>
        ) : null}

        <Section
          title="Source"
          footer={
            document ? (
              <NativeText>
                {`${documentTypeLabel(document.document_type)} · read ${relativeDay(document.created_at).toLowerCase()}${
                  document.source_uri && !/\.pdf(\?|$)/i.test(document.source_uri) ? ' · page kept' : ''
                }`}
              </NativeText>
            ) : undefined
          }
        >
          {course.evidence_id ? (
            <NativeButton
              label="The printed row"
              systemImage="text.magnifyingglass"
              onPress={() => openSource(course.title ?? 'Transcript row', course.evidence_id)}
            />
          ) : null}
          {importId ? (
            <NativeButton label="Undo import" systemImage="arrow.uturn.backward" role="destructive" onPress={undo} />
          ) : null}
        </Section>
    </NativeScreen>
  );
}
