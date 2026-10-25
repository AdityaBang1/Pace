import * as Haptics from 'expo-haptics';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Alert, Text, View } from 'react-native';

import {
  getCourse,
  importForCourse,
  sourceUriForImport,
  undoImport,
  type CourseDetail,
  type EvidenceRow,
} from '../../src/database/repo';
import { creditsValue, termLabel } from '../../src/domain/academics';
import { levelLabel, levelOf } from '../../src/domain/gpa';
import { deleteStoredSources } from '../../src/features/scan/storage';
import { EvidenceSheet, type SourceLine } from '../../src/features/review/EvidenceSheet';
import {
  Button,
  Card,
  Chip,
  Divider,
  EmptyState,
  FieldRow,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../../src/ui/components';
import { daysLabel, documentTypeLabel, evidenceExcerpt, relativeDay, timeRangeLabel, weightLabel } from '../../src/ui/format';
import { space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

function methodLabel(row: EvidenceRow): string {
  return row.confidence_source === 'model'
    ? 'Classified on this device by the section model'
    : 'Read from the page by the deterministic reader';
}

/** One metric: the number carries the screen, its label only names it. */
function Metric({ value, label }: { value: string; label: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flex: 1, gap: 2 }}>
      <Text style={[type.display, { color: palette.ink, fontVariant: ['tabular-nums'] }]}>{value}</Text>
      <Text style={[type.caption, { color: palette.inkMuted }]}>{label}</Text>
    </View>
  );
}

export default function CourseScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
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
      <Screen>
        <PageHeader title="Course" onBack={() => router.back()} />
        {loadError ? (
          <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
            <Notice title="This course could not be read" body={loadError} tone="danger" />
            <Button title="Try again" kind="secondary" onPress={() => void load()} />
          </View>
        ) : (
          <EmptyState
            title={missing ? 'This course is gone' : 'Loading'}
            body={missing ? 'It was removed or its import was undone.' : 'Reading the course.'}
          />
        )}
      </Screen>
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

  // Values that were not printed are omitted, never shown as a dash.
  const facts: { label: string; value: string }[] = [];
  if (course.term) facts.push({ label: 'Term', value: termLabel(course.term) });
  if (course.code) facts.push({ label: 'Code', value: course.code });
  if (course.instructor) facts.push({ label: 'Instructor', value: course.instructor });
  if (!inProgress && level !== 'Regular') facts.push({ label: 'Level', value: level });

  return (
    <Screen scroll>
      <PageHeader title={course.title ?? course.code ?? 'Course'} onBack={() => router.back()} />

      <View style={{ paddingHorizontal: space.xl, gap: space.lg }}>
        {(course.grade || course.credits != null || inProgress) && (
          <View style={{ flexDirection: 'row', gap: space.lg }}>
            {inProgress ? (
              <Metric value={creditsValue(course.credits)} label="Credit" />
            ) : course.grade ? (
              <Metric value={course.grade} label="Final Mark" />
            ) : null}
            {!inProgress && course.credits != null ? (
              <Metric value={creditsValue(course.credits)} label="Credit" />
            ) : null}
            {inProgress ? (
              <View style={{ justifyContent: 'center' }}>
                <Chip label="In progress" tone="accent" />
              </View>
            ) : null}
          </View>
        )}

        {facts.length ? (
          <Card padded={false}>
            <View style={{ paddingHorizontal: space.lg }}>
              {facts.map((fact, index) => (
                <View key={fact.label}>
                  {index > 0 ? <Divider /> : null}
                  <FieldRow label={fact.label} value={fact.value} />
                </View>
              ))}
            </View>
          </Card>
        ) : null}
      </View>

      {grading.length ? (
        <>
          <SectionHeader
            right={<Text style={[type.caption, { color: palette.inkMuted }]}>{weightLabel(total)}</Text>}
          >
            GRADING
          </SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {grading.map((row, index) => (
                <View key={row.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <View style={{ paddingHorizontal: space.lg }}>
                    <FieldRow
                      label={row.name}
                      value={weightLabel(row.weight)}
                      onPress={() => openSource(`${row.name} ${weightLabel(row.weight)}`, row.evidence_id)}
                    />
                  </View>
                </View>
              ))}
            </Card>
            {/* The bar is the hierarchy, read left to right: category widths are weights. */}
            <View
              style={{
                flexDirection: 'row',
                height: 6,
                borderRadius: 3,
                overflow: 'hidden',
                backgroundColor: palette.surfaceSunken,
                marginTop: space.md,
              }}
            >
              {grading.map((row, index) => (
                <View
                  key={row.id}
                  style={{
                    flex: Math.max(row.weight, 0.01),
                    backgroundColor: palette.accent,
                    opacity: 1 - Math.min(index * 0.14, 0.5),
                  }}
                />
              ))}
            </View>
          </View>
        </>
      ) : null}

      {meetings.length ? (
        <>
          <SectionHeader>MEETING</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {meetings.map((meeting, index) => {
                let days: string[] = [];
                try {
                  days = JSON.parse(meeting.days) as string[];
                } catch {
                  days = [];
                }
                const time = timeRangeLabel(meeting.start_time, meeting.end_time);
                return (
                  <View key={meeting.id}>
                    {index > 0 ? <Divider inset={space.lg} /> : null}
                    <View style={{ paddingHorizontal: space.lg }}>
                      <FieldRow
                        label={daysLabel(days)}
                        value={[time, meeting.room].filter(Boolean).join(' · ') || '—'}
                        onPress={() => openSource(daysLabel(days), meeting.evidence_id)}
                      />
                    </View>
                  </View>
                );
              })}
            </Card>
          </View>
        </>
      ) : null}

      {policies.length ? (
        <>
          <SectionHeader>POLICIES</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {policies.map((policy, index) => (
                <View key={policy.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <View style={{ paddingHorizontal: space.lg }}>
                    <FieldRow
                      label={policy.field.replace(/_/g, ' ')}
                      value={<Text style={[type.caption, { color: palette.inkMuted, textAlign: 'right' }]} numberOfLines={3}>{policy.value}</Text>}
                      onPress={() => openSource(policy.value, policy.evidence_id)}
                    />
                  </View>
                </View>
              ))}
            </Card>
          </View>
        </>
      ) : null}

      <SectionHeader>SOURCE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        {course.evidence_id ? (
          <Card padded={false}>
            <View style={{ paddingHorizontal: space.lg }}>
              <FieldRow
                label="The printed row"
                value="View"
                onPress={() => openSource(course.title ?? 'Transcript row', course.evidence_id)}
              />
            </View>
          </Card>
        ) : null}
        {document ? (
          <Text style={[type.caption, { color: palette.inkFaint }]}>
            {documentTypeLabel(document.document_type)} · read {relativeDay(document.created_at).toLowerCase()}
            {document.source_uri && !/\.pdf(\?|$)/i.test(document.source_uri) ? ' · page kept' : ''}
          </Text>
        ) : null}
        {importId ? <Button title="Undo import" kind="quiet" onPress={undo} /> : null}
      </View>

      <EvidenceSheet visible={source !== null} source={source} onClose={() => setSource(null)} />
    </Screen>
  );
}
