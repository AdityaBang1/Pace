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
import { deleteStoredSources } from '../../src/features/scan/storage';
import { PressableScale } from '../../src/ui/press';
import { EvidenceSheet, type SourceLine } from '../../src/features/review/EvidenceSheet';
import { Button, Card, Divider, EmptyState, Notice, PageHeader, Screen, SectionHeader } from '../../src/ui/components';
import {
  creditsLabel,
  daysLabel,
  documentTypeLabel,
  evidenceExcerpt,
  relativeDay,
  timeRangeLabel,
  weightLabel,
} from '../../src/ui/format';
import { radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

function methodLabel(row: EvidenceRow): string {
  return row.confidence_source === 'model'
    ? 'Classified on this device by the section model'
    : 'Read from the page by the deterministic reader';
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
            body={
              missing
                ? 'It was removed or its import was undone.'
                : 'Reading what Pace stored for this course.'
            }
          />
        )}
      </Screen>
    );
  }

  const { course, grading, meetings, policies, document } = detail;
  const total = grading.reduce((sum, row) => sum + row.weight, 0);
  const hasTotal = grading.length > 0;
  const credits = creditsLabel(course.credits);
  const evidenceById = new Map(detail.evidence.map((row) => [row.id, row]));

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
            // A destructive commit: the heavier impact lands with the action, not
            // with the navigation that follows it.
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

  return (
    <Screen scroll>
      <PageHeader
        title={course.title ?? course.code ?? 'Course'}
        subtitle={[course.code, course.instructor, credits].filter(Boolean).join(' · ') || undefined}
        onBack={() => router.back()}
      />

      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <View style={{ flexDirection: 'row', gap: space.sm, flexWrap: 'wrap' }}>
          {course.document_type ? (
            <View
              style={{
                backgroundColor: palette.surfaceSunken,
                borderRadius: radius.sm,
                paddingHorizontal: 9,
                paddingVertical: 5,
              }}
            >
              <Text style={[type.caption, { color: palette.inkMuted, fontWeight: '600' }]}>
                {documentTypeLabel(course.document_type)}
              </Text>
            </View>
          ) : null}
          {document ? (
            <Text style={[type.caption, { color: palette.inkFaint, alignSelf: 'center' }]}>
              Imported {relativeDay(document.created_at).toLowerCase()}
            </Text>
          ) : null}
        </View>

        {course.instructor || credits ? (
          <Card>
            {course.instructor ? (
              <>
                <Text style={[type.label, { color: palette.inkFaint }]}>INSTRUCTOR</Text>
                <Text style={[type.body, { color: palette.ink, marginTop: 4 }]}>{course.instructor}</Text>
              </>
            ) : null}
            {course.instructor && credits ? <View style={{ height: space.md }} /> : null}
            {credits ? (
              <>
                <Text style={[type.label, { color: palette.inkFaint }]}>CREDITS</Text>
                <Text style={[type.body, { color: palette.ink, marginTop: 4 }]}>{credits}</Text>
              </>
            ) : null}
          </Card>
        ) : null}
      </View>

      <SectionHeader
        right={hasTotal ? <Text style={[type.caption, { color: palette.inkMuted }]}>Total {weightLabel(total)}</Text> : null}
      >
        GRADING
      </SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          {grading.length === 0 ? (
            <View style={{ padding: space.lg }}>
              <Text style={[type.caption, { color: palette.inkMuted }]}>No grading breakdown was imported.</Text>
            </View>
          ) : null}
          {grading.map((row, index) => (
            <View key={row.id}>
              {index > 0 ? <Divider inset={space.lg} /> : null}
              <PressableScale
                accessibilityLabel={`${row.name}, ${weightLabel(row.weight)}. View source.`}
                onPress={() => openSource(`${row.name} ${weightLabel(row.weight)}`, row.evidence_id)}
                style={{
                  paddingHorizontal: space.lg,
                  paddingVertical: space.md,
                  flexDirection: 'row',
                  alignItems: 'center',
                  gap: space.md,
                }}
              >
                <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                  {row.name}
                </Text>
                <Text style={[type.number, { color: palette.ink }]}>{weightLabel(row.weight)}</Text>
              </PressableScale>
            </View>
          ))}
        </Card>

        {hasTotal ? (
          <View style={{ marginTop: space.md }}>
            {/* The bar is the hierarchy, read left to right: category widths are weights. */}
            <View
              style={{
                flexDirection: 'row',
                height: 6,
                borderRadius: 3,
                overflow: 'hidden',
                backgroundColor: palette.surfaceSunken,
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
            <View style={{ flexDirection: 'row', marginTop: space.sm }}>
              <Text style={[type.caption, { color: palette.inkFaint }]}>
                {Math.abs(total - 100) <= 5
                  ? 'Weights add up to 100%'
                  : `Weights add up to ${weightLabel(total)} — the syllabus may use nested weights`}
              </Text>
            </View>
          </View>
        ) : null}
      </View>

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
                    <PressableScale
                      onPress={() => openSource(daysLabel(days), meeting.evidence_id)}
                      style={{ padding: space.lg, gap: 2 }}
                    >
                      <Text style={[type.bodyStrong, { color: palette.ink }]}>{daysLabel(days)}</Text>
                      <Text style={[type.caption, { color: palette.inkMuted }]}>
                        {[time, meeting.room].filter(Boolean).join(' · ') || 'No time recorded'}
                      </Text>
                    </PressableScale>
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
                  <PressableScale
                    onPress={() => openSource(policy.value, policy.evidence_id)}
                    style={{ padding: space.lg, gap: 3 }}
                  >
                    <Text style={[type.bodyStrong, { color: palette.ink }]}>
                      {policy.field.replace(/_/g, ' ')}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted }]} numberOfLines={3}>
                      {policy.value}
                    </Text>
                  </PressableScale>
                </View>
              ))}
            </Card>
          </View>
        </>
      ) : null}

      <SectionHeader>SOURCE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Card>
          <Text style={[type.body, { color: palette.ink }]}>
            {document ? `${documentTypeLabel(document.document_type)} document` : 'No source document'}
          </Text>
          <Text style={[type.caption, { color: palette.inkMuted, marginTop: 4 }]}>
            {document
              ? `Read ${relativeDay(document.created_at).toLowerCase()} · ${document.ocr_text.length.toLocaleString()} characters of text kept for checking`
              : 'The document behind this course was removed.'}
          </Text>
        </Card>

        {importId ? <Button title="Undo this import" kind="quiet" onPress={undo} /> : null}
      </View>
    </Screen>
  );
}
