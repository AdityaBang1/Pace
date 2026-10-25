import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';

import {
  ensureDefaults,
  ensureStructure,
  readStructure,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import {
  getSetting,
  latestTranscript,
  listCourses,
  readGpaScaleValue,
  type CourseRow,
  type TranscriptRow,
} from '../../src/database/repo';
import { creditsValue, gpaValue } from '../../src/domain/academics';
import { parseScale } from '../../src/domain/gpa';
import {
  CREDIT_TARGET_KEY,
  GPA_TARGET_KEY,
  evaluatePace,
  parseTargets,
  type PaceStatus,
} from '../../src/domain/pace';
import { buildPathway, type PathwayView } from '../../src/domain/pathways';
import { buildRecord, type RecordView } from '../../src/domain/record';
import { requirementProgress } from '../../src/domain/requirements';
import {
  Button,
  Divider,
  EmptyState,
  IconButton,
  Notice,
  Screen,
  SectionHeader,
} from '../../src/ui/components';
import { PressableScale } from '../../src/ui/press';
import { space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * Home answers one question: where am I academically right now.
 *
 * One number, the two credit totals kept apart, the courses being taken, and the
 * plan the rest of the app is measuring. Everything below the number is a row that
 * opens the place it came from.
 */
interface HomeState {
  record: RecordView;
  transcript: TranscriptRow | null;
  pace: PaceStatus;
  pathway: PathwayView | null;
  hasTargets: boolean;
  /** A value the page printed is never presented as one Pace derived. */
  gpa: { value: number; label: string } | null;
}

export default function Home() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [data, setData] = useState<HomeState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      // A course imported before the academic structure existed has no year or
      // schedule; placing it first means every number below has a place to come from.
      await ensureStructure(db);
      await ensureDefaults(db);
      const [courses, transcript, scaleJson, gpaTarget, creditTarget] = await Promise.all([
        listCourses(db),
        latestTranscript(db),
        readGpaScaleValue(db),
        getSetting(db, GPA_TARGET_KEY, ''),
        getSetting(db, CREDIT_TARGET_KEY, ''),
      ]);
      const structure: AcademicStructure = await readStructure(db);
      const scale = parseScale(scaleJson);
      const record = buildRecord(scale, structure.years, structure.terms, structure.schedules, courses);
      const active =
        structure.pathways.find((row) => row.is_active === 1) ?? structure.pathways[0] ?? null;
      const pathway = active
        ? buildPathway(active, scale, structure.years, structure.schedules, courses)
        : null;
      const targets = parseTargets({ gpa: gpaTarget, credits: creditTarget });
      const pace = evaluatePace(
        pathway,
        requirementProgress(structure.requirements, structure.links, courses),
        targets,
      );
      const calculated = record.gpa.weighted;
      const reported = transcript?.reported_gpa ?? null;
      const gpa =
        calculated != null
          ? { value: calculated, label: 'Weighted · Calculated' }
          : reported != null
            ? { value: reported, label: 'Weighted · Reported' }
            : null;
      setData({
        record,
        transcript,
        pace,
        pathway,
        hasTargets: targets.gpa != null || targets.credits != null || structure.requirements.length > 0,
        gpa,
      });
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const current = data?.record.current.slice(0, 5) ?? [];
  const issue = data?.pace.issues[0] ?? null;
  const showVerdict = data != null && data.pace.state !== 'incomplete_setup';

  return (
    <Screen scroll edges="top" contentStyle={{ paddingBottom: space.xxxl }}>
      <View
        style={{
          paddingHorizontal: space.xl,
          paddingTop: space.md,
          flexDirection: 'row',
          alignItems: 'center',
        }}
      >
        <Text style={[type.display, { color: palette.ink, flex: 1 }]} accessibilityRole="header">
          Pace
        </Text>
        <IconButton glyph="+" label="Import a transcript" tone="accent" onPress={() => router.push('/scan')} />
        <IconButton glyph="⚙" label="Settings" onPress={() => router.push('/settings')} />
      </View>

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg, gap: space.sm }}>
          <Notice tone="danger" title="Your data could not be loaded" body={error} />
          <Button title="Try Again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {!error && data && data.record.courseCount === 0 ? (
        <View style={{ paddingTop: space.xxl }}>
          <EmptyState
            title="No academics yet"
            body="Import a transcript to begin."
            action={<Button title="Import Transcript" onPress={() => router.push('/scan')} />}
          />
        </View>
      ) : null}

      {!error && data && data.record.courseCount > 0 ? (
        <View>
          {/* The one number that matters, with what it rests on said quietly. */}
          <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl }}>
            {showVerdict ? (
              <PressableScale
                accessibilityRole="button"
                accessibilityLabel={issue ? issue.label : 'On pace'}
                onPress={() => router.push('/requirements')}
                style={{ marginBottom: space.md }}
              >
                <Text style={[type.label, { color: issue ? palette.warn : palette.accent }]}>
                  {issue ? 'NEEDS ATTENTION' : 'ON PACE'}
                </Text>
                {issue ? (
                  <Text style={[type.body, { color: palette.ink, marginTop: 2 }]} numberOfLines={2}>
                    {issue.label}
                  </Text>
                ) : (
                  <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                    {data.pace.pathwayName ?? 'Your plan'} reaches your targets
                  </Text>
                )}
              </PressableScale>
            ) : null}

            {data.gpa ? (
              <View>
                <Text
                  style={{
                    fontSize: 44,
                    lineHeight: 48,
                    fontWeight: '600',
                    letterSpacing: -1.4,
                    fontVariant: ['tabular-nums'],
                    color: palette.ink,
                  }}
                  numberOfLines={1}
                  adjustsFontSizeToFit
                >
                  {gpaValue(data.gpa.value)}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                  {data.gpa.label}
                </Text>
              </View>
            ) : null}

            <View style={{ flexDirection: 'row', marginTop: space.lg }}>
              <Metric value={creditsValue(data.record.credits.earned)} label="Earned" />
              <Metric value={creditsValue(data.record.credits.inProgress)} label="In Progress" />
            </View>

            {!data.hasTargets ? (
              <PressableScale
                accessibilityRole="button"
                accessibilityLabel="Set targets"
                onPress={() => router.push('/targets')}
                style={{ paddingVertical: space.sm }}
              >
                <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>
                  Set Targets
                </Text>
              </PressableScale>
            ) : null}
          </View>

          {current.length ? (
            <View>
              <SectionHeader
                right={
                  <Text style={[type.caption, { color: palette.inkFaint }]}>
                    {data.record.current.length}
                  </Text>
                }
              >
                Current
              </SectionHeader>
              <View style={{ paddingHorizontal: space.xl }}>
                {current.map((course, index) => (
                  <View key={course.id}>
                    {index > 0 ? <Divider /> : null}
                    <CourseRowRow
                      course={course}
                      onPress={() => router.push(`/course/${course.id}`)}
                    />
                  </View>
                ))}
                <Divider />
                <PressableScale
                  accessibilityRole="button"
                  accessibilityLabel="See all current courses"
                  onPress={() => router.push('/academics')}
                  style={{ paddingVertical: space.md }}
                >
                  <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>
                    See All
                  </Text>
                </PressableScale>
              </View>
            </View>
          ) : null}

          {data.pathway ? (
            <View>
              <SectionHeader>Plan</SectionHeader>
              <View style={{ paddingHorizontal: space.xl }}>
                <PressableScale
                  accessibilityRole="button"
                  accessibilityLabel={`Open plan ${data.pathway.pathway.name}`}
                  onPress={() => router.push('/plan')}
                  style={{ paddingVertical: space.md }}
                >
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                    <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                      {data.pathway.pathway.name}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted }]} numberOfLines={1}>
                      {[
                        data.pathway.gpa.weighted != null
                          ? `${gpaValue(data.pathway.gpa.weighted)} projected`
                          : null,
                        `${creditsValue(data.pathway.credits.all)} credits`,
                      ]
                        .filter(Boolean)
                        .join(' · ')}
                    </Text>
                  </View>
                </PressableScale>
              </View>
            </View>
          ) : null}

          {data.transcript ? (
            <View style={{ paddingHorizontal: space.xl, marginTop: space.xl }}>
              <Divider />
              <PressableScale
                accessibilityRole="button"
                accessibilityLabel="Open the imported transcript"
                onPress={() => router.push('/transcript')}
                style={{ paddingVertical: space.md }}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                  <Text style={[type.body, { color: palette.ink, flex: 1 }]}>Transcript</Text>
                  <Text style={[type.caption, { color: palette.inkMuted }]}>
                    {data.transcript.student_name ?? 'Imported'}
                  </Text>
                </View>
              </PressableScale>
              <Divider />
            </View>
          ) : null}
        </View>
      ) : null}
    </Screen>
  );
}

function CourseRowRow({ course, onPress }: { course: CourseRow; onPress: () => void }) {
  const { palette } = useTheme();
  return (
    <PressableScale
      accessibilityRole="button"
      accessibilityLabel={`Open ${course.title ?? 'course'}`}
      onPress={onPress}
      style={{ paddingVertical: space.md }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
        <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
          {course.title ?? course.code ?? 'Course'}
        </Text>
        <Text style={[type.caption, { color: palette.inkMuted }]}>{creditsValue(course.credits)}</Text>
      </View>
    </PressableScale>
  );
}

/** One number and its label: the number is 30pt, the label is quiet. */
function Metric({ value, label }: { value: string; label: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flex: 1, paddingRight: space.sm }}>
      <Text
        style={{
          fontSize: 30,
          lineHeight: 34,
          fontWeight: '600',
          letterSpacing: -0.8,
          fontVariant: ['tabular-nums'],
          color: palette.ink,
        }}
        numberOfLines={1}
        adjustsFontSizeToFit
      >
        {value}
      </Text>
      <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
        {label}
      </Text>
    </View>
  );
}
