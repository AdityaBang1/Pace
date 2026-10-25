import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert, Modal, Pressable, Text, View } from 'react-native';

import {
  deleteSchedule,
  listScheduleCourses,
  readStructure,
  setMainSchedule,
  updateCourse,
  type AcademicStructure,
  type AcademicYearRow,
  type ScheduleRow,
} from '../../src/database/academicRepo';
import { readGpaScaleValue, type CourseRow } from '../../src/database/repo';
import { creditsValue, gpaValue } from '../../src/domain/academics';
import { LETTERS, parseScale, type GpaScale } from '../../src/domain/gpa';
import { computeGpa, creditsOf, statusOf } from '../../src/domain/record';
import {
  Button,
  Divider,
  EmptyState,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../../src/ui/components';
import { PressableScale } from '../../src/ui/press';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * One schedule: the courses one year holds under one name.
 *
 * A schedule is where a scenario lives, so the projected number and the expected
 * grades that produce it are on the same screen. Editing an expected grade rewrites
 * this schedule's own row and nothing else — the actual record of a past year is a
 * different schedule, and a copy of it is a third.
 */
export default function ScheduleScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [schedule, setSchedule] = useState<ScheduleRow | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [editing, setEditing] = useState<CourseRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const [next, rows, scaleValue] = await Promise.all([
        readStructure(db),
        listScheduleCourses(db, id),
        readGpaScaleValue(db),
      ]);
      setStructure(next);
      setSchedule(next.schedules.find((row) => row.id === id) ?? null);
      setCourses(rows);
      setScale(parseScale(scaleValue));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'This schedule could not be read.');
    } finally {
      setLoaded(true);
    }
  }, [db, id]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const year: AcademicYearRow | null = useMemo(
    () => structure?.years.find((row) => row.id === schedule?.academic_year_id) ?? null,
    [structure, schedule],
  );

  const gpa = useMemo(() => computeGpa(scale, courses), [scale, courses]);
  const credits = useMemo(() => creditsOf(courses), [courses]);
  const projected = courses.some((course) => statusOf(course) !== 'completed');
  const isMain = schedule?.is_main === 1;
  const isScenario = schedule?.type === 'scenario' || schedule?.source === 'duplicate';

  const chooseExpected = async (course: CourseRow, grade: string) => {
    setEditing(null);
    await updateCourse(db, course.id, { expectedGrade: grade });
    await load();
  };

  const makeMain = () => {
    if (!schedule) return;
    Alert.alert('Make main?', `${schedule.name} becomes this year's default schedule.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Make Main',
        onPress: () => {
          void (async () => {
            await setMainSchedule(db, schedule.id);
            await load();
          })();
        },
      },
    ]);
  };

  const remove = () => {
    if (!schedule) return;
    Alert.alert('Delete schedule?', 'Its courses stay, but lose their place in this year.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deleteSchedule(db, schedule.id);
            router.back();
          })();
        },
      },
    ]);
  };

  if (loaded && !schedule) {
    return (
      <Screen edges="top">
        <PageHeader title="Schedule" onBack={() => router.back()} />
        <EmptyState title="This schedule is gone" body="It was deleted or its year was removed." />
      </Screen>
    );
  }

  if (!schedule || !structure) {
    return (
      <Screen edges="top">
        <PageHeader title="Schedule" onBack={() => router.back()} />
        {error ? (
          <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
            <Notice tone="danger" title="Couldn't read this schedule" body={error} />
            <Button title="Try Again" kind="secondary" onPress={() => void load()} />
          </View>
        ) : null}
      </Screen>
    );
  }

  return (
    <Screen scroll edges="top">
      <PageHeader title={schedule.name} subtitle={year?.name ?? undefined} onBack={() => router.back()} />

      <View style={{ flexDirection: 'row', paddingHorizontal: space.xl, paddingTop: space.lg }}>
        <Metric value={gpaValue(gpa.weighted)} label={projected ? 'Projected' : 'Weighted'} />
        <Metric value={creditsValue(credits.all)} label="Credits" />
        <Metric value={String(courses.length)} label={courses.length === 1 ? 'Course' : 'Courses'} />
      </View>

      <View
        style={{
          flexDirection: 'row',
          gap: space.sm,
          paddingHorizontal: space.xl,
          paddingTop: space.lg,
          flexWrap: 'wrap',
        }}
      >
        <Button
          title="Edit"
          kind="secondary"
          compact
          onPress={() => router.push(`/editors/schedule?id=${schedule.id}`)}
        />
        <Button
          title="Duplicate"
          kind="secondary"
          compact
          onPress={() => router.push(`/editors/schedule?duplicateFrom=${schedule.id}`)}
        />
        {!isMain ? <Button title="Make Main" kind="secondary" compact onPress={makeMain} /> : null}
        {isScenario || year?.status === 'future' ? (
          <Button title="Delete" kind="quiet" compact onPress={remove} />
        ) : null}
      </View>

      <SectionHeader
        right={<Text style={[type.caption, { color: palette.inkFaint }]}>{courses.length}</Text>}
      >
        Courses
      </SectionHeader>
      <Divider />

      {courses.length === 0 ? (
        <EmptyState
          title="No courses"
          body="Add one to start this schedule."
          action={
            <Button
              title="Add Course"
              compact
              onPress={() =>
                router.push(
                  `/editors/course?mode=add&scheduleId=${schedule.id}&yearId=${schedule.academic_year_id}`,
                )
              }
            />
          }
        />
      ) : (
        courses.map((course) => {
          const status = statusOf(course);
          const mark = status === 'completed' ? (course.grade ?? '—') : (course.expected_grade ?? '—');
          return (
            <View key={course.id}>
              <PressableScale
                accessibilityLabel={`Open ${course.title ?? 'course'}`}
                onPress={() => router.push(`/course/${course.id}`)}
                style={{ paddingVertical: space.md, paddingHorizontal: space.xl }}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                  <View style={{ flex: 1 }}>
                    <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={2}>
                      {course.title ?? course.code ?? 'Course'}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                      {[
                        course.level === 'ap' ? 'AP' : course.level === 'honors' ? 'Honors' : 'Regular',
                        `${creditsValue(course.credits)} cr`,
                        status === 'completed' ? 'Completed' : status === 'current' ? 'In progress' : 'Planned',
                      ].join(' · ')}
                    </Text>
                  </View>
                  {status === 'completed' ? (
                    <Text style={[type.number, { color: palette.ink }]}>{mark}</Text>
                  ) : (
                    <Pressable
                      accessibilityRole="button"
                      accessibilityLabel={`Expected grade for ${course.title ?? 'course'}`}
                      hitSlop={8}
                      onPress={() => setEditing(course)}
                    >
                      <Text style={[type.number, { color: palette.accent }]}>
                        {course.expected_grade ? `Expected ${course.expected_grade}` : 'Set grade'}
                      </Text>
                    </Pressable>
                  )}
                </View>
              </PressableScale>
              <Divider inset={space.xl} />
            </View>
          );
        })
      )}

      {projected && gpa.weighted != null ? (
        <>
          <SectionHeader>What If</SectionHeader>
          <View style={{ paddingHorizontal: space.xl, gap: space.xs }}>
            <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: space.sm }}>
              <Text
                style={[
                  type.display,
                  { color: palette.ink, fontSize: 34, fontVariant: ['tabular-nums'] },
                ]}
              >
                {gpaValue(gpa.weighted)}
              </Text>
              <Text style={[type.caption, { color: palette.inkMuted }]}>
                {gpa.missing > 0 ? `${gpa.missing} need a grade` : 'Projected weighted GPA'}
              </Text>
            </View>
            {isScenario ? (
              <Button
                title="Save as Schedule"
                kind="secondary"
                compact
                onPress={() => router.push(`/editors/schedule?id=${schedule.id}&saveAs=1`)}
              />
            ) : null}
          </View>
        </>
      ) : null}

      <View style={{ height: space.xxl }} />

      <Modal
        visible={editing !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setEditing(null)}
      >
        <Pressable
          accessibilityLabel="Close grade picker"
          onPress={() => setEditing(null)}
          style={{
            flex: 1,
            backgroundColor: palette.overlay,
            justifyContent: 'flex-end',
            padding: space.lg,
          }}
        >
          <View
            style={{
              backgroundColor: palette.surface,
              borderRadius: radius.lg,
              borderWidth: hairline,
              borderColor: palette.line,
              padding: space.md,
              gap: space.xs,
            }}
          >
            <Text style={[type.label, { color: palette.inkFaint, paddingHorizontal: space.sm }]}>
              Expected
            </Text>
            {LETTERS.map((letter) => (
              <PressableScale
                key={letter}
                accessibilityRole="button"
                accessibilityLabel={`Expected ${letter}`}
                onPress={() => {
                  if (editing) void chooseExpected(editing, letter);
                }}
                style={{ paddingVertical: space.md, paddingHorizontal: space.sm }}
              >
                <Text style={[type.body, { color: palette.ink }]}>{letter}</Text>
              </PressableScale>
            ))}
          </View>
        </Pressable>
      </Modal>
    </Screen>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flex: 1 }}>
      <Text
        style={{
          fontSize: 28,
          lineHeight: 32,
          fontWeight: '600',
          letterSpacing: -0.6,
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
