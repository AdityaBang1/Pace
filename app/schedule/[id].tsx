import { Button as NativeButton, Section } from '@expo/ui/swift-ui';
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
import { LabeledContent, NativeScreen, NativeText, Value } from '../../src/ui/native';
import { PressableScale } from '../../src/ui/press';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

const levelOfCourse = (course: CourseRow): string =>
  course.level === 'ap' ? 'AP' : course.level === 'honors' ? 'Honors' : 'Regular';

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
      <NativeScreen title="Schedule" onBack={() => router.back()}>
        <Section title="This schedule is gone">
          <NativeText>It was deleted or its year was removed.</NativeText>
        </Section>
      </NativeScreen>
    );
  }

  if (!schedule || !structure) {
    return (
      <NativeScreen title="Schedule" onBack={() => router.back()}>
        {error ? (
          <Section title="Couldn’t read this schedule">
            <NativeText>{error}</NativeText>
            <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void load()} />
          </Section>
        ) : null}
      </NativeScreen>
    );
  }

  const pending = courses.filter((course) => statusOf(course) !== 'completed');

  return (
    <>
      <NativeScreen title={schedule.name} subtitle={year?.name ?? undefined} onBack={() => router.back()}>
        <Section title={projected ? 'Projected' : 'Weighted'}>
          <LabeledContent label="GPA">
            <Value>{gpa.weighted != null ? gpaValue(gpa.weighted) : '—'}</Value>
          </LabeledContent>
          <LabeledContent label="Credits">
            <Value>{creditsValue(credits.all)}</Value>
          </LabeledContent>
          <LabeledContent label="Courses">
            <Value>{String(courses.length)}</Value>
          </LabeledContent>
        </Section>

        <Section title="Schedule">
          <NativeButton
            label="Edit"
            systemImage="pencil"
            onPress={() => router.push(`/editors/schedule?id=${schedule.id}`)}
          />
          <NativeButton
            label="Duplicate"
            systemImage="plus.square.on.square"
            onPress={() => router.push(`/editors/schedule?duplicateFrom=${schedule.id}`)}
          />
          {!isMain ? <NativeButton label="Make Main" systemImage="checkmark.circle" onPress={makeMain} /> : null}
          {isScenario || year?.status === 'future' ? (
            <NativeButton label="Delete" systemImage="trash" role="destructive" onPress={remove} />
          ) : null}
        </Section>

        <Section
          title="Courses"
          footer={courses.length === 0 ? <NativeText>Add one to start this schedule.</NativeText> : undefined}
        >
          {courses.map((course) => {
            const status = statusOf(course);
            const mark = status === 'completed' ? (course.grade ?? '—') : (course.expected_grade ?? '—');
            return (
              <NativeButton
                key={course.id}
                label={`${course.title ?? course.code ?? 'Course'} · ${levelOfCourse(course)} · ${creditsValue(
                  course.credits,
                )} cr · ${mark}`}
                systemImage={status === 'completed' ? 'checkmark.circle' : 'circle.dashed'}
                onPress={() => router.push(`/course/${course.id}`)}
              />
            );
          })}
          <NativeButton
            label="Add Course"
            systemImage="plus"
            onPress={() =>
              router.push(`/editors/course?mode=add&scheduleId=${schedule.id}&yearId=${schedule.academic_year_id}`)
            }
          />
        </Section>

        {pending.length ? (
          <Section
            title="Expected grades"
            footer={<NativeText>Tap a course to change the grade its projection uses.</NativeText>}
          >
            {pending.map((course) => (
              <NativeButton
                key={course.id}
                label={`${course.title ?? course.code ?? 'Course'} · ${course.expected_grade ?? 'Set grade'}`}
                systemImage="a.circle"
                onPress={() => setEditing(course)}
              />
            ))}
          </Section>
        ) : null}

        {projected ? (
          <Section
            title="What If"
            footer={
              <NativeText>
                {gpa.missing > 0 ? `${gpa.missing} need a grade` : 'Projected weighted GPA'}
              </NativeText>
            }
          >
            <LabeledContent label="Projected GPA">
              <Value>{gpa.weighted != null ? gpaValue(gpa.weighted) : '—'}</Value>
            </LabeledContent>
            {isScenario ? (
              <NativeButton
                label="Save as Schedule"
                systemImage="square.and.arrow.down"
                onPress={() => router.push(`/editors/schedule?id=${schedule.id}&saveAs=1`)}
              />
            ) : null}
          </Section>
        ) : null}
      </NativeScreen>

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
            <Text style={[type.label, { color: palette.inkFaint, paddingHorizontal: space.sm }]}>Expected</Text>
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
    </>
  );
}
