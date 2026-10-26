import { Button as NativeButton, Menu, Section } from '@expo/ui/swift-ui';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert } from 'react-native';

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

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [schedule, setSchedule] = useState<ScheduleRow | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [scale, setScale] = useState<GpaScale | null>(null);
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
          footer={
            <NativeText>
              {courses.length === 0
                ? 'Add one to start this schedule.'
                : 'A course that is not finished yet opens its expected grade.'}
            </NativeText>
          }
        >
          {courses.map((course) => {
            const status = statusOf(course);
            const completed = status === 'completed';
            const mark = completed ? (course.grade ?? '—') : (course.expected_grade ?? 'No grade');
            const label = `${course.title ?? course.code ?? 'Course'} · ${levelOfCourse(course)} · ${creditsValue(
              course.credits,
            )} cr · ${mark}`;
            // One row per course, not a course list and then a second list of the
            // same courses with a grade beside them. A finished course opens its
            // page; a projected one opens the grade picker over the same row, and
            // the course page is the last item in that menu.
            return completed ? (
              <NativeButton
                key={course.id}
                label={label}
                systemImage="checkmark.circle"
                onPress={() => router.push(`/course/${course.id}`)}
              />
            ) : (
              <Menu key={course.id} label={label} systemImage="a.circle">
                {LETTERS.map((letter) => (
                  <NativeButton
                    key={letter}
                    label={letter}
                    systemImage={course.expected_grade === letter ? 'checkmark' : undefined}
                    onPress={() => void chooseExpected(course, letter)}
                  />
                ))}
                <NativeButton
                  label="Open Course"
                  systemImage="arrow.up.forward.app"
                  onPress={() => router.push(`/course/${course.id}`)}
                />
              </Menu>
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
    </>
  );
}
