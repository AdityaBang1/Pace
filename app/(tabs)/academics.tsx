import { Button as NativeButton, LabeledContent, Menu, Section, Text as NativeText } from '@expo/ui/swift-ui';
import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert } from 'react-native';

import {
  ensureStructure,
  readStructure,
  type AcademicStructure,
  type AcademicYearRow,
  type TermRow,
} from '../../src/database/academicRepo';
import { listCourses, readGpaScaleValue, type CourseRow } from '../../src/database/repo';
import { creditsValue, gpaValue } from '../../src/domain/academics';
import { parseScale, type GpaScale } from '../../src/domain/gpa';
import {
  buildRecord,
  statusLabel,
  statusOf,
  type ScheduleView,
  type YearView,
} from '../../src/domain/record';
import { NativeScreen, Value } from '../../src/ui/native';

const ALL_TERMS = 'all';

/**
 * The academic record, scoped to one year, one of its schedules and a term.
 *
 * Native sections replace the old tile grid: a student scanning their record wants
 * the mark against the course, not a card to rearrange. The scope stays a system
 * modal because picking a term should return to this list, not stack a screen.
 */
export default function Academics() {
  const db = useSQLiteContext();
  const router = useRouter();

  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [yearId, setYearId] = useState<string | null>(null);
  const [scheduleId, setScheduleId] = useState<string | null>(null);
  const [termId, setTermId] = useState<string>(ALL_TERMS);

  const refresh = useCallback(async () => {
    try {
      await ensureStructure(db);
      const [rows, nextStructure, scaleJson] = await Promise.all([
        listCourses(db),
        readStructure(db),
        readGpaScaleValue(db),
      ]);
      setCourses(rows);
      setStructure(nextStructure);
      setScale(parseScale(scaleJson));
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

  const record = useMemo(() => {
    if (!structure) return null;
    return buildRecord(scale, structure.years, structure.terms, structure.schedules, courses);
  }, [structure, scale, courses]);

  const year: YearView | null = useMemo(() => {
    if (!record) return null;
    return (
      record.years.find((view) => view.year.id === yearId) ??
      record.currentYear ??
      [...record.years].reverse().find((view) => view.year.status === 'past') ??
      record.years[0] ??
      null
    );
  }, [record, yearId]);

  const schedule: ScheduleView | null = useMemo(() => {
    if (!year) return null;
    return year.schedules.find((view) => view.schedule.id === scheduleId) ?? year.main ?? null;
  }, [year, scheduleId]);

  const term: TermRow | null = useMemo(() => {
    if (!year || termId === ALL_TERMS) return null;
    return year.terms.find((row) => row.id === termId) ?? null;
  }, [year, termId]);

  const visible = useMemo(() => {
    if (!schedule) return [];
    if (!term) return schedule.courses;
    return schedule.courses.filter((course) => course.academic_term_id === term.id);
  }, [schedule, term]);

  /** Current courses first, then one group per printed term, each keeping its order. */
  const groups = useMemo(() => {
    const current = visible.filter((course) => statusOf(course) === 'current');
    const finished = visible.filter((course) => statusOf(course) !== 'current');
    const byTerm = new Map<string, CourseRow[]>();
    for (const course of finished) {
      const key = course.academic_term_id ?? '';
      byTerm.set(key, [...(byTerm.get(key) ?? []), course]);
    }
    const ordered = [...byTerm.entries()].sort((left, right) => {
      const order = (id: string) =>
        year?.terms.findIndex((row) => row.id === id) ?? Number.MAX_SAFE_INTEGER;
      return order(left[0]) - order(right[0]);
    });
    const out: { key: string; title: string; courses: CourseRow[] }[] = [];
    if (current.length) out.push({ key: 'current', title: 'Current', courses: current });
    for (const [key, list] of ordered) {
      const label = year?.terms.find((row) => row.id === key)?.name ?? 'No term';
      out.push({ key: key || 'none', title: label, courses: list });
    }
    return out;
  }, [visible, year]);

  const addMenu = () => {
    if (!year || !schedule) return;
    Alert.alert('Add', undefined, [
      {
        text: 'Course',
        onPress: () =>
          router.push(
            `/editors/course?mode=add&scheduleId=${schedule.schedule.id}&yearId=${year.year.id}`,
          ),
      },
      { text: 'Schedule', onPress: () => router.push(`/editors/schedule?yearId=${year.year.id}`) },
      { text: 'Academic Year', onPress: () => router.push('/editors/year') },
      { text: 'Import Transcript', onPress: () => router.push('/scan') },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const addCourse = () => {
    if (year && schedule) {
      router.push(`/editors/course?mode=add&scheduleId=${schedule.schedule.id}&yearId=${year.year.id}`);
      return;
    }
    router.push('/editors/year');
  };

  /** A year owns its schedules and terms, so choosing one resets the two below it. */
  const pickYear = (next: AcademicYearRow) => {
    setYearId(next.id);
    const main = structure?.schedules.find(
      (row) => row.academic_year_id === next.id && row.is_main === 1,
    );
    setScheduleId(main?.id ?? null);
    setTermId(ALL_TERMS);
  };

  const gpa = schedule?.gpa.weighted ?? year?.gpa.weighted ?? null;
  const unweighted = schedule?.gpa.unweighted ?? year?.gpa.unweighted ?? null;
  const credits = schedule?.credits ?? year?.credits ?? { earned: 0, inProgress: 0, planned: 0, all: 0 };
  const yearSchedules = (structure?.schedules ?? []).filter(
    (row) => row.academic_year_id === year?.year.id,
  );

  return (
    <>
      <NativeScreen
        title="Academics"
        actions={[{ systemName: 'plus', label: 'Add', onPress: addMenu }]}
      >
        {error ? (
          <Section title="Couldn't load courses">
            <NativeText>{error}</NativeText>
            <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void refresh()} />
          </Section>
        ) : null}

        {!record || record.years.length === 0 ? (
          error ? null : (
            <Section title="No courses">
              <NativeText>Import a transcript, or start with a year.</NativeText>
              <NativeButton
                label="Import Transcript"
                systemImage="doc.badge.plus"
                onPress={() => router.push('/scan')}
              />
              <NativeButton
                label="Add Year"
                systemImage="calendar.badge.plus"
                onPress={() => router.push('/editors/year')}
              />
            </Section>
          )
        ) : (
          <>
            <Section
              title="Scope"
              footer={<NativeText>What the numbers below describe.</NativeText>}
            >
              <Menu label={year?.year.name ?? 'Year'} systemImage="calendar">
                {(structure?.years ?? []).map((row) => (
                  <NativeButton
                    key={row.id}
                    label={row.status === 'current' ? `${row.name} · Current` : row.name}
                    systemImage={row.id === year?.year.id ? 'checkmark' : undefined}
                    onPress={() => {
                      Haptics.selectionAsync().catch(() => {});
                      pickYear(row);
                    }}
                  />
                ))}
              </Menu>
              {year ? (
                <Menu
                  label={schedule?.schedule.name ?? 'No schedule'}
                  systemImage="calendar.badge.clock"
                >
                  {yearSchedules.map((row) => (
                    <NativeButton
                      key={row.id}
                      label={row.is_main === 1 ? `${row.name} · Main` : row.name}
                      systemImage={row.id === schedule?.schedule.id ? 'checkmark' : undefined}
                      onPress={() => setScheduleId(row.id)}
                    />
                  ))}
                </Menu>
              ) : null}
              {year && year.terms.length ? (
                <Menu label={term?.name ?? 'All terms'} systemImage="square.stack">
                  <NativeButton
                    label="All terms"
                    systemImage={termId === ALL_TERMS ? 'checkmark' : undefined}
                    onPress={() => setTermId(ALL_TERMS)}
                  />
                  {year.terms.map((row) => (
                    <NativeButton
                      key={row.id}
                      label={row.name}
                      systemImage={row.id === termId ? 'checkmark' : undefined}
                      onPress={() => setTermId(row.id)}
                    />
                  ))}
                </Menu>
              ) : null}
            </Section>

            <Section title="Record">
              <LabeledContent label="Weighted">
                <Value>{gpaValue(gpa)}</Value>
              </LabeledContent>
              {unweighted != null ? (
                <LabeledContent label="Unweighted">
                  <Value>{gpaValue(unweighted)}</Value>
                </LabeledContent>
              ) : null}
              <LabeledContent label="Earned">
                <Value>{creditsValue(credits.earned)}</Value>
              </LabeledContent>
              <LabeledContent label="In progress">
                <Value>{creditsValue(credits.inProgress)}</Value>
              </LabeledContent>
              {gpa == null ? (
                <NativeButton
                  label="Set GPA Scale"
                  systemImage="percent"
                  onPress={() => router.push('/gpa-scale')}
                />
              ) : null}
            </Section>

            <Section title="More">
              <NativeButton
                label="Record"
                systemImage="list.bullet.rectangle"
                onPress={() => router.push('/academic-record')}
              />
              <NativeButton
                label="Requirements"
                systemImage="checklist"
                onPress={() => router.push('/requirements')}
              />
              <NativeButton
                label="Scores"
                systemImage="list.number"
                onPress={() => router.push('/tests')}
              />
              <NativeButton
                label="Transcript"
                systemImage="doc.text"
                onPress={() => router.push('/transcript')}
              />
            </Section>

            {visible.length === 0 ? (
              <Section title={term ? term.name : 'No courses'}>
                <NativeText>No courses in this scope.</NativeText>
                <NativeButton label="Add Course" systemImage="plus" onPress={addCourse} />
              </Section>
            ) : (
              groups.map((group) => (
                <Section
                  key={group.key}
                  title={group.title}
                  footer={
                    <NativeText>
                      {`${group.courses.length} course${group.courses.length === 1 ? '' : 's'} · ${creditsValue(
                        Math.round(
                          group.courses.reduce((sum, course) => sum + (course.credits ?? 0), 0) * 100,
                        ) / 100,
                      )} credits`}
                    </NativeText>
                  }
                >
                  {group.courses.map((course) => (
                    <NativeButton
                      key={course.id}
                      label={`${course.title ?? course.code ?? 'Course'} · ${statusLabel(course)}`}
                      onPress={() => router.push(`/course/${course.id}`)}
                    />
                  ))}
                </Section>
              ))
            )}
          </>
        )}
      </NativeScreen>

    </>
  );
}
