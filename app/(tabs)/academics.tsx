import { Button as NativeButton, LabeledContent, Section, Text as NativeText } from '@expo/ui/swift-ui';
import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert, Modal, Pressable, ScrollView, Text, View } from 'react-native';

import {
  ensureStructure,
  readStructure,
  type AcademicStructure,
  type AcademicYearRow,
  type ScheduleRow,
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
import { IconButton } from '../../src/ui/components';
import { NativeScreen, Value } from '../../src/ui/native';
import { PressableScale } from '../../src/ui/press';
import { radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

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
  const { palette } = useTheme();

  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [yearId, setYearId] = useState<string | null>(null);
  const [scheduleId, setScheduleId] = useState<string | null>(null);
  const [termId, setTermId] = useState<string>(ALL_TERMS);
  const [scopeOpen, setScopeOpen] = useState(false);

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

  const gpa = schedule?.gpa.weighted ?? year?.gpa.weighted ?? null;
  const unweighted = schedule?.gpa.unweighted ?? year?.gpa.unweighted ?? null;
  const credits = schedule?.credits ?? year?.credits ?? { earned: 0, inProgress: 0, planned: 0, all: 0 };
  const scopeLabel = [year?.year.name ?? 'Year', schedule?.schedule.name ?? 'Schedule', term?.name]
    .filter(Boolean)
    .join(' · ');

  return (
    <>
      <NativeScreen
        topInset={false}
        title="Academics"
        right={
          <IconButton glyph="+" symbol="plus" label="Add" size={22} onPress={addMenu} tone="accent" />
        }
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
            <Section title="Scope">
              <NativeButton
                label={scopeLabel}
                systemImage="chevron.up.chevron.down"
                onPress={() => {
                  Haptics.selectionAsync().catch(() => {});
                  setScopeOpen(true);
                }}
              />
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

      <ScopeSheet
        open={scopeOpen}
        structure={structure}
        year={year?.year ?? null}
        schedule={schedule?.schedule ?? null}
        termId={termId}
        onClose={() => setScopeOpen(false)}
        onPickYear={(next) => {
          setYearId(next.id);
          const main = structure?.schedules.find(
            (row) => row.academic_year_id === next.id && row.is_main === 1,
          );
          setScheduleId(main?.id ?? null);
          setTermId(ALL_TERMS);
        }}
        onPickSchedule={setScheduleId}
        onPickTerm={setTermId}
      />
    </>
  );
}

/**
 * The scope picker: year, then that year's schedules, then its terms.
 *
 * A native modal, because it is a choice, not a destination: picking a term should
 * return the list, not stack another screen under the tab bar.
 */
function ScopeSheet({
  open,
  structure,
  year,
  schedule,
  termId,
  onClose,
  onPickYear,
  onPickSchedule,
  onPickTerm,
}: {
  open: boolean;
  structure: AcademicStructure | null;
  year: AcademicYearRow | null;
  schedule: ScheduleRow | null;
  termId: string;
  onClose: () => void;
  onPickYear: (year: AcademicYearRow) => void;
  onPickSchedule: (id: string) => void;
  onPickTerm: (id: string) => void;
}) {
  const { palette } = useTheme();
  const years = structure?.years ?? [];
  const schedules = (structure?.schedules ?? []).filter(
    (row) => row.academic_year_id === year?.id,
  );
  const terms = (structure?.terms ?? []).filter((row) => row.academic_year_id === year?.id);

  const row = (label: string, selected: boolean, onPress: () => void, sub?: string) => (
    <PressableScale key={`${label}-${sub ?? ''}`} accessibilityRole="button" onPress={onPress}>
      <View
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          paddingVertical: space.md,
          paddingHorizontal: space.xl,
          gap: space.md,
        }}
      >
        <Text
          style={[type.body, { color: selected ? palette.ink : palette.inkMuted, flex: 1 }]}
          numberOfLines={1}
        >
          {label}
        </Text>
        {sub ? <Text style={[type.caption, { color: palette.inkFaint }]}>{sub}</Text> : null}
        {selected ? (
          <Text style={{ color: palette.accent, fontSize: 16 }}>✓</Text>
        ) : null}
      </View>
    </PressableScale>
  );

  return (
    <Modal visible={open} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={{ flex: 1, backgroundColor: palette.overlay }} onPress={onClose} />
      <View
        style={{
          backgroundColor: palette.background,
          borderTopLeftRadius: radius.xl,
          borderTopRightRadius: radius.xl,
          maxHeight: '70%',
          paddingBottom: space.xxl,
        }}
      >
        <ScrollView keyboardShouldPersistTaps="handled">
          <Text
            style={[
              type.label,
              { color: palette.inkFaint, paddingHorizontal: space.xl, paddingTop: space.lg },
            ]}
          >
            Academic Year
          </Text>
          {years.map((item) =>
            row(
              item.name,
              item.id === year?.id,
              () => {
                onPickYear(item);
              },
              item.status,
            ),
          )}

          {schedules.length ? (
            <>
              <Text
                style={[
                  type.label,
                  { color: palette.inkFaint, paddingHorizontal: space.xl, paddingTop: space.lg },
                ]}
              >
                Schedule
              </Text>
              {schedules.map((item) =>
                row(
                  item.name,
                  item.id === schedule?.id,
                  () => onPickSchedule(item.id),
                  item.is_main === 1 ? 'Main' : undefined,
                ),
              )}
            </>
          ) : null}

          {terms.length ? (
            <>
              <Text
                style={[
                  type.label,
                  { color: palette.inkFaint, paddingHorizontal: space.xl, paddingTop: space.lg },
                ]}
              >
                Term
              </Text>
              {row('All', termId === ALL_TERMS, () => onPickTerm(ALL_TERMS))}
              {terms.map((item) => row(item.name, item.id === termId, () => onPickTerm(item.id)))}
            </>
          ) : null}
        </ScrollView>
      </View>
    </Modal>
  );
}
