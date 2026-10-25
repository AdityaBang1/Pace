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
import { parseScale, levelLabel, type GpaScale } from '../../src/domain/gpa';
import { buildRecord, statusLabel, type ScheduleView, type YearView } from '../../src/domain/record';
import { Button, EmptyState, IconButton, Notice, PageHeader, Screen } from '../../src/ui/components';
import { PressableScale } from '../../src/ui/press';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

const ALL_TERMS = 'all';

/**
 * The academic workspace.
 *
 * One scope (a year, one of its schedules, a term) and the courses inside it, as
 * tiles a person can reorganise and edit. Everything else — the record, requirements,
 * scores — is one tap away, because the grid is what a student comes here for.
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

  const termName = (course: CourseRow): string | null => {
    if (!course.academic_term_id) return null;
    return structure?.terms.find((row) => row.id === course.academic_term_id)?.name ?? null;
  };

  const gpa = schedule?.gpa.weighted ?? year?.gpa.weighted ?? null;
  const unweighted = schedule?.gpa.unweighted ?? year?.gpa.unweighted ?? null;
  const credits = schedule?.credits ?? year?.credits ?? { earned: 0, inProgress: 0, planned: 0, all: 0 };

  return (
    <Screen edges="top" scroll>
      <PageHeader
        title="Academics"
        right={
          <IconButton glyph="+" label="Add" size={26} onPress={addMenu} tone="accent" />
        }
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice tone="danger" title="Couldn't load courses" body={error} />
          <Button title="Try Again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {!record || record.years.length === 0 ? (
        error ? null : (
          <EmptyState
            title="No courses"
            body="Import a transcript, or add a course."
            action={
              <View style={{ flexDirection: 'row', gap: space.sm }}>
                <Button title="Import" onPress={() => router.push('/scan')} />
                <Button title="Add Course" kind="quiet" onPress={() => router.push('/editors/year')} />
              </View>
            }
          />
        )
      ) : (
        <>
          <PressableScale
            accessibilityRole="button"
            accessibilityLabel="Change year, schedule and term"
            onPress={() => {
              Haptics.selectionAsync().catch(() => {});
              setScopeOpen(true);
            }}
            style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}
          >
            <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: space.sm }}>
              <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                {year?.year.name ?? 'Year'}
              </Text>
              <Text style={[type.caption, { color: palette.inkMuted, flexShrink: 1 }]} numberOfLines={1}>
                {schedule?.schedule.name ?? 'Schedule'}
                {term ? ` · ${term.name}` : ''}
              </Text>
              <Text style={{ color: palette.inkFaint, fontSize: 13 }}>⌄</Text>
            </View>
          </PressableScale>

          <View
            style={{
              flexDirection: 'row',
              paddingHorizontal: space.xl,
              paddingBottom: space.lg,
              borderBottomWidth: hairline,
              borderBottomColor: palette.line,
              gap: space.xl,
            }}
          >
            <Metric
              value={gpaValue(gpa)}
              label={unweighted != null ? 'Weighted · Calculated' : 'Calculated GPA'}
            />
            {unweighted != null ? <Metric value={gpaValue(unweighted)} label="Unweighted" /> : null}
            <Metric value={creditsValue(credits.earned)} label="Earned" />
            <Metric value={creditsValue(credits.inProgress)} label="In Progress" />
          </View>

          {gpa == null ? (
            <View style={{ paddingHorizontal: space.xl, paddingTop: space.md }}>
              <PressableScale
                accessibilityRole="button"
                onPress={() => router.push('/gpa-scale')}
              >
                <Text style={[type.caption, { color: palette.accent }]}>Set GPA Scale</Text>
              </PressableScale>
            </View>
          ) : null}

          <View
            style={{
              flexDirection: 'row',
              paddingHorizontal: space.xl,
              paddingTop: space.lg,
              gap: space.lg,
            }}
          >
            <Text
              accessibilityRole="button"
              onPress={() => router.push('/academic-record')}
              style={[type.caption, { color: palette.accent }]}
            >
              Record
            </Text>
            <Text
              accessibilityRole="button"
              onPress={() => router.push('/requirements')}
              style={[type.caption, { color: palette.accent }]}
            >
              Requirements
            </Text>
            <Text
              accessibilityRole="button"
              onPress={() => router.push('/tests')}
              style={[type.caption, { color: palette.accent }]}
            >
              Tests
            </Text>
          </View>

          {visible.length === 0 ? (
            <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl, gap: space.sm }}>
              <Text style={[type.body, { color: palette.inkMuted }]}>No courses</Text>
              <Button
                title="Add Course"
                kind="quiet"
                compact
                onPress={() =>
                  year && schedule
                    ? router.push(
                        `/editors/course?mode=add&scheduleId=${schedule.schedule.id}&yearId=${year.year.id}`,
                      )
                    : router.push('/editors/year')
                }
              />
            </View>
          ) : (
            <View
              style={{
                flexDirection: 'row',
                flexWrap: 'wrap',
                gap: space.md,
                paddingHorizontal: space.xl,
                paddingTop: space.lg,
              }}
            >
              {visible.map((course) => (
                <PressableScale
                  key={course.id}
                  accessibilityLabel={`${course.title ?? 'Course'}, ${statusLabel(course)}`}
                  onPress={() => router.push(`/course/${course.id}`)}
                  onLongPress={() => router.push(`/editors/course?id=${course.id}`)}
                  style={{ flexBasis: '48%', flexGrow: 1 }}
                >
                  <View
                    style={{
                      backgroundColor: palette.surface,
                      borderRadius: radius.lg,
                      borderWidth: hairline,
                      borderColor: palette.line,
                      padding: space.md,
                      minHeight: 118,
                      justifyContent: 'space-between',
                    }}
                  >
                    <Text
                      style={[type.bodyStrong, { color: palette.ink }]}
                      numberOfLines={2}
                    >
                      {course.title ?? course.code ?? 'Course'}
                    </Text>
                    <Text
                      style={[
                        type.number,
                        { color: palette.ink, marginTop: space.sm, fontSize: 19 },
                      ]}
                      numberOfLines={1}
                    >
                      {statusLabel(course)}
                    </Text>
                    <Text
                      style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}
                      numberOfLines={1}
                    >
                      {[
                        levelLabel(course.level as 'regular' | 'honors' | 'ap'),
                        `${creditsValue(course.credits ?? 0)} cr`,
                        termName(course),
                      ]
                        .filter(Boolean)
                        .join(' · ')}
                    </Text>
                  </View>
                </PressableScale>
              ))}
            </View>
          )}
        </>
      )}

      <ScopeSheet
        open={scopeOpen}
        structure={structure}
        year={year?.year ?? null}
        schedule={schedule?.schedule ?? null}
        termId={termId}
        onClose={() => setScopeOpen(false)}
        onPickYear={(next) => {
          setYearId(next.id);
          const main = structure?.schedules.find((row) => row.academic_year_id === next.id && row.is_main === 1);
          setScheduleId(main?.id ?? null);
          setTermId(ALL_TERMS);
        }}
        onPickSchedule={setScheduleId}
        onPickTerm={setTermId}
      />
    </Screen>
  );
}

/** One number and its label. The number carries the weight; the label is quiet. */
function Metric({ value, label }: { value: string; label: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flexShrink: 1 }}>
      <Text
        style={{
          fontSize: 26,
          lineHeight: 30,
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

/**
 * The scope picker: year, then that year's schedules, then its terms.
 *
 * A native modal, because it is a choice, not a destination: picking a term should
 * return the grid, not stack another screen under the tab bar.
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
