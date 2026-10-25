import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Text, View } from 'react-native';

import { ensureStructure, readStructure, type AcademicStructure } from '../src/database/academicRepo';
import {
  latestTranscript,
  listCourses,
  readGpaScaleValue,
  type CourseRow,
  type TranscriptRow,
} from '../src/database/repo';
import { creditsValue, gpaValue, termLabel } from '../src/domain/academics';
import { parseScale, pointsFor, unweightedPoints, type GpaScale } from '../src/domain/gpa';
import { buildRecord, computeGpa, creditsOf, statusOf, type YearView } from '../src/domain/record';
import {
  Button,
  Card,
  Divider,
  EmptyState,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../src/ui/components';
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

interface Loaded {
  structure: AcademicStructure;
  courses: CourseRow[];
  scale: GpaScale | null;
  transcript: TranscriptRow | null;
}

/**
 * The record, in full: the GPA Pace can justify, how each year and term adds up to
 * it, and the scores that are not part of it.
 *
 * Every number here comes from the scale the user set. Where the transcript printed
 * a GPA of its own it is shown beside the calculated one and labelled Reported, so
 * the two can never be mistaken for each other.
 */
export default function AcademicRecord() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      await ensureStructure(db);
      const [structure, courses, scaleJson, transcript] = await Promise.all([
        readStructure(db),
        listCourses(db),
        readGpaScaleValue(db),
        latestTranscript(db),
      ]);
      setData({ structure, courses, scale: parseScale(scaleJson), transcript });
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      let live = true;
      void (async () => {
        if (!live) return;
        await refresh();
      })();
      return () => {
        live = false;
      };
    }, [refresh]),
  );

  const record = useMemo(
    () =>
      data
        ? buildRecord(data.scale, data.structure.years, data.structure.terms, data.structure.schedules, data.courses)
        : null,
    [data],
  );

  const printedTermGpas = useMemo(() => {
    if (!data?.transcript?.term_gpas) return {} as Record<string, number>;
    try {
      const parsed: unknown = JSON.parse(data.transcript.term_gpas);
      if (!parsed || typeof parsed !== 'object') return {};
      const out: Record<string, number> = {};
      for (const [term, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof value === 'number' && Number.isFinite(value)) out[term] = value;
      }
      return out;
    } catch {
      return {};
    }
  }, [data]);

  const scoredCourses = (year: YearView) =>
    year.courses.filter((course) => {
      if (course.include_in_gpa === 0) return false;
      if (statusOf(course) !== 'completed') return false;
      return Boolean(course.grade);
    });

  const reported = data?.transcript?.reported_gpa ?? null;

  return (
    <Screen scroll>
      <PageHeader title="Record" onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice tone="danger" title="Couldn’t load your record" body={error} />
          <Button title="Try Again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {!error && record && record.courseCount === 0 ? (
        <EmptyState
          title="No academics yet"
          body="Import a transcript."
          action={<Button title="Import Transcript" onPress={() => router.push('/scan')} />}
        />
      ) : null}

      {record && record.courseCount > 0 ? (
        <>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: space.xl, paddingTop: space.md }}>
            {[
              { value: gpaValue(record.gpa.weighted), label: 'Calculated' },
              { value: gpaValue(record.gpa.unweighted), label: 'Unweighted' },
              { value: creditsValue(record.credits.earned), label: 'Earned' },
              { value: creditsValue(record.credits.inProgress), label: 'In Progress' },
            ].map((metric) => (
              <View key={metric.label} style={{ width: '50%', marginBottom: space.lg }}>
                <Text style={[type.title, { color: palette.ink, fontVariant: ['tabular-nums'] }]}>
                  {metric.value}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                  {metric.label}
                </Text>
              </View>
            ))}
          </View>

          {reported != null ? (
            <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
              <Card>
                <Text style={[type.caption, { color: palette.inkMuted }]}>Reported</Text>
                <Text style={[type.title, { color: palette.ink, fontVariant: ['tabular-nums'] }]}>
                  {gpaValue(reported)}
                </Text>
                <Text style={[type.caption, { color: palette.inkFaint, marginTop: 2 }]}>
                  Weighted · printed on the transcript
                </Text>
              </Card>
            </View>
          ) : null}

          {!data?.scale ? (
            <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
              <Notice tone="accent" body="Set a GPA scale to see calculated values." />
            </View>
          ) : null}

          <SectionHeader>GPA BREAKDOWN</SectionHeader>
          {record.years
            .filter((year) => year.courses.length > 0)
            .map((year) => {
              const gpa = computeGpa(data?.scale ?? null, year.courses);
              const totals = creditsOf(year.courses);
              const contributions = scoredCourses(year);
              return (
                <View key={year.year.id} style={{ marginBottom: space.lg }}>
                  <View
                    style={{
                      flexDirection: 'row',
                      alignItems: 'baseline',
                      paddingHorizontal: space.xl,
                      paddingBottom: space.xs,
                    }}
                  >
                    <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                      {year.year.name}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted }]}>
                      {gpaValue(gpa.weighted)} · {creditsValue(totals.earned)} cr
                    </Text>
                  </View>
                  <Divider />
                  <View style={{ paddingHorizontal: space.xl, paddingVertical: space.sm }}>
                    <Text style={[type.caption, { color: palette.inkFaint }]}>
                      Weighted {gpaValue(gpa.weighted)} · Unweighted {gpaValue(gpa.unweighted)} ·{' '}
                      {creditsValue(totals.earned)} earned
                      {totals.inProgress ? ` · ${creditsValue(totals.inProgress)} in progress` : ''}
                    </Text>
                  </View>

                  {year.terms.length ? (
                    <View style={{ paddingHorizontal: space.xl, paddingBottom: space.sm }}>
                      {year.terms.map((term) => {
                        const termCourses = year.courses.filter(
                          (course) => course.academic_term_id === term.id,
                        );
                        const calculated = computeGpa(data?.scale ?? null, termCourses);
                        const printed = printedTermGpas[term.name] ?? null;
                        return (
                          <View
                            key={term.id}
                            style={{ flexDirection: 'row', alignItems: 'baseline', paddingVertical: 3 }}
                          >
                            <Text style={[type.caption, { color: palette.inkMuted, flex: 1 }]} numberOfLines={1}>
                              {termLabel(term.name)}
                            </Text>
                            <Text style={[type.caption, { color: palette.inkFaint }]}>
                              {printed != null ? `Reported ${gpaValue(printed)}` : `GPA ${gpaValue(calculated.weighted)}`}
                            </Text>
                          </View>
                        );
                      })}
                    </View>
                  ) : null}

                  {contributions.length ? (
                    <View style={{ paddingHorizontal: space.xl }}>
                      <Card padded={false}>
                        {contributions.map((course, index) => {
                          const mark = course.grade;
                          const points = data?.scale
                            ? pointsFor(data.scale, course.level ?? 'regular', mark)
                            : null;
                          const plain = data?.scale ? unweightedPoints(data.scale, mark) : null;
                          return (
                            <View key={course.id}>
                              {index > 0 ? <Divider inset={space.md} /> : null}
                              <View
                                style={{
                                  flexDirection: 'row',
                                  alignItems: 'center',
                                  gap: space.md,
                                  paddingHorizontal: space.md,
                                  paddingVertical: space.sm,
                                }}
                              >
                                <Text
                                  style={[type.caption, { color: palette.ink, flex: 1 }]}
                                  numberOfLines={1}
                                >
                                  {course.title ?? course.code ?? 'Course'}
                                </Text>
                                <Text style={[type.caption, { color: palette.inkMuted }]}>{mark ?? '—'}</Text>
                                <Text
                                  style={[
                                    type.caption,
                                    { color: palette.inkFaint, width: 42, textAlign: 'right' },
                                  ]}
                                >
                                  {points != null ? points.toFixed(2) : '—'}
                                  {points != null && plain != null && points !== plain ? '*' : ''}
                                </Text>
                              </View>
                            </View>
                          );
                        })}
                      </Card>
                      <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.xs }]}>
                        * weight applied
                      </Text>
                    </View>
                  ) : null}
                </View>
              );
            })}

          <SectionHeader right={<Text style={[type.caption, { color: palette.accent }]}>All</Text>}>
            SCORES
          </SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {data?.structure.tests.length ? (
                data.structure.tests.slice(0, 4).map((test, index) => (
                  <View key={test.id}>
                    {index > 0 ? <Divider inset={space.lg} /> : null}
                    <View
                      style={{
                        flexDirection: 'row',
                        alignItems: 'center',
                        gap: space.md,
                        padding: space.lg,
                      }}
                    >
                      <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                        {test.name}
                      </Text>
                      <Text style={[type.caption, { color: palette.inkMuted }]} numberOfLines={1}>
                        {test.taken_on ?? ''}
                      </Text>
                      <Text style={[type.number, { color: palette.ink }]}>{test.total_score ?? '—'}</Text>
                    </View>
                  </View>
                ))
              ) : (
                <View style={{ padding: space.lg }}>
                  <Text style={[type.caption, { color: palette.inkMuted }]}>No scores</Text>
                </View>
              )}
            </Card>
            <View style={{ paddingTop: space.sm }}>
              <Button title="Scores" kind="quiet" compact onPress={() => router.push('/tests')} />
            </View>
          </View>
        </>
      ) : null}
    </Screen>
  );
}
