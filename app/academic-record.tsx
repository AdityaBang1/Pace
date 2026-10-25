import { Button as NativeButton, LabeledContent, Section } from '@expo/ui/swift-ui';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';

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
import { NativeScreen, Value } from '../src/ui/native';

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
  const tests = data?.structure.tests ?? [];

  return (
    <NativeScreen title="Record" onBack={() => router.back()}>
      {error ? (
        <Section title="Couldn’t load your record" footer={<Value>{error}</Value>}>
          <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void refresh()} />
        </Section>
      ) : null}

      {!error && record && record.courseCount === 0 ? (
        <Section title="No academics yet">
          <NativeButton label="Import Transcript" systemImage="doc.badge.plus" onPress={() => router.push('/scan')} />
        </Section>
      ) : null}

      {record && record.courseCount > 0 ? (
        <>
          <Section title="GPA">
            <LabeledContent label="Weighted">
              <Value>{gpaValue(record.gpa.weighted)}</Value>
            </LabeledContent>
            <LabeledContent label="Unweighted">
              <Value>{gpaValue(record.gpa.unweighted)}</Value>
            </LabeledContent>
            <LabeledContent label="Credits earned">
              <Value>{creditsValue(record.credits.earned)}</Value>
            </LabeledContent>
            <LabeledContent label="In progress">
              <Value>{creditsValue(record.credits.inProgress)}</Value>
            </LabeledContent>
          </Section>

          {reported != null ? (
            <Section title="Reported" footer={<Value>Printed on the transcript</Value>}>
              <LabeledContent label="Weighted">
                <Value>{gpaValue(reported)}</Value>
              </LabeledContent>
            </Section>
          ) : null}

          {!data?.scale ? (
            <Section title="Calculated" footer={<Value>Set a GPA scale to see calculated values.</Value>}>
              <NativeButton label="GPA Scale" systemImage="percent" onPress={() => router.push('/gpa-scale')} />
            </Section>
          ) : null}

          {record.years
            .filter((year) => year.courses.length > 0)
            .map((year) => {
              const gpa = computeGpa(data?.scale ?? null, year.courses);
              const totals = creditsOf(year.courses);
              const contributions = scoredCourses(year);
              return (
                <Section
                  key={year.year.id}
                  title={year.year.name}
                  footer={
                    <Value>
                      {`Weighted ${gpaValue(gpa.weighted)} · Unweighted ${gpaValue(gpa.unweighted)} · ${creditsValue(totals.earned)} earned`}
                    </Value>
                  }
                >
                  {year.terms.map((term) => {
                    const termCourses = year.courses.filter(
                      (course) => course.academic_term_id === term.id,
                    );
                    const calculated = computeGpa(data?.scale ?? null, termCourses);
                    const printed = printedTermGpas[term.name] ?? null;
                    return (
                      <LabeledContent key={term.id} label={termLabel(term.name)}>
                        <Value>
                          {printed != null
                            ? `Reported ${gpaValue(printed)}`
                            : `GPA ${gpaValue(calculated.weighted)}`}
                        </Value>
                      </LabeledContent>
                    );
                  })}
                  {contributions.map((course) => {
                    const mark = course.grade;
                    const points = data?.scale
                      ? pointsFor(data.scale, course.level ?? 'regular', mark)
                      : null;
                    const plain = data?.scale ? unweightedPoints(data.scale, mark) : null;
                    const weighted = points != null && plain != null && points !== plain;
                    return (
                      <LabeledContent key={course.id} label={course.title ?? course.code ?? 'Course'}>
                        <Value>
                          {`${mark ?? '—'} · ${points != null ? points.toFixed(2) : '—'}${weighted ? '*' : ''}`}
                        </Value>
                      </LabeledContent>
                    );
                  })}
                </Section>
              );
            })}

          <Section title="Scores" footer={<Value>* weight applied</Value>}>
            {tests.slice(0, 4).map((test) => (
              <LabeledContent key={test.id} label={test.name}>
                <Value>{test.total_score ?? '—'}</Value>
              </LabeledContent>
            ))}
            {tests.length ? null : <LabeledContent label="No scores"><Value>—</Value></LabeledContent>}
            <NativeButton label="All Scores" systemImage="list.number" onPress={() => router.push('/tests')} />
          </Section>
        </>
      ) : null}
    </NativeScreen>
  );
}
