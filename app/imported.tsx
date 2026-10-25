import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';

import { ensureStructure, readStructure, type AcademicStructure } from '../src/database/academicRepo';
import {
  latestTranscript,
  listCourses,
  termGpasOf,
  type CourseRow,
  type TranscriptRow,
} from '../src/database/repo';
import { creditsValue, gpaValue, termLabel } from '../src/domain/academics';
import { buildRecord, statusOf, type RecordView } from '../src/domain/record';
import { Button as NativeButton, Section, LabeledContent } from '@expo/ui/swift-ui';
import { NativeScreen, Value } from '../src/ui/native';

/**
 * What was just imported, shown once, before the record itself.
 *
 * The review screen showed the page's own rows and the counts it found; this is the
 * same transcript after the write: what Pace actually holds, what it made current,
 * and the way into the record. It is a preview, not a second confirmation — the
 * import is already done.
 */
export default function Imported() {
  const db = useSQLiteContext();
  const router = useRouter();
  const [record, setRecord] = useState<RecordView | null>(null);
  const [transcript, setTranscript] = useState<TranscriptRow | null>(null);
  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);

  const refresh = useCallback(async () => {
    await ensureStructure(db);
    const [rows, page, data] = await Promise.all([listCourses(db), latestTranscript(db), readStructure(db)]);
    setCourses(rows);
    setTranscript(page);
    setStructure(data);
    setRecord(buildRecord(null, data.years, data.terms, data.schedules, rows));
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const termGpas = termGpasOf(transcript);
  const current = courses.filter((course) => statusOf(course) === 'current');
  const terms = record?.years.flatMap((year) => year.terms) ?? [];
  const termsWithCourses = terms.filter((term) =>
    courses.some((course) => course.academic_term_id === term.id),
  );

  return (
    <NativeScreen title="Imported">
      <Section title="Summary">
        <LabeledContent label="Courses">
          <Value>{String(courses.length)}</Value>
        </LabeledContent>
        <LabeledContent label="Earned">
          <Value>{creditsValue(record?.credits.earned ?? 0)}</Value>
        </LabeledContent>
        <LabeledContent label="In progress">
          <Value>{creditsValue(record?.credits.inProgress ?? 0)}</Value>
        </LabeledContent>
        {transcript?.reported_gpa != null ? (
          <LabeledContent label="GPA · Reported">
            <Value>{gpaValue(transcript.reported_gpa)}</Value>
          </LabeledContent>
        ) : null}
      </Section>

      {current.length ? (
        <Section title="Current">
          {current.map((course) => (
            <LabeledContent key={course.id} label={course.title ?? course.code ?? 'Course'}>
              <Value>{`${creditsValue(course.credits)} cr`}</Value>
            </LabeledContent>
          ))}
        </Section>
      ) : null}

      {termsWithCourses.length ? (
        <Section title="Terms">
          {termsWithCourses.map((term) => {
            const held = courses.filter((course) => course.academic_term_id === term.id);
            const printed = termGpas[term.name];
            return (
              <LabeledContent key={term.id} label={termLabel(term.name)}>
                <Value>
                  {`${held.length} course${held.length === 1 ? '' : 's'}${printed != null ? ` · ${gpaValue(printed)}` : ''}`}
                </Value>
              </LabeledContent>
            );
          })}
        </Section>
      ) : null}

      {structure?.years.length ? (
        <Section title="Years">
          {structure.years.map((year) => (
            <LabeledContent key={year.id} label={year.name}>
              <Value>{year.status}</Value>
            </LabeledContent>
          ))}
        </Section>
      ) : null}

      <Section>
        <NativeButton
          label="Open Academics"
          systemImage="books.vertical.fill"
          onPress={() => router.replace('/academics')}
        />
        <NativeButton label="Home" systemImage="house.fill" onPress={() => router.replace('/')} />
      </Section>
    </NativeScreen>
  );
}
