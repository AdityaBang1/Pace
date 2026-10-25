import { Button as NativeButton, LabeledContent, Section } from '@expo/ui/swift-ui';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';

import { latestTranscript, listCourses, type CourseRow, type TranscriptRow } from '../src/database/repo';
import { creditsValue, gpaValue, summarise } from '../src/domain/academics';
import { NativeScreen, Value } from '../src/ui/native';

/**
 * The transcript itself, as an object worth opening.
 *
 * The courses it created have their own screens; this one answers "what did Pace
 * import, and what did the page itself state?" A printed GPA is labelled as
 * reported, never as one Pace calculated, and a page that printed none simply has
 * no GPA here rather than a dash.
 */
export default function Transcript() {
  const db = useSQLiteContext();
  const router = useRouter();
  const [transcript, setTranscript] = useState<TranscriptRow | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextTranscript, nextCourses] = await Promise.all([latestTranscript(db), listCourses(db)]);
      setTranscript(nextTranscript);
      setCourses(nextCourses);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    } finally {
      setLoaded(true);
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const summary = summarise(courses);
  const student = transcript?.student_name ?? null;
  const school = transcript?.school ?? null;
  const gpa = transcript?.reported_gpa ?? null;
  // What the import recorded is what the page stated; the courses are the same
  // record, so either source is honest, and the stored one is preferred.
  const earned = transcript?.earned_credits ?? summary.earned;
  const inProgress = transcript?.in_progress_credits ?? summary.inProgress;
  const courseCount = transcript?.course_count ?? summary.count;
  const termCount = summary.terms.length;

  const empty = loaded && !error && !transcript && courses.length === 0;

  return (
    <NativeScreen title="Transcript" onBack={() => router.back()}>
      {error ? (
        <Section title="Couldn’t load your transcript" footer={<Value>{error}</Value>}>
          <NativeButton label="Try again" systemImage="arrow.clockwise" onPress={() => void refresh()} />
        </Section>
      ) : null}

      {empty ? (
        <Section title="Nothing imported">
          <NativeButton label="Import" systemImage="doc.badge.plus" onPress={() => router.push('/scan')} />
        </Section>
      ) : null}

      {!empty && (transcript || courses.length) ? (
        <>
          <Section title="Student">
            <LabeledContent label={student ? 'Name' : 'Document'}>
              <Value>{student ?? school ?? 'Transcript'}</Value>
            </LabeledContent>
            {school && student ? (
              <LabeledContent label="School">
                <Value>{school}</Value>
              </LabeledContent>
            ) : null}
          </Section>

          <Section title="Summary">
            {gpa != null ? (
              <LabeledContent label="Weighted GPA">
                <Value>{gpaValue(gpa)}</Value>
              </LabeledContent>
            ) : null}
            <LabeledContent label="Earned">
              <Value>{creditsValue(earned)}</Value>
            </LabeledContent>
            <LabeledContent label="In progress">
              <Value>{creditsValue(inProgress)}</Value>
            </LabeledContent>
          </Section>

          {termCount ? (
            <Section
              title="History"
              footer={
                <Value>{`${courseCount} course${courseCount === 1 ? '' : 's'} · ${termCount} term${termCount === 1 ? '' : 's'}`}</Value>
              }
            >
              {summary.terms.map((term, index) => (
                <LabeledContent
                  key={term.term || `term-${index}`}
                  label={`${term.label} · ${term.courses.length}`}
                >
                  <Value>{creditsValue(term.credits)}</Value>
                </LabeledContent>
              ))}
            </Section>
          ) : null}

          <Section title="Source">
            <LabeledContent label="Document">
              <Value>{school ?? student ?? 'Transcript'}</Value>
            </LabeledContent>
          </Section>
        </>
      ) : null}
    </NativeScreen>
  );
}
