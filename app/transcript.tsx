import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';

import { latestTranscript, listCourses, type CourseRow, type TranscriptRow } from '../src/database/repo';
import { creditsValue, gpaValue, summarise } from '../src/domain/academics';
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
  const { palette } = useTheme();
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

  const metrics: { value: string; label: string }[] = [];
  if (gpa != null) metrics.push({ value: gpaValue(gpa), label: 'Weighted GPA' });
  metrics.push({ value: creditsValue(earned), label: 'Earned' });
  metrics.push({ value: creditsValue(inProgress), label: 'In Progress' });

  const empty = loaded && !error && !transcript && courses.length === 0;

  return (
    <Screen scroll>
      <PageHeader title="Transcript" onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice tone="danger" title="Couldn’t load your transcript" body={error} />
          <Button title="Try again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {empty ? (
        <EmptyState
          title="Nothing imported"
          body="Import a transcript to see it here."
          action={<Button title="Import" onPress={() => router.push('/scan')} />}
        />
      ) : null}

      {!empty && (transcript || courses.length) ? (
        <>
          <View style={{ paddingHorizontal: space.xl }}>
            <Text style={[type.title, { color: palette.ink }]} numberOfLines={2}>
              {student ?? transcript?.school ?? 'Transcript'}
            </Text>
            {school && student ? (
              <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                {school}
              </Text>
            ) : null}
          </View>

          <View
            style={{
              flexDirection: 'row',
              gap: space.xl,
              paddingHorizontal: space.xl,
              paddingTop: space.xl,
            }}
          >
            {metrics.map((metric) => (
              <View key={metric.label} style={{ flex: 1 }}>
                <Text style={[type.title, { color: palette.ink, fontVariant: ['tabular-nums'] }]}>
                  {metric.value}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                  {metric.label}
                </Text>
              </View>
            ))}
          </View>

          <View style={{ paddingHorizontal: space.xl, paddingTop: space.md }}>
            <Text style={[type.caption, { color: palette.inkFaint }]}>
              {courseCount} course{courseCount === 1 ? '' : 's'} · {termCount} term
              {termCount === 1 ? '' : 's'}
            </Text>
          </View>

          {termCount ? (
            <>
              <SectionHeader>HISTORY</SectionHeader>
              <View style={{ paddingHorizontal: space.xl }}>
                <Card padded={false}>
                  {summary.terms.map((term, index) => (
                    <View key={term.term || `term-${index}`}>
                      {index > 0 ? <Divider inset={space.lg} /> : null}
                      <View
                        style={{
                          flexDirection: 'row',
                          alignItems: 'center',
                          gap: space.md,
                          padding: space.lg,
                        }}
                      >
                        <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                          {term.label}
                        </Text>
                        <Text style={[type.caption, { color: palette.inkFaint }]}>
                          {term.courses.length} course{term.courses.length === 1 ? '' : 's'}
                        </Text>
                        <Text style={[type.number, { color: palette.ink }]}>{creditsValue(term.credits)}</Text>
                      </View>
                    </View>
                  ))}
                </Card>
              </View>
            </>
          ) : null}

          <SectionHeader>SOURCE</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md, padding: space.lg }}>
                <Text style={[type.body, { color: palette.inkMuted }]}>Source</Text>
                <Text
                  style={[type.body, { color: palette.ink, flex: 1, textAlign: 'right' }]}
                  numberOfLines={2}
                >
                  {school ?? student ?? 'Transcript'}
                </Text>
              </View>
            </Card>
          </View>
        </>
      ) : null}
    </Screen>
  );
}
