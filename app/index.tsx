import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import Animated, { FadeIn, useReducedMotion } from 'react-native-reanimated';

import { latestTranscript, listCourses, type TranscriptRow } from '../src/database/repo';
import { creditsValue, gpaValue, summarise, type AcademicSummary } from '../src/domain/academics';
import { Button, Divider, Notice, Screen, SectionHeader } from '../src/ui/components';
import { EASE_OUT, REVEAL_MS } from '../src/ui/motion';
import { PressableScale } from '../src/ui/press';
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

/**
 * Home answers one question: where am I academically right now.
 *
 * The numbers carry the screen, and everything below them is a row that opens the
 * place it came from. The transcript is the setup; this is what it bought.
 */
export default function Home() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const reduceMotion = useReducedMotion();
  const [summary, setSummary] = useState<AcademicSummary | null>(null);
  const [transcript, setTranscript] = useState<TranscriptRow | null>(null);
  // A read failure is not an empty library: keep what is on screen and say so.
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [courses, record] = await Promise.all([listCourses(db), latestTranscript(db)]);
      setSummary(summarise(courses));
      setTranscript(record);
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

  const entering = reduceMotion ? undefined : FadeIn.duration(REVEAL_MS).easing(EASE_OUT);
  const hasCourses = (summary?.count ?? 0) > 0;

  return (
    <Screen scroll contentStyle={{ paddingBottom: space.xxxl }}>
      <View
        style={{
          paddingHorizontal: space.xl,
          paddingTop: space.md,
          flexDirection: 'row',
          alignItems: 'center',
        }}
      >
        <Text style={[type.display, { color: palette.ink }]} accessibilityRole="header">
          Pace
        </Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Settings and privacy"
          onPress={() => router.push('/settings')}
          hitSlop={10}
          style={({ pressed }) => ({ marginLeft: 'auto', opacity: pressed ? 0.5 : 1, paddingVertical: space.sm })}
        >
          <Text style={[type.caption, { color: palette.inkMuted }]}>Settings</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Import a transcript"
          onPress={() => router.push('/scan')}
          hitSlop={10}
          style={({ pressed }) => ({
            marginLeft: space.lg,
            opacity: pressed ? 0.5 : 1,
            paddingVertical: space.sm,
          })}
        >
          <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>Import</Text>
        </Pressable>
      </View>

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg, gap: space.sm }}>
          <Notice tone="danger" title="Your data could not be loaded" body={error} />
          <Button title="Try again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {!error && !hasCourses ? (
        <View style={{ paddingHorizontal: space.xl, paddingTop: space.xxl, gap: space.lg }}>
          <Text style={[type.heading, { color: palette.ink }]}>No academics yet</Text>
          <Button title="Import Transcript" onPress={() => router.push('/scan')} />
        </View>
      ) : null}

      {hasCourses && summary ? (
        <Animated.View entering={entering}>
          <View
            style={{
              flexDirection: 'row',
              paddingHorizontal: space.xl,
              paddingTop: space.xl,
              paddingBottom: space.lg,
            }}
          >
            {transcript?.reported_gpa != null ? (
              <Metric value={gpaValue(transcript.reported_gpa)} label="Weighted · Reported" />
            ) : null}
            <Metric value={creditsValue(summary.earned)} label="Credits Earned" />
            <Metric value={creditsValue(summary.inProgress)} label="In Progress" />
          </View>

          {summary.current.length ? (
            <>
              <SectionHeader
                right={<Text style={[type.caption, { color: palette.inkFaint }]}>{summary.current.length}</Text>}
              >
                CURRENT
              </SectionHeader>
              <View style={{ paddingHorizontal: space.xl }}>
                {summary.current.map((course, index) => (
                  <View key={course.id}>
                    {index > 0 ? <Divider /> : null}
                    <PressableScale
                      accessibilityLabel={`Open ${course.title}`}
                      onPress={() => router.push(`/course/${course.id}`)}
                      style={{ paddingVertical: space.md }}
                    >
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                        <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                          {course.title}
                        </Text>
                        <Text style={[type.number, { color: palette.inkMuted }]}>
                          {creditsValue(course.credits)}
                        </Text>
                      </View>
                    </PressableScale>
                  </View>
                ))}
              </View>
            </>
          ) : null}

          {summary.terms.length ? (
            <>
              <SectionHeader
                right={
                  <Pressable
                    accessibilityRole="button"
                    onPress={() => router.push('/academics')}
                    hitSlop={8}
                    style={({ pressed }) => ({ opacity: pressed ? 0.5 : 1 })}
                  >
                    <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>See all</Text>
                  </Pressable>
                }
              >
                HISTORY
              </SectionHeader>
              <View style={{ paddingHorizontal: space.xl }}>
                {summary.terms.slice(0, 2).map((term, index) => (
                  <View key={term.label}>
                    {index > 0 ? <Divider /> : null}
                    <PressableScale
                      accessibilityLabel={`Open ${term.label}`}
                      onPress={() => router.push('/academics')}
                      style={{ paddingVertical: space.md }}
                    >
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                        <View style={{ flex: 1 }}>
                          <Text style={[type.body, { color: palette.ink }]}>{term.label}</Text>
                          <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                            {term.courses.length} {term.courses.length === 1 ? 'course' : 'courses'}
                          </Text>
                        </View>
                        <Text style={[type.number, { color: palette.inkMuted }]}>{creditsValue(term.credits)}</Text>
                      </View>
                    </PressableScale>
                  </View>
                ))}
              </View>
            </>
          ) : null}

          <View style={{ paddingHorizontal: space.xl, marginTop: space.lg }}>
            <Divider />
            <LinkRow label="Transcript" value={transcript?.student_name ?? null} onPress={() => router.push('/transcript')} />
            <Divider />
            <LinkRow label="Plan" onPress={() => router.push('/plan')} />
            <Divider />
          </View>
        </Animated.View>
      ) : null}
    </Screen>
  );
}

/** One number and its label: the number is 30pt, the label is quiet. */
function Metric({ value, label }: { value: string; label: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flex: 1, paddingRight: space.sm }}>
      <Text
        style={{
          fontSize: 30,
          lineHeight: 34,
          fontWeight: '600',
          letterSpacing: -0.8,
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

function LinkRow({ label, value, onPress }: { label: string; value?: string | null; onPress: () => void }) {
  const { palette } = useTheme();
  return (
    <PressableScale
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={{ paddingVertical: space.md, borderTopWidth: 0 }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
        <Text style={[type.body, { color: palette.ink, flex: 1 }]}>{label}</Text>
        {value ? (
          <Text style={[type.caption, { color: palette.inkMuted }]} numberOfLines={1}>
            {value}
          </Text>
        ) : null}
        <Text style={{ color: palette.inkFaint, fontSize: 18, marginTop: -2 }}>›</Text>
      </View>
    </PressableScale>
  );
}
