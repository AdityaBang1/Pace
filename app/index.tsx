import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import Animated, { FadeIn, useReducedMotion } from 'react-native-reanimated';

import { courseCount, listCourses, listImports, type CourseRow, type ImportRow } from '../src/database/repo';
import { documentTypeLabel, relativeDay } from '../src/ui/format';
import { Button, Card, Divider, Notice, Screen, SectionHeader } from '../src/ui/components';
import { EASE_OUT, REVEAL_MS } from '../src/ui/motion';
import { PressableScale } from '../src/ui/press';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

export default function Home() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const reduceMotion = useReducedMotion();
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [imports, setImports] = useState<ImportRow[]>([]);
  const [count, setCount] = useState(0);
  const [loaded, setLoaded] = useState(false);
  // A read failure is not an empty library. Keeping the last good rows on screen
  // and saying what happened avoids telling the user their data is gone.
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextCourses, nextImports, nextCount] = await Promise.all([
        listCourses(db),
        listImports(db, 4),
        courseCount(db),
      ]);
      setCourses(nextCourses);
      setImports(nextImports);
      setCount(nextCount);
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

  // The list arrives on focus, which is tens of times a day: a short fade only,
  // never a slide, and it is gone entirely under reduced motion.
  const entering = reduceMotion ? undefined : FadeIn.duration(REVEAL_MS).easing(EASE_OUT);

  return (
    <Screen scroll contentStyle={{ paddingBottom: space.xxxl }}>
      <View style={{ paddingHorizontal: space.xl, paddingTop: space.md, flexDirection: 'row', alignItems: 'center' }}>
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
      </View>

      <View style={{ paddingHorizontal: space.xl, paddingTop: space.sm }}>
        <Text style={[type.body, { color: palette.inkMuted, maxWidth: 320 }]}>
          Turn the academic information you already have into something useful.
        </Text>
      </View>

      {/* The signature interaction: one large target, no competing primary actions. */}
      <PressableScale
        accessibilityLabel="Scan a document"
        onPress={() => router.push('/scan')}
        style={{
          margin: space.xl,
          marginBottom: space.sm,
          backgroundColor: palette.ink,
          borderRadius: radius.xl,
          padding: space.xl,
          minHeight: 132,
          justifyContent: 'space-between',
        }}
      >
        <View style={{ flexDirection: 'row', alignItems: 'flex-start' }}>
          <View style={{ flex: 1, paddingRight: space.md }}>
            <Text style={[type.heading, { color: palette.background }]}>Scan a document</Text>
            <Text style={[type.caption, { color: palette.background, opacity: 0.72, marginTop: 6 }]}>
              A syllabus or a course catalog page. Photograph it or choose a photo.
            </Text>
          </View>
          <View
            style={{
              width: 34,
              height: 34,
              borderRadius: 17,
              borderWidth: hairline,
              borderColor: palette.background,
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Text style={{ color: palette.background, fontSize: 18, marginTop: -2 }}>+</Text>
          </View>
        </View>
        <Text style={[type.caption, { color: palette.background, opacity: 0.55, marginTop: space.lg }]}>
          Read on this iPhone · nothing is uploaded
        </Text>
      </PressableScale>

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg, gap: space.sm }}>
          <Notice
            tone="danger"
            title="Your data could not be loaded"
            body={`${error} Nothing has been deleted: this is a read failure, not an empty library.`}
          />
          <Button title="Try again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {loaded && !error && count === 0 ? (
        <Animated.View entering={entering} style={{ paddingHorizontal: space.xl, paddingTop: space.lg }}>
          <Text style={[type.body, { color: palette.inkMuted }]}>
            Nothing imported yet. Pace reads the page, proposes the course, its grading breakdown, its meeting
            times and its policies, and waits for you to approve them.
          </Text>
        </Animated.View>
      ) : null}

      {imports.length ? (
        <>
          <SectionHeader
            right={
              <Pressable
                accessibilityRole="button"
                onPress={() => router.push('/academics')}
                hitSlop={8}
                style={({ pressed }) => ({ opacity: pressed ? 0.5 : 1 })}
              >
                <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>
                  {count} course{count === 1 ? '' : 's'}
                </Text>
              </Pressable>
            }
          >
            RECENT IMPORTS
          </SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {imports.map((row, index) => (
                <View key={row.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <PressableScale
                    disabled={!row.course_id || row.status === 'undone'}
                    onPress={() => row.course_id && router.push(`/course/${row.course_id}`)}
                    style={{ padding: space.lg }}
                  >
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <View style={{ flex: 1, paddingRight: space.md }}>
                        <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                          {row.course_title ?? 'Unnamed document'}
                        </Text>
                        <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                          {documentTypeLabel(row.document_type)} · {relativeDay(row.created_at)}
                          {row.status === 'undone' ? ' · undone' : ''}
                        </Text>
                      </View>
                      <Text style={{ color: palette.inkFaint, fontSize: 18 }}>›</Text>
                    </View>
                  </PressableScale>
                </View>
              ))}
            </Card>
          </View>
        </>
      ) : null}

      {courses.length ? (
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
            COURSES
          </SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {courses.slice(0, 3).map((course, index) => (
                <View key={course.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <PressableScale
                    onPress={() => router.push(`/course/${course.id}`)}
                    style={{ padding: space.lg }}
                  >
                    <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                      {course.title ?? course.code ?? 'Course'}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                      {[course.code, course.instructor].filter(Boolean).join(' · ') || 'No details yet'}
                    </Text>
                  </PressableScale>
                </View>
              ))}
            </Card>
          </View>
        </>
      ) : null}

      <View style={{ padding: space.xl, paddingTop: space.xxl }}>
        <Text style={[type.caption, { color: palette.inkFaint }]}>
          Pace reads documents on this device. There is no account, no server, and no upload.
        </Text>
      </View>
    </Screen>
  );
}
