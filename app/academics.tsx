import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Pressable, Text, View } from 'react-native';

import { listCourses, type CourseRow } from '../src/database/repo';
import { creditsValue, summarise, type HistoryCourse } from '../src/domain/academics';
import { Button, Divider, EmptyState, Notice, PageHeader, Screen } from '../src/ui/components';
import { PressableScale } from '../src/ui/press';
import { hairline, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

type Tab = 'current' | 'history';

export default function Academics() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('history');

  const refresh = useCallback(async () => {
    try {
      setCourses(await listCourses(db));
      setError(null);
    } catch (cause) {
      // An unreadable database is not "no courses": keep the rows already shown
      // and surface the failure with a way to try again.
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const summary = useMemo(() => summarise(courses), [courses]);
  const hasCourses = courses.length > 0;

  const openCourse = (id: string) => router.push(`/course/${id}`);

  const courseRow = (course: HistoryCourse, last: boolean, showMark: boolean) => (
    <View key={course.id}>
      <PressableScale
        accessibilityLabel={`Open ${course.title}`}
        onPress={() => openCourse(course.id)}
        style={{ paddingVertical: space.md, paddingHorizontal: space.xl }}
      >
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ flex: 1, paddingRight: space.md }}>
            <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={2}>
              {course.title}
            </Text>
            {course.code ? (
              <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>{course.code}</Text>
            ) : null}
          </View>
          <Text style={[showMark ? type.number : type.body, { color: showMark ? palette.ink : palette.inkMuted }]}>
            {showMark ? course.mark ?? '—' : creditsValue(course.credits)}
          </Text>
        </View>
      </PressableScale>
      {last ? null : <Divider inset={space.xl} />}
    </View>
  );

  return (
    <Screen scroll>
      <PageHeader
        title="Academics"
        subtitle={
          hasCourses
            ? `${summary.count} course${summary.count === 1 ? '' : 's'} · ${creditsValue(summary.earned)} credits`
            : undefined
        }
        onBack={() => router.back()}
        right={
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Import a transcript"
            onPress={() => router.push('/scan')}
            hitSlop={10}
            style={({ pressed }) => ({ opacity: pressed ? 0.5 : 1, paddingVertical: space.sm })}
          >
            <Text style={[type.bodyStrong, { color: palette.accent }]}>Import</Text>
          </Pressable>
        }
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice
            tone="danger"
            title="Couldn't load your courses"
            body={error}
          />
          <Button title="Try again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {!hasCourses ? (
        error ? null : (
          <EmptyState
            title="No courses yet"
            body="Import a transcript to build your record."
            action={<Button title="Import Transcript" onPress={() => router.push('/scan')} />}
          />
        )
      ) : (
        <>
          <View
            style={{
              flexDirection: 'row',
              gap: space.xl,
              paddingHorizontal: space.xl,
              paddingTop: space.md,
              borderBottomWidth: hairline,
              borderBottomColor: palette.line,
            }}
          >
            {(
              [
                { key: 'current', label: 'Current' },
                { key: 'history', label: 'History' },
              ] as { key: Tab; label: string }[]
            ).map((item) => {
              const selected = tab === item.key;
              return (
                <Pressable
                  key={item.key}
                  accessibilityRole="tab"
                  accessibilityState={{ selected }}
                  onPress={() => setTab(item.key)}
                  hitSlop={8}
                  style={{ paddingBottom: space.sm + 4 }}
                >
                  <Text
                    style={[
                      type.bodyStrong,
                      { color: selected ? palette.ink : palette.inkMuted, fontWeight: selected ? '600' : '400' },
                    ]}
                  >
                    {item.label}
                  </Text>
                  <View
                    style={{
                      height: 2,
                      marginTop: space.sm,
                      marginBottom: -space.sm - hairline,
                      borderRadius: 1,
                      backgroundColor: selected ? palette.ink : 'transparent',
                    }}
                  />
                </Pressable>
              );
            })}
          </View>

          {tab === 'current' ? (
            summary.current.length ? (
              <View style={{ paddingTop: space.sm }}>
                {summary.current.map((course, index) =>
                  courseRow(course, index === summary.current.length - 1, false),
                )}
              </View>
            ) : (
              <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl }}>
                <Text style={[type.body, { color: palette.inkMuted }]}>Nothing in progress.</Text>
              </View>
            )
          ) : (
            <View>
              {summary.terms.map((group) => (
                <View key={group.term || 'none'}>
                  <View
                    style={{
                      flexDirection: 'row',
                      alignItems: 'baseline',
                      paddingHorizontal: space.xl,
                      paddingTop: space.xl,
                      paddingBottom: space.xs,
                    }}
                  >
                    <Text style={[type.bodyStrong, { color: palette.ink }]}>{group.label}</Text>
                    <Text style={[type.caption, { color: palette.inkFaint, marginLeft: space.sm }]}>
                      {group.courses.length} course{group.courses.length === 1 ? '' : 's'}
                    </Text>
                    <Text style={[type.caption, { color: palette.inkMuted, marginLeft: 'auto' }]}>
                      {creditsValue(group.credits)} credits
                    </Text>
                  </View>
                  <Divider />
                  {group.courses.map((course, index) =>
                    courseRow(course, index === group.courses.length - 1, true),
                  )}
                </View>
              ))}
            </View>
          )}
        </>
      )}
    </Screen>
  );
}
