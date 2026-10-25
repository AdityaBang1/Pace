import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';

import { listCourses, type CourseRow } from '../src/database/repo';
import { Button, Card, Divider, EmptyState, Notice, PageHeader, Screen } from '../src/ui/components';
import { creditsLabel } from '../src/ui/format';
import { PressableScale } from '../src/ui/press';
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

export default function Academics() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <Screen scroll>
      <PageHeader
        title="Academics"
        subtitle={
          courses.length
            ? `${courses.length} course${courses.length === 1 ? '' : 's'} imported from your documents`
            : 'Courses you import appear here'
        }
        onBack={() => router.back()}
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice
            tone="danger"
            title="Your courses could not be loaded"
            body={`${error} Nothing has been deleted: this is a read failure.`}
          />
          <Button title="Try again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {courses.length === 0 ? (
        error ? null : (
          <EmptyState
            title="No courses yet"
            body="Scan a syllabus and approve what Pace reads. Your courses, grading breakdowns and meeting times stay on this device."
          />
        )
      ) : (
        <View style={{ paddingHorizontal: space.xl }}>
          <Card padded={false}>
            {courses.map((course, index) => {
              const credits = creditsLabel(course.credits);
              const detail = [course.code, course.instructor, credits].filter(Boolean).join(' · ');
              return (
                <View key={course.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <PressableScale
                    accessibilityLabel={`Open ${course.title ?? course.code ?? 'course'}`}
                    onPress={() => router.push(`/course/${course.id}`)}
                    style={{ padding: space.lg }}
                  >
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <View style={{ flex: 1, paddingRight: space.md }}>
                        <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={2}>
                          {course.title ?? course.code ?? 'Course'}
                        </Text>
                        {detail ? (
                          <Text style={[type.caption, { color: palette.inkMuted, marginTop: 3 }]} numberOfLines={1}>
                            {detail}
                          </Text>
                        ) : null}
                      </View>
                      {course.grading_count ? (
                        <Text style={[type.caption, { color: palette.inkFaint }]}>
                          {course.grading_count} categor{course.grading_count === 1 ? 'y' : 'ies'}
                        </Text>
                      ) : null}
                    </View>
                  </PressableScale>
                </View>
              );
            })}
          </Card>
        </View>
      )}
    </Screen>
  );
}
