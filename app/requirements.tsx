import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';

import { ensureStructure, readStructure } from '../src/database/academicRepo';
import { listCourses, type CourseRow } from '../src/database/repo';
import { creditsValue } from '../src/domain/academics';
import { requirementProgress, type RequirementProgress } from '../src/domain/requirements';
import {
  Button,
  Card,
  Divider,
  EmptyState,
  IconButton,
  Notice,
  PageHeader,
  Screen,
} from '../src/ui/components';
import { PressableScale } from '../src/ui/press';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

/**
 * Requirements the student typed, and how far their courses get them.
 *
 * Nothing here is assumed: Pace has no list of what a school requires, so a bar
 * only exists because a person named the requirement and set the credits.
 */
export default function Requirements() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [progress, setProgress] = useState<RequirementProgress[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      await ensureStructure(db);
      const [structure, courses] = await Promise.all([readStructure(db), listCourses(db)]);
      setProgress(requirementProgress(structure.requirements, structure.links, courses));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    } finally {
      setLoaded(true);
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

  return (
    <Screen scroll>
      <PageHeader
        title="Requirements"
        onBack={() => router.back()}
        right={
          <IconButton
            glyph="+"
            symbol="plus"
            label="Add requirement"
            onPress={() => router.push('/editors/requirement')}
          />
        }
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice tone="danger" title="Couldn’t load requirements" body={error} />
          <Button title="Try Again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {loaded && !error && progress.length === 0 ? (
        <EmptyState
          title="No requirements"
          body="Set your own credit goals."
          action={<Button title="Add Requirement" onPress={() => router.push('/editors/requirement')} />}
        />
      ) : null}

      {progress.map((row) => {
        const required = row.requirement.required_credits;
        const earnedShare = required > 0 ? Math.min(1, row.earned / required) : 0;
        const progressShare = required > 0 ? Math.min(1, (row.earned + row.inProgress) / required) : 0;
        return (
          <View key={row.requirement.id} style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
            <Card padded={false}>
              <PressableScale
                accessibilityRole="button"
                accessibilityLabel={`Edit ${row.requirement.name}`}
                onPress={() => router.push(`/editors/requirement?id=${row.requirement.id}`)}
                style={{ padding: space.lg }}
              >
                <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: space.md }}>
                  <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                    {row.requirement.name}
                  </Text>
                  <Text style={[type.number, { color: palette.ink }]}>
                    {creditsValue(row.earned)}
                    <Text style={[type.caption, { color: palette.inkFaint }]}>
                      {' '}
                      / {creditsValue(required)}
                    </Text>
                  </Text>
                </View>

                <View
                  style={{
                    height: 6,
                    borderRadius: radius.sm,
                    backgroundColor: palette.surfaceSunken,
                    marginTop: space.sm,
                    overflow: 'hidden',
                  }}
                >
                  <View
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${progressShare * 100}%`,
                      backgroundColor: palette.accentSoft,
                      borderRadius: radius.sm,
                    }}
                  />
                  <View
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${earnedShare * 100}%`,
                      backgroundColor: row.complete ? palette.accent : palette.accent,
                      borderRadius: radius.sm,
                    }}
                  />
                </View>

                <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.xs }]}>
                  {row.complete
                    ? 'Complete'
                    : [
                        row.inProgress ? `${creditsValue(row.inProgress)} in progress` : null,
                        row.planned ? `${creditsValue(row.planned)} planned` : null,
                        row.remaining ? `${creditsValue(row.remaining)} remaining` : null,
                      ]
                        .filter(Boolean)
                        .join(' · ')}
                </Text>
              </PressableScale>
            </Card>
          </View>
        );
      })}
      <View style={{ height: hairline }} />
    </Screen>
  );
}
