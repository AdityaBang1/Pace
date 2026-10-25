import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';

import { readStructure, type TestScoreRow, type TestSection } from '../src/database/academicRepo';
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
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

function sectionsOf(test: TestScoreRow): TestSection[] {
  try {
    const parsed: unknown = JSON.parse(test.sections);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (row): row is TestSection =>
        !!row &&
        typeof (row as TestSection).name === 'string' &&
        typeof (row as TestSection).score === 'string',
    );
  } catch {
    return [];
  }
}

/** The scores that sit beside the GPA rather than inside it. */
export default function Tests() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [tests, setTests] = useState<TestScoreRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const structure = await readStructure(db);
      setTests(structure.tests);
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
        title="Scores"
        onBack={() => router.back()}
        right={<IconButton glyph="+" label="Add score" onPress={() => router.push('/editors/test')} />}
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md, gap: space.sm }}>
          <Notice tone="danger" title="Couldn’t load scores" body={error} />
          <Button title="Try Again" kind="secondary" compact onPress={() => void refresh()} />
        </View>
      ) : null}

      {loaded && !error && tests.length === 0 ? (
        <EmptyState
          title="No scores"
          action={<Button title="Add Score" onPress={() => router.push('/editors/test')} />}
          body="SAT, ACT, PSAT, AP — anything you want on the record."
        />
      ) : null}

      {tests.length ? (
        <View style={{ paddingHorizontal: space.xl }}>
          <Card padded={false}>
            {tests.map((test, index) => {
              const sections = sectionsOf(test);
              return (
                <View key={test.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <PressableScale
                    accessibilityRole="button"
                    accessibilityLabel={`Edit ${test.name}`}
                    onPress={() => router.push(`/editors/test?id=${test.id}`)}
                    style={{ padding: space.lg }}
                  >
                    <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: space.md }}>
                      <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                        {test.name}
                      </Text>
                      <Text style={[type.number, { color: palette.ink }]} numberOfLines={1}>
                        {test.total_score ?? '—'}
                        {test.max_score ? (
                          <Text style={[type.caption, { color: palette.inkFaint }]}> / {test.max_score}</Text>
                        ) : null}
                      </Text>
                    </View>
                    <Text style={[type.caption, { color: palette.inkFaint, marginTop: 2 }]} numberOfLines={1}>
                      {[test.kind, test.taken_on].filter(Boolean).join(' · ')}
                    </Text>
                    {sections.map((section) => (
                      <View
                        key={section.name}
                        style={{ flexDirection: 'row', alignItems: 'baseline', marginTop: space.xs }}
                      >
                        <Text style={[type.caption, { color: palette.inkMuted, flex: 1 }]} numberOfLines={1}>
                          {section.name}
                        </Text>
                        <Text style={[type.caption, { color: palette.ink }]}>{section.score}</Text>
                      </View>
                    ))}
                  </PressableScale>
                </View>
              );
            })}
          </Card>
        </View>
      ) : null}
    </Screen>
  );
}
