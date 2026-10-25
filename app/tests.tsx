import { Button as NativeButton, Section } from '@expo/ui/swift-ui';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';

import { readStructure, type TestScoreRow, type TestSection } from '../src/database/academicRepo';
import { IconButton } from '../src/ui/components';
import { LabeledContent, NativeScreen, NativeText, Value } from '../src/ui/native';

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
    <NativeScreen
      title="Scores"
      onBack={() => router.back()}
      right={
        <IconButton
          glyph="+"
          symbol="plus"
          label="Add score"
          tone="accent"
          onPress={() => router.push('/editors/test')}
        />
      }
    >
      {error ? (
        <Section title="Couldn’t load scores">
          <NativeText>{error}</NativeText>
          <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void refresh()} />
        </Section>
      ) : null}

      {loaded && !error && tests.length === 0 ? (
        <Section title="No scores">
          <NativeText>SAT, ACT, PSAT, AP — anything you want on the record.</NativeText>
          <NativeButton label="Add Score" systemImage="plus" onPress={() => router.push('/editors/test')} />
        </Section>
      ) : null}

      {tests.map((test) => {
        const sections = sectionsOf(test);
        const total = test.total_score
          ? `${test.total_score}${test.max_score ? ` / ${test.max_score}` : ''}`
          : '—';
        return (
          <Section
            key={test.id}
            title={test.name}
            footer={<NativeText>{[test.kind, test.taken_on].filter(Boolean).join(' · ')}</NativeText>}
          >
            <LabeledContent label="Total">
              <Value>{total}</Value>
            </LabeledContent>
            {sections.map((section) => (
              <LabeledContent key={section.name} label={section.name}>
                <Value>{section.score}</Value>
              </LabeledContent>
            ))}
            <NativeButton
              label="Edit"
              systemImage="pencil"
              onPress={() => router.push(`/editors/test?id=${test.id}`)}
            />
          </Section>
        );
      })}
    </NativeScreen>
  );
}
