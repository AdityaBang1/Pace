import {
  Button as NativeButton,
  ProgressView,
  Section,
} from '@expo/ui/swift-ui';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';

import { ensureStructure, readStructure } from '../src/database/academicRepo';
import { listCourses } from '../src/database/repo';
import { creditsValue } from '../src/domain/academics';
import { requirementProgress, type RequirementProgress } from '../src/domain/requirements';
import { IconButton } from '../src/ui/components';
import { LabeledContent, NativeScreen, NativeText, Value } from '../src/ui/native';

/**
 * Requirements the student typed, and how far their courses get them.
 *
 * Nothing here is assumed: Pace has no list of what a school requires, so a bar
 * only exists because a person named the requirement and set the credits.
 */
export default function Requirements() {
  const db = useSQLiteContext();
  const router = useRouter();
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
    <NativeScreen
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
    >
      {error ? (
        <Section title="Couldn’t load requirements">
          <NativeText>{error}</NativeText>
          <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void refresh()} />
        </Section>
      ) : null}

      {loaded && !error && progress.length === 0 ? (
        <Section title="No requirements">
          <NativeText>Set your own credit goals.</NativeText>
          <NativeButton
            label="Add Requirement"
            systemImage="plus"
            onPress={() => router.push('/editors/requirement')}
          />
        </Section>
      ) : null}

      {progress.map((row) => {
        const required = row.requirement.required_credits;
        // A credit in progress has not been earned, so the bar fills with what is
        // finished and stops where the requirement does.
        const share = required > 0 ? Math.min(1, Math.max(0, (row.earned + row.inProgress) / required)) : 0;
        const caption = row.complete
          ? 'Complete'
          : [
              row.inProgress ? `${creditsValue(row.inProgress)} in progress` : null,
              row.planned ? `${creditsValue(row.planned)} planned` : null,
              row.remaining ? `${creditsValue(row.remaining)} remaining` : null,
            ]
              .filter(Boolean)
              .join(' · ');
        return (
          <Section
            key={row.requirement.id}
            title={row.requirement.name}
            footer={<NativeText>{caption}</NativeText>}
          >
            <LabeledContent label="Credits">
              <Value>{`${creditsValue(row.earned)} / ${creditsValue(required)}`}</Value>
            </LabeledContent>
            <ProgressView value={share} />
            <NativeButton
              label="Edit"
              systemImage="pencil"
              onPress={() => router.push(`/editors/requirement?id=${row.requirement.id}`)}
            />
          </Section>
        );
      })}
    </NativeScreen>
  );
}
