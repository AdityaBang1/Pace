import { Button as NativeButton, LabeledContent, Menu, Section, Text as NativeText } from '@expo/ui/swift-ui';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';

import {
  ensureDefaults,
  ensureStructure,
  readStructure,
  type AcademicStructure,
} from '../src/database/academicRepo';
import { listCourses, readGpaScaleValue, type CourseRow } from '../src/database/repo';
import { creditsValue, gpaValue } from '../src/domain/academics';
import { parseScale, type GpaScale } from '../src/domain/gpa';
import { buildPathway, comparePathways, type PathwayView } from '../src/domain/pathways';
import { requirementProgress } from '../src/domain/requirements';
import { NativeScreen, Value } from '../src/ui/native';

/**
 * Two pathways, side by side.
 *
 * Every number here is the difference between two projections the rest of the app
 * already reports, and every year difference is a real course set difference. A year
 * where nothing changed says so and stays quiet, because the differences are what a
 * person came to read.
 */
export default function ComparePathways() {
  const db = useSQLiteContext();
  const router = useRouter();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [aId, setAId] = useState<string | null>(null);
  const [bId, setBId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      await ensureStructure(db);
      await ensureDefaults(db);
      const [nextStructure, nextCourses, scaleJson] = await Promise.all([
        readStructure(db),
        listCourses(db),
        readGpaScaleValue(db),
      ]);
      setStructure(nextStructure);
      setCourses(nextCourses);
      setScale(parseScale(scaleJson));
      setError(null);
      const rows = nextStructure.pathways;
      setAId((current) => current ?? (rows.find((row) => row.is_active === 1) ?? rows[0])?.id ?? null);
      setBId(
        (current) =>
          current ?? rows.find((row) => row.id !== (rows.find((item) => item.is_active === 1) ?? rows[0])?.id)?.id ?? null,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const pathways = structure?.pathways ?? [];
  const rowA = pathways.find((row) => row.id === aId) ?? null;
  const rowB = pathways.find((row) => row.id === bId) ?? null;

  const viewA = useMemo<PathwayView | null>(() => {
    if (!structure || !rowA) return null;
    return buildPathway(rowA, scale, structure.years, structure.schedules, courses);
  }, [structure, rowA, scale, courses]);

  const viewB = useMemo<PathwayView | null>(() => {
    if (!structure || !rowB) return null;
    return buildPathway(rowB, scale, structure.years, structure.schedules, courses);
  }, [structure, rowB, scale, courses]);

  const comparison = useMemo(
    () => (viewA && viewB ? comparePathways(viewA, viewB) : null),
    [viewA, viewB],
  );

  const requirementCount = useCallback(
    (view: PathwayView) => {
      if (!structure || !structure.requirements.length) return null;
      const progress = requirementProgress(structure.requirements, structure.links, view.courses);
      return {
        complete: progress.filter((row) => row.complete).length,
        total: progress.length,
      };
    },
    [structure],
  );

  const countsA = viewA ? requirementCount(viewA) : null;
  const countsB = viewB ? requirementCount(viewB) : null;

  return (
    <>
      <NativeScreen title="Compare" onBack={() => router.back()}>
        {error ? (
          <Section title="Couldn't load pathways">
            <NativeText>{error}</NativeText>
            <NativeButton label="Try Again" onPress={() => void refresh()} />
          </Section>
        ) : null}

        {pathways.length < 2 ? (
          <Section title="Needs two pathways" footer={<NativeText>A second selection of schedules.</NativeText>}>
            <NativeButton label="New Pathway" onPress={() => router.push('/editors/pathway')} />
          </Section>
        ) : !comparison || !viewA || !viewB ? null : (
          <>
            <Section title="Paths" footer={<NativeText>Pick the two selections of schedules to compare.</NativeText>}>
              <Menu label={`A · ${viewA.pathway.name}`} systemImage="a.circle">
                {pathways.map((row) => (
                  <NativeButton
                    key={`a-${row.id}`}
                    label={row.is_active === 1 ? `${row.name} · Active` : row.name}
                    systemImage={row.id === aId ? 'checkmark' : undefined}
                    onPress={() => setAId(row.id)}
                  />
                ))}
              </Menu>
              <Menu label={`B · ${viewB.pathway.name}`} systemImage="b.circle">
                {pathways.map((row) => (
                  <NativeButton
                    key={`b-${row.id}`}
                    label={row.is_active === 1 ? `${row.name} · Active` : row.name}
                    systemImage={row.id === bId ? 'checkmark' : undefined}
                    onPress={() => setBId(row.id)}
                  />
                ))}
              </Menu>
            </Section>

            <Section title={`A · ${viewA.pathway.name}`}>
              <LabeledContent label="GPA">
                <Value>{gpaValue(viewA.gpa.weighted)}</Value>
              </LabeledContent>
              <LabeledContent label="Credits">
                <Value>{creditsValue(viewA.credits.all)}</Value>
              </LabeledContent>
              {countsA ? (
                <LabeledContent label="Requirements">
                  <Value>{`${countsA.complete} / ${countsA.total}`}</Value>
                </LabeledContent>
              ) : null}
            </Section>

            <Section title={`B · ${viewB.pathway.name}`}>
              <LabeledContent label="GPA">
                <Value>{gpaValue(viewB.gpa.weighted)}</Value>
              </LabeledContent>
              <LabeledContent label="Credits">
                <Value>{creditsValue(viewB.credits.all)}</Value>
              </LabeledContent>
              {countsB ? (
                <LabeledContent label="Requirements">
                  <Value>{`${countsB.complete} / ${countsB.total}`}</Value>
                </LabeledContent>
              ) : null}
            </Section>

            <Section title="Difference">
              <LabeledContent label="GPA">
                <Value>
                  {comparison.weightedDelta == null
                    ? 'Needs grades'
                    : signed(comparison.weightedDelta, 2)}
                </Value>
              </LabeledContent>
              <LabeledContent label="Credits">
                <Value>{signed(comparison.creditDelta, 1)}</Value>
              </LabeledContent>
              <LabeledContent label="Years changed">
                <Value>{String(comparison.differenceCount)}</Value>
              </LabeledContent>
            </Section>

            {comparison.years.map((row) => (
              <Section
                key={row.year.id}
                title={row.year.name}
                footer={
                  <NativeText>
                    {`${row.scheduleA?.name ?? 'No schedule'} · ${row.scheduleB?.name ?? 'No schedule'}`}
                  </NativeText>
                }
              >
                {row.same ? (
                  <LabeledContent label="Result">
                    <NativeText>Same</NativeText>
                  </LabeledContent>
                ) : (
                  <>
                    {row.removed.map((title) => (
                      <NativeText key={`r-${title}`}>{`− ${title}`}</NativeText>
                    ))}
                    {row.added.map((title) => (
                      <NativeText key={`a-${title}`}>{`+ ${title}`}</NativeText>
                    ))}
                    {row.changed.map((change) => (
                      <NativeText key={`c-${change.title}`}>
                        {`${change.title} · ${change.from} → ${change.to}`}
                      </NativeText>
                    ))}
                  </>
                )}
              </Section>
            ))}
          </>
        )}
      </NativeScreen>
    </>
  );
}

function signed(value: number, digits: number): string {
  if (value === 0) return (0).toFixed(digits);
  const magnitude = Math.abs(value).toFixed(digits);
  return value > 0 ? `+${magnitude}` : `−${magnitude}`;
}
