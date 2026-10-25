import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert } from 'react-native';

import { Button as NativeButton, LabeledContent, Section, Text as NativeText } from '@expo/ui/swift-ui';

import {
  activatePathway,
  deletePathway,
  ensureDefaults,
  ensureStructure,
  parseSelections,
  readStructure,
  savePathway,
  type AcademicStructure,
  type PathwayRow,
} from '../../src/database/academicRepo';
import { listCourses, readGpaScaleValue, type CourseRow } from '../../src/database/repo';
import { creditsValue, gpaValue } from '../../src/domain/academics';
import { parseScale, type GpaScale } from '../../src/domain/gpa';
import { buildPathway, type PathwayView } from '../../src/domain/pathways';
import { IconButton } from '../../src/ui/components';
import { NativeScreen, Value } from '../../src/ui/native';

/**
 * Plan: which schedules the student's years are made of, and which of those
 * selections is the one Pace reports on.
 *
 * A pathway is a choice, not a copy: it points at schedules that already exist, so
 * changing a year here never rewrites a course and a comparison is between real
 * sets of courses rather than between lists of names. The screen is a system form:
 * a summary section, the years, then the pathways and the actions on them.
 */
export default function Plan() {
  const db = useSQLiteContext();
  const router = useRouter();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      // Idempotent: a store imported before years existed gets them here, and a
      // store with no pathway gets one, so this screen is never an empty shell.
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
  const activePathway = pathways.find((row) => row.is_active === 1) ?? pathways[0] ?? null;

  const view = useMemo<PathwayView | null>(() => {
    if (!structure || !activePathway) return null;
    return buildPathway(activePathway, scale, structure.years, structure.schedules, courses);
  }, [structure, activePathway, scale, courses]);

  const totalCredits = view
    ? Math.round((view.credits.earned + view.credits.inProgress + view.credits.planned) * 100) / 100
    : 0;

  const makeActive = (row: PathwayRow) => {
    Haptics.selectionAsync().catch(() => {});
    void (async () => {
      await activatePathway(db, row.id);
      await refresh();
    })();
  };

  const duplicate = (row: PathwayRow) => {
    void (async () => {
      await savePathway(db, { name: `${row.name} copy`, selections: parseSelections(row) });
      await refresh();
    })();
  };

  const confirmDelete = (row: PathwayRow) => {
    Alert.alert('Delete pathway?', row.name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deletePathway(db, row.id);
            await refresh();
          })();
        },
      },
    ]);
  };

  const missingGpaNote = view
    ? view.gpa.weighted == null
      ? view.needingGrades
        ? `${view.needingGrades} course${view.needingGrades === 1 ? '' : 's'} need an expected grade.`
        : 'A projected GPA needs a GPA scale.'
      : null
    : null;

  return (
    <NativeScreen
      topInset={false}
      title="Plan"
      right={
        <IconButton
          glyph="+"
          symbol="plus"
          label="New pathway"
          onPress={() => router.push('/editors/pathway')}
          tone="accent"
          size={22}
        />
      }
    >
      {error ? (
        <Section title="Couldn't load your plan">
          <NativeText>{error}</NativeText>
          <NativeButton label="Try Again" onPress={() => void refresh()} />
        </Section>
      ) : null}

      {structure && (!pathways.length || !view || !activePathway) ? (
        <Section title="No pathway" footer={<NativeText>One schedule for each year.</NativeText>}>
          <NativeButton label="Create Pathway" onPress={() => router.push('/editors/pathway')} />
        </Section>
      ) : null}

      {view && activePathway ? (
        <>
          <Section title={activePathway.name} footer={missingGpaNote ? <NativeText>{missingGpaNote}</NativeText> : undefined}>
            <LabeledContent label="Projected GPA">
              <Value>{gpaValue(view.gpa.weighted)}</Value>
            </LabeledContent>
            <LabeledContent label="Credits">
              <Value>{creditsValue(totalCredits)}</Value>
            </LabeledContent>
            <LabeledContent label="Earned">
              <Value>{creditsValue(view.credits.earned)}</Value>
            </LabeledContent>
            <LabeledContent label="Planned">
              <Value>{creditsValue(view.credits.planned)}</Value>
            </LabeledContent>
            <LabeledContent label="Years">
              <Value>{String(view.years.length)}</Value>
            </LabeledContent>
          </Section>

          <Section title="Years">
            {view.years.map((row) =>
              row.schedule ? (
                <NativeButton
                  key={row.year.id}
                  label={`${row.year.name} · ${row.schedule.name}${
                    row.explicit ? '' : ' · main'
                  } · ${row.courses.length} course${row.courses.length === 1 ? '' : 's'} · ${creditsValue(
                    row.credits.all,
                  )} cr`}
                  systemImage="calendar"
                  onPress={() => router.push(`/schedule/${row.schedule?.id}`)}
                />
              ) : (
                <LabeledContent key={row.year.id} label={row.year.name}>
                  <NativeText>No schedule</NativeText>
                </LabeledContent>
              ),
            )}
          </Section>

          <Section title="Pathways">
            {pathways.map((row) => (
              <NativeButton
                key={row.id}
                label={row.is_active === 1 ? `${row.name} · Active` : row.name}
                systemImage={row.is_active === 1 ? 'checkmark.circle.fill' : 'circle'}
                onPress={() => makeActive(row)}
              />
            ))}
          </Section>

          <Section title={`Manage · ${activePathway.name}`}>
            <NativeButton
              label="Edit"
              systemImage="pencil"
              onPress={() => router.push(`/editors/pathway?id=${activePathway.id}`)}
            />
            <NativeButton
              label="Duplicate"
              systemImage="plus.square.on.square"
              onPress={() => duplicate(activePathway)}
            />
            <NativeButton
              label="Delete"
              systemImage="trash"
              role="destructive"
              onPress={() => confirmDelete(activePathway)}
            />
          </Section>

          <Section title="GPA Scale">
            <LabeledContent label="Scale">
              <Value>{scale ? 'Set' : 'Required'}</Value>
            </LabeledContent>
            <NativeButton label="Open GPA Scale" systemImage="percent" onPress={() => router.push('/gpa-scale')} />
          </Section>

          {pathways.length > 1 ? (
            <Section>
              <NativeButton
                label="Compare pathways"
                systemImage="arrow.left.arrow.right"
                onPress={() => router.push('/compare-pathways')}
              />
            </Section>
          ) : null}
        </>
      ) : null}
    </NativeScreen>
  );
}
