import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Alert, Text, View } from 'react-native';

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
import {
  Button,
  Chip,
  Divider,
  EmptyState,
  IconButton,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../../src/ui/components';
import { PressableScale } from '../../src/ui/press';
import { space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * Plan: which schedules the student's years are made of, and which of those
 * selections is the one Pace reports on.
 *
 * A pathway is a choice, not a copy: it points at schedules that already exist, so
 * changing a year here never rewrites a course and a comparison is between real
 * sets of courses rather than between lists of names.
 */
export default function Plan() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

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

  const openMenu = (row: PathwayRow) => {
    Alert.alert(row.name, undefined, [
      { text: 'Edit', onPress: () => router.push(`/editors/pathway?id=${row.id}`) },
      { text: 'Duplicate', onPress: () => duplicate(row) },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => confirmDelete(row),
      },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  return (
    <Screen scroll edges="tab">
      <PageHeader
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
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
          <Notice tone="danger" title="Couldn't load your plan." body={error} />
          <View style={{ height: space.sm }} />
          <Button title="Try Again" kind="secondary" onPress={() => void refresh()} />
        </View>
      ) : null}

      {!structure ? null : pathways.length === 0 || !view || !activePathway ? (
        <EmptyState
          title="No pathway"
          body="A pathway picks one schedule for each year."
          action={<Button title="Create Pathway" onPress={() => router.push('/editors/pathway')} />}
        />
      ) : (
        <>
          <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end' }}>
              <View style={{ flex: 1 }}>
                <Text style={[type.title, { color: palette.ink }]} numberOfLines={1}>
                  {activePathway.name}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                  {view.years.length} year{view.years.length === 1 ? '' : 's'}
                </Text>
              </View>
              <View style={{ alignItems: 'flex-end' }}>
                <Text
                  style={{
                    fontSize: 30,
                    lineHeight: 34,
                    fontWeight: '600',
                    letterSpacing: -0.8,
                    fontVariant: ['tabular-nums'],
                    color: palette.ink,
                  }}
                >
                  {gpaValue(view.gpa.weighted)}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                  Projected GPA
                </Text>
              </View>
            </View>

            <View style={{ flexDirection: 'row', gap: space.xl, marginTop: space.lg }}>
              <Stat label="Credits" value={creditsValue(totalCredits)} />
              <Stat label="Earned" value={creditsValue(view.credits.earned)} />
              <Stat label="Planned" value={creditsValue(view.credits.planned)} />
            </View>

            {view.gpa.weighted == null ? (
              <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.md }]}>
                {view.needingGrades
                  ? `${view.needingGrades} course${view.needingGrades === 1 ? '' : 's'} need an expected grade.`
                  : 'A projected GPA needs a GPA scale.'}
              </Text>
            ) : null}
          </View>

          <SectionHeader>Years</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <View
              style={{
                borderTopWidth: 0.5,
                borderColor: palette.line,
              }}
            >
              {view.years.map((row) => {
                const schedule = row.schedule;
                return (
                  <View key={row.year.id}>
                    <PressableScale
                      accessibilityLabel={schedule ? `Open ${row.year.name} schedule` : row.year.name}
                      disabled={!schedule}
                      onPress={() => {
                        if (schedule) router.push(`/schedule/${schedule.id}`);
                      }}
                      style={{ paddingVertical: space.md }}
                    >
                      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <View style={{ flex: 1, paddingRight: space.md }}>
                          <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                            {row.year.name}
                          </Text>
                          <Text
                            style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}
                            numberOfLines={1}
                          >
                            {schedule
                              ? `${schedule.name}${row.explicit ? '' : ' · main'} · ${row.courses.length} course${
                                  row.courses.length === 1 ? '' : 's'
                                }`
                              : 'No schedule'}
                          </Text>
                        </View>
                        <Text style={[type.number, { color: palette.ink }]}>
                          {schedule ? creditsValue(row.credits.all) : '—'}
                        </Text>
                        {schedule ? (
                          <Text style={{ color: palette.inkFaint, fontSize: 18, marginLeft: space.sm }}>
                            ›
                          </Text>
                        ) : null}
                      </View>
                    </PressableScale>
                    <Divider />
                  </View>
                );
              })}
            </View>
          </View>

          <SectionHeader
            right={
              <Text style={[type.caption, { color: palette.inkFaint }]}>{pathways.length}</Text>
            }
          >
            Pathways
          </SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <View style={{ borderColor: palette.line }}>
              {pathways.map((row) => {
                const isActive = row.is_active === 1;
                return (
                  <View key={row.id}>
                    <PressableScale
                      accessibilityLabel={`Use ${row.name}`}
                      accessibilityState={{ selected: isActive }}
                      onPress={() => makeActive(row)}
                      onLongPress={() => openMenu(row)}
                      style={{ paddingVertical: space.md }}
                    >
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.sm }}>
                        <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                          {row.name}
                        </Text>
                        {isActive ? <Chip label="Active" tone="accent" /> : null}
                      </View>
                    </PressableScale>
                    <Divider />
                  </View>
                );
              })}
            </View>
          </View>

          <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg }}>
            {pathways.length > 1 ? (
              <Button title="Compare" kind="secondary" onPress={() => router.push('/compare-pathways')} />
            ) : (
              <Text style={[type.caption, { color: palette.inkFaint }]}>
                Two pathways can be compared.
              </Text>
            )}
          </View>

          <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl }}>
            <PressableScale
              accessibilityRole="button"
              onPress={() => router.push('/gpa-scale')}
              style={{ paddingVertical: space.md }}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Text style={[type.body, { color: palette.ink, flex: 1 }]}>GPA Scale</Text>
                <Text style={[type.caption, { color: scale ? palette.inkMuted : palette.warn }]}>
                  {scale ? 'Set' : 'Required'}
                </Text>
                <Text style={{ color: palette.inkFaint, fontSize: 18, marginLeft: space.sm }}>›</Text>
              </View>
            </PressableScale>
          </View>
        </>
      )}
    </Screen>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  const { palette } = useTheme();
  return (
    <View>
      <Text style={[type.number, { color: palette.ink }]}>{value}</Text>
      <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>{label}</Text>
    </View>
  );
}
