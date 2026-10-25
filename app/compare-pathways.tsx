import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Modal, Text, View } from 'react-native';

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
import {
  Button,
  Chip,
  Divider,
  EmptyState,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../src/ui/components';
import { PressableScale } from '../src/ui/press';
import { radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

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
  const { palette } = useTheme();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [aId, setAId] = useState<string | null>(null);
  const [bId, setBId] = useState<string | null>(null);
  const [picking, setPicking] = useState<'a' | 'b' | null>(null);
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
    <Screen scroll edges="top">
      <PageHeader title="Compare" onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
          <Notice tone="danger" title="Couldn't load pathways." body={error} />
        </View>
      ) : null}

      {pathways.length < 2 ? (
        <EmptyState
          title="Needs two pathways"
          body="Compare a second selection of schedules."
          action={<Button title="New Pathway" onPress={() => router.push('/editors/pathway')} />}
        />
      ) : !comparison || !viewA || !viewB ? (
        <View style={{ padding: space.xl }}>
          <Text style={[type.caption, { color: palette.inkFaint }]}>Choose two pathways.</Text>
        </View>
      ) : (
        <>
          <View style={{ paddingHorizontal: space.xl }}>
            <View style={{ flexDirection: 'row', gap: space.md }}>
              <Side
                tag="A"
                name={viewA.pathway.name}
                onPress={() => setPicking('a')}
              />
              <Side
                tag="B"
                name={viewB.pathway.name}
                onPress={() => setPicking('b')}
              />
            </View>

            <View style={{ flexDirection: 'row', gap: space.md, marginTop: space.lg }}>
              <Column
                gpa={viewA.gpa.weighted}
                credits={viewA.credits}
                requirements={countsA}
              />
              <Column
                gpa={viewB.gpa.weighted}
                credits={viewB.credits}
                requirements={countsB}
              />
            </View>

            <View
              style={{
                marginTop: space.lg,
                paddingTop: space.md,
                borderTopWidth: 0.5,
                borderColor: palette.line,
                flexDirection: 'row',
                flexWrap: 'wrap',
                gap: space.md,
              }}
            >
              <Delta
                label="GPA"
                text={
                  comparison.weightedDelta == null
                    ? 'Needs grades'
                    : signed(comparison.weightedDelta, 2)
                }
                tone={
                  comparison.weightedDelta == null
                    ? 'quiet'
                    : comparison.weightedDelta > 0
                      ? 'up'
                      : comparison.weightedDelta < 0
                        ? 'down'
                        : 'quiet'
                }
              />
              <Delta
                label="Credits"
                text={signed(comparison.creditDelta, 1)}
                tone={
                  comparison.creditDelta > 0
                    ? 'up'
                    : comparison.creditDelta < 0
                      ? 'down'
                      : 'quiet'
                }
              />
              <Delta
                label="Years changed"
                text={String(comparison.differenceCount)}
                tone="quiet"
              />
            </View>
          </View>

          <SectionHeader>Years</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            {comparison.years.map((row) => (
              <View key={row.year.id} style={{ paddingVertical: space.md }}>
                <View style={{ flexDirection: 'row', alignItems: 'baseline' }}>
                  <Text style={[type.bodyStrong, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                    {row.year.name}
                  </Text>
                  {row.same ? (
                    <Text style={[type.caption, { color: palette.inkFaint }]}>Same</Text>
                  ) : null}
                </View>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]} numberOfLines={1}>
                  {row.scheduleA?.name ?? 'No schedule'} · {row.scheduleB?.name ?? 'No schedule'}
                </Text>

                {row.same ? null : (
                  <View style={{ marginTop: space.sm, gap: 2 }}>
                    {row.removed.map((title) => (
                      <Text key={`r-${title}`} style={[type.caption, { color: palette.danger }]}>
                        − {title}
                      </Text>
                    ))}
                    {row.added.map((title) => (
                      <Text key={`a-${title}`} style={[type.caption, { color: palette.high }]}>
                        + {title}
                      </Text>
                    ))}
                    {row.changed.map((change) => (
                      <Text
                        key={`c-${change.title}`}
                        style={[type.caption, { color: palette.inkMuted }]}
                      >
                        {change.title} · {change.from} → {change.to}
                      </Text>
                    ))}
                  </View>
                )}
                <Divider />
              </View>
            ))}
          </View>
        </>
      )}

      <Modal
        visible={picking !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setPicking(null)}
      >
        <View
          style={{
            flex: 1,
            backgroundColor: palette.overlay,
            justifyContent: 'flex-end',
            padding: space.lg,
          }}
        >
          <View
            style={{
              backgroundColor: palette.surface,
              borderRadius: radius.lg,
              overflow: 'hidden',
            }}
          >
            <Text
              style={[
                type.label,
                { color: palette.inkFaint, paddingHorizontal: space.lg, paddingTop: space.lg },
              ]}
            >
              Pathway
            </Text>
            {pathways.map((row) => (
              <PressableScale
                key={row.id}
                accessibilityLabel={`Choose ${row.name}`}
                onPress={() => {
                  if (picking === 'a') setAId(row.id);
                  else setBId(row.id);
                  setPicking(null);
                }}
                style={{ paddingHorizontal: space.lg, paddingVertical: space.md }}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.sm }}>
                  <Text style={[type.body, { color: palette.ink, flex: 1 }]} numberOfLines={1}>
                    {row.name}
                  </Text>
                  {row.is_active === 1 ? <Chip label="Active" tone="accent" /> : null}
                </View>
              </PressableScale>
            ))}
            <View style={{ padding: space.lg }}>
              <Button title="Cancel" kind="quiet" onPress={() => setPicking(null)} />
            </View>
          </View>
        </View>
      </Modal>
    </Screen>
  );
}

function signed(value: number, digits: number): string {
  if (value === 0) return (0).toFixed(digits);
  const magnitude = Math.abs(value).toFixed(digits);
  return value > 0 ? `+${magnitude}` : `−${magnitude}`;
}

function Side({ tag, name, onPress }: { tag: string; name: string; onPress: () => void }) {
  const { palette } = useTheme();
  return (
    <PressableScale accessibilityLabel={`Change pathway ${tag}`} onPress={onPress} style={{ flex: 1 }}>
      <View
        style={{
          borderWidth: 0.5,
          borderColor: palette.line,
          borderRadius: radius.md,
          padding: space.md,
        }}
      >
        <Text style={[type.label, { color: palette.inkFaint }]}>{tag}</Text>
        <Text style={[type.bodyStrong, { color: palette.ink, marginTop: 2 }]} numberOfLines={1}>
          {name}
        </Text>
      </View>
    </PressableScale>
  );
}

function Column({
  gpa,
  credits,
  requirements,
}: {
  gpa: number | null;
  credits: { earned: number; inProgress: number; planned: number; all: number };
  requirements: { complete: number; total: number } | null;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ flex: 1 }}>
      <Text style={[type.number, { color: palette.ink }]}>{gpaValue(gpa)}</Text>
      <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>GPA</Text>
      <Text style={[type.number, { color: palette.ink, marginTop: space.md }]}>
        {creditsValue(credits.all)}
      </Text>
      <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>Credits</Text>
      {requirements ? (
        <>
          <Text style={[type.number, { color: palette.ink, marginTop: space.md }]}>
            {requirements.complete} / {requirements.total}
          </Text>
          <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>Requirements</Text>
        </>
      ) : null}
    </View>
  );
}

function Delta({
  label,
  text,
  tone,
}: {
  label: string;
  text: string;
  tone: 'up' | 'down' | 'quiet';
}) {
  const { palette } = useTheme();
  const color = tone === 'up' ? palette.high : tone === 'down' ? palette.danger : palette.inkMuted;
  return (
    <View style={{ minWidth: 88 }}>
      <Text style={[type.number, { color }]}>{text}</Text>
      <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>{label}</Text>
    </View>
  );
}
