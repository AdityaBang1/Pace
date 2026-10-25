import { useFocusEffect, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import {
  deletePlannedCourse,
  latestTranscript,
  listCourses,
  listPlannedCourses,
  readGpaScaleValue,
  upsertPlannedCourse,
  type PlannedCourseRow,
  type TranscriptRow,
} from '../src/database/repo';
import { creditsValue, gpaValue, summarise, type AcademicSummary } from '../src/domain/academics';
import {
  DEFAULT_SCALE,
  LEVELS,
  LETTERS,
  levelLabel,
  parseScale,
  projectGpa,
  type CourseLevel,
  type GpaScale,
} from '../src/domain/gpa';
import { Button, Card, Divider, EmptyState, PageHeader, Screen, SectionHeader } from '../src/ui/components';
import { PressableScale } from '../src/ui/press';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

interface Draft {
  id: string | null;
  name: string;
  credits: string;
  level: CourseLevel;
  expectedGrade: string;
}

const NEW_DRAFT: Draft = { id: null, name: '', credits: '1', level: 'regular', expectedGrade: 'A' };

export default function Plan() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [planned, setPlanned] = useState<PlannedCourseRow[]>([]);
  const [summary, setSummary] = useState<AcademicSummary | null>(null);
  const [transcript, setTranscript] = useState<TranscriptRow | null>(null);
  const [scale, setScale] = useState<GpaScale | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);

  const refresh = useCallback(async () => {
    const [rows, courses, latest, scaleJson] = await Promise.all([
      listPlannedCourses(db),
      listCourses(db),
      latestTranscript(db),
      readGpaScaleValue(db),
    ]);
    setPlanned(rows);
    setSummary(summarise(courses));
    setTranscript(latest);
    setScale(parseScale(scaleJson));
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  const projection = useMemo(() => {
    const history = summary?.scaled ?? [];
    const plannedInput = planned.map((row) => ({
      credits: row.credits,
      level: row.level,
      expectedGrade: row.expected_grade,
    }));
    return projectGpa(scale, history, plannedInput);
  }, [planned, scale, summary]);

  const save = async () => {
    if (!draft) return;
    const name = draft.name.trim();
    if (!name) return;
    const credits = Number(draft.credits.replace(',', '.'));
    await upsertPlannedCourse(db, {
      id: draft.id ?? `plan-${Date.now().toString(36)}`,
      name,
      credits: Number.isFinite(credits) && credits > 0 ? credits : 1,
      level: draft.level,
      expectedGrade: draft.expectedGrade,
      term: null,
    });
    setDraft(null);
    await refresh();
  };

  const remove = async () => {
    if (!draft?.id) return;
    await deletePlannedCourse(db, draft.id);
    setDraft(null);
    await refresh();
  };

  const scaleLabel = scale ? 'Custom' : 'Not set';
  const reportedGpa = transcript?.reported_gpa ?? null;

  return (
    <Screen scroll>
      <PageHeader
        title="Plan"
        onBack={() => router.back()}
        right={
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Add a planned course"
            hitSlop={10}
            onPress={() => setDraft({ ...NEW_DRAFT })}
            style={({ pressed }) => ({ opacity: pressed ? 0.5 : 1, paddingVertical: space.sm })}
          >
            <Text style={[type.body, { color: palette.accent, fontWeight: '600' }]}>Add Course</Text>
          </Pressable>
        }
      />

      {draft ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
          <Card>
            <TextInput
              value={draft.name}
              onChangeText={(name) => setDraft({ ...draft, name })}
              placeholder="Course name"
              placeholderTextColor={palette.inkFaint}
              autoFocus
              returnKeyType="done"
              style={[type.bodyStrong, styles.input, { color: palette.ink, borderColor: palette.line }]}
            />
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md, marginTop: space.md }}>
              <Text style={[type.body, { color: palette.inkMuted }]}>Credits</Text>
              <TextInput
                value={draft.credits}
                onChangeText={(credits) => setDraft({ ...draft, credits })}
                keyboardType="decimal-pad"
                style={[
                  type.number,
                  styles.credits,
                  { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surfaceSunken },
                ]}
              />
            </View>

            <Text style={[type.label, { color: palette.inkFaint, marginTop: space.lg, marginBottom: space.sm }]}>
              Level
            </Text>
            <View style={{ flexDirection: 'row', gap: space.sm }}>
              {LEVELS.map((row) => (
                <Option
                  key={row.value}
                  label={row.label}
                  selected={draft.level === row.value}
                  onPress={() => setDraft({ ...draft, level: row.value })}
                />
              ))}
            </View>

            <Text style={[type.label, { color: palette.inkFaint, marginTop: space.lg, marginBottom: space.sm }]}>
              Expected
            </Text>
            <View style={{ flexDirection: 'row', gap: space.sm }}>
              {LETTERS.map((letter) => (
                <Option
                  key={letter}
                  label={letter}
                  selected={draft.expectedGrade === letter}
                  onPress={() => setDraft({ ...draft, expectedGrade: letter })}
                />
              ))}
            </View>

            <View style={{ flexDirection: 'row', gap: space.sm, marginTop: space.xl }}>
              <View style={{ flex: 1 }}>
                <Button title="Save" onPress={() => void save()} disabled={!draft.name.trim()} />
              </View>
              <View style={{ flex: 1 }}>
                <Button title="Cancel" kind="quiet" onPress={() => setDraft(null)} />
              </View>
            </View>
            {draft.id ? (
              <View style={{ marginTop: space.sm }}>
                <Button title="Remove" kind="quiet" compact onPress={() => void remove()} />
              </View>
            ) : null}
          </Card>
        </View>
      ) : null}

      {planned.length === 0 && !draft ? (
        <EmptyState
          title="Nothing planned"
          body="Add the courses you are thinking about taking."
          action={<Button title="Add Course" onPress={() => setDraft({ ...NEW_DRAFT })} />}
        />
      ) : (
        planned.length > 0 && (
          <View style={{ paddingHorizontal: space.xl }}>
            <Card padded={false}>
              {planned.map((row, index) => (
                <View key={row.id}>
                  {index > 0 ? <Divider inset={space.lg} /> : null}
                  <PressableScale
                    accessibilityLabel={`Edit ${row.name}`}
                    onPress={() =>
                      setDraft({
                        id: row.id,
                        name: row.name,
                        credits: String(row.credits),
                        level: row.level,
                        expectedGrade: row.expected_grade,
                      })
                    }
                    style={{ padding: space.lg }}
                  >
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <View style={{ flex: 1, paddingRight: space.md }}>
                        <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                          {row.name}
                        </Text>
                        <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                          {levelLabel(row.level)} · {creditsValue(row.credits)} credits · {row.expected_grade}
                        </Text>
                      </View>
                      <Text style={{ color: palette.inkFaint, fontSize: 18 }}>›</Text>
                    </View>
                  </PressableScale>
                </View>
              ))}
            </Card>
          </View>
        )
      )}

      <SectionHeader>WHAT IF</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between' }}>
              <View>
                <Text style={[type.title, { color: palette.ink, fontVariant: ['tabular-nums'] }]}>
                  {gpaValue(projection.gpa)}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>Projected</Text>
              </View>
              <View style={{ alignItems: 'flex-end' }}>
                <Text style={[type.number, { color: palette.ink }]}>
                  {gpaValue(reportedGpa)}
                </Text>
                <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
                  {reportedGpa == null ? 'GPA on file' : 'Reported'}
                </Text>
              </View>
            </View>

            <View style={{ height: space.lg }} />
            {projection.reason === 'no-scale' ? (
              <FieldLink label="Set GPA Scale" value="Required" onPress={() => router.push('/gpa-scale')} />
            ) : projection.reason === 'no-history' ? (
              <Row label="Basis" value="No graded courses" />
            ) : projection.reason === 'no-courses' ? (
              <Row label="Basis" value="Nothing planned" />
            ) : (
              <Row
                label="Basis"
                value={`${creditsValue(projection.credits)} credits · ${planned.length} planned`}
              />
            )}
            <Divider />
            <FieldLink
              label="GPA Scale"
              value={scaleLabel}
              onPress={() => router.push('/gpa-scale')}
            />
          </View>
        </Card>
        {scale ? null : (
          <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.sm }]}>
            A projection needs a scale. {DEFAULT_SCALE.regular.A.toFixed(1)} is a common starting point.
          </Text>
        )}
      </View>
    </Screen>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  const { palette } = useTheme();
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', minHeight: 36 }}>
      <Text style={[type.body, { color: palette.inkMuted }]}>{label}</Text>
      <Text style={[type.body, { color: palette.ink }]} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

function FieldLink({ label, value, onPress }: { label: string; value: string; onPress: () => void }) {
  const { palette } = useTheme();
  return (
    <PressableScale accessibilityRole="button" onPress={onPress}>
      <View
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          minHeight: 40,
        }}
      >
        <Text style={[type.body, { color: palette.ink }]}>{label}</Text>
        <Text style={[type.body, { color: palette.inkMuted }]}>
          {value}
          <Text style={{ color: palette.inkFaint }}>  ›</Text>
        </Text>
      </View>
    </PressableScale>
  );
}

function Option({
  label,
  selected,
  onPress,
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  const { palette } = useTheme();
  return (
    <PressableScale accessibilityState={{ selected }} onPress={onPress}>
      <View
        style={{
          paddingVertical: 7,
          paddingHorizontal: space.md,
          borderRadius: radius.sm,
          backgroundColor: selected ? palette.accent : palette.surfaceSunken,
        }}
      >
        <Text
          style={[
            type.caption,
            { color: selected ? palette.accentInk : palette.inkMuted, fontWeight: '600' },
          ]}
        >
          {label}
        </Text>
      </View>
    </PressableScale>
  );
}

const styles = StyleSheet.create({
  input: {
    borderBottomWidth: hairline,
    paddingVertical: space.sm,
  },
  credits: {
    marginLeft: 'auto',
    minWidth: 64,
    textAlign: 'right',
    paddingVertical: 6,
    paddingHorizontal: space.sm,
    borderRadius: radius.sm,
    borderWidth: hairline,
  },
});
