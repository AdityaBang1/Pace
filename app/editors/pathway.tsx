import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useState } from 'react';
import { Alert, ScrollView, Switch, Text, TextInput, View } from 'react-native';

import {
  deletePathway,
  ensureDefaults,
  readStructure,
  savePathway,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { Button, Chip, Notice, PageHeader, Screen } from '../../src/ui/components';
import { space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * A pathway is a name and one schedule per year.
 *
 * "Main" means the year is left alone: the projection falls back to that year's main
 * schedule. Choosing a schedule stores it explicitly, which is what makes a pathway
 * differ from another in a comparison.
 */
export default function PathwayEditor() {
  const params = useLocalSearchParams<{ id?: string }>();
  const id = typeof params.id === 'string' ? params.id : undefined;
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [name, setName] = useState('');
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [active, setActive] = useState(false);
  const [firstPathway, setFirstPathway] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      await ensureDefaults(db);
      const next = await readStructure(db);
      setStructure(next);
      const row = id ? next.pathways.find((item) => item.id === id) : undefined;
      if (id && !row) {
        setError('This pathway no longer exists.');
        return;
      }
      setFirstPathway(next.pathways.length === 0);
      if (row) {
        setName(row.name);
        setActive(row.is_active === 1);
        try {
          const parsed: unknown = JSON.parse(row.selections);
          setSelections(parsed && typeof parsed === 'object' ? (parsed as Record<string, string>) : {});
        } catch {
          setSelections({});
        }
      } else {
        setActive(next.pathways.length === 0);
      }
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db, id]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    const trimmed = name.trim();
    if (!trimmed || saving) return;
    setSaving(true);
    try {
      await savePathway(db, {
        id,
        name: trimmed,
        selections,
        active: active || firstPathway,
      });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save this pathway.');
      setSaving(false);
    }
  };

  const remove = () => {
    if (!id) return;
    Alert.alert('Delete pathway?', name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deletePathway(db, id);
            router.back();
          })();
        },
      },
    ]);
  };

  const years = structure?.years ?? [];
  const schedules = structure?.schedules ?? [];

  return (
    <Screen scroll edges="none">
      <PageHeader title={id ? 'Pathway' : 'New Pathway'} onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
          <Notice tone="danger" title="Couldn't load this pathway." body={error} />
        </View>
      ) : null}

      <View style={{ paddingHorizontal: space.xl }}>
        <TextInput
          value={name}
          onChangeText={setName}
          placeholder="Name"
          placeholderTextColor={palette.inkFaint}
          autoFocus={!id}
          returnKeyType="done"
          style={[
            type.heading,
            {
              color: palette.ink,
              borderBottomWidth: 0.5,
              borderColor: palette.line,
              paddingVertical: space.sm,
            },
          ]}
        />

        <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: space.lg }}>
          <View style={{ flex: 1 }}>
            <Text style={[type.body, { color: palette.ink }]}>Active</Text>
            <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>
              Home and Pace report on this pathway.
            </Text>
          </View>
          <Switch
            value={active}
            onValueChange={setActive}
            disabled={firstPathway}
            trackColor={{ true: palette.accent, false: palette.lineStrong }}
          />
        </View>

        <Text style={[type.label, { color: palette.inkFaint, marginTop: space.xl }]}>Years</Text>
        {years.length === 0 ? (
          <Text style={[type.caption, { color: palette.inkMuted, marginTop: space.sm }]}>
            Import a transcript or add a year first.
          </Text>
        ) : null}
        {years.map((year) => {
          const yearSchedules = schedules.filter((schedule) => schedule.academic_year_id === year.id);
          const chosen = selections[year.id] ?? null;
          return (
            <View key={year.id} style={{ marginTop: space.lg }}>
              <Text style={[type.bodyStrong, { color: palette.ink }]} numberOfLines={1}>
                {year.name}
              </Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={{ gap: space.sm, paddingVertical: space.sm }}
              >
                <Chip
                  label="Main"
                  selected={chosen === null}
                  tone={chosen === null ? 'accent' : 'neutral'}
                  onPress={() =>
                    setSelections((current) => {
                      const next = { ...current };
                      delete next[year.id];
                      return next;
                    })
                  }
                />
                {yearSchedules.map((schedule) => (
                  <Chip
                    key={schedule.id}
                    label={schedule.name}
                    selected={chosen === schedule.id}
                    tone={chosen === schedule.id ? 'accent' : 'neutral'}
                    onPress={() =>
                      setSelections((current) => ({ ...current, [year.id]: schedule.id }))
                    }
                  />
                ))}
              </ScrollView>
            </View>
          );
        })}

        <View style={{ marginTop: space.xxl, gap: space.sm }}>
          <Button
            title="Save"
            onPress={() => void save()}
            disabled={!name.trim()}
            loading={saving}
          />
          {id ? <Button title="Delete" kind="quiet" onPress={remove} /> : null}
        </View>
        <View style={{ height: space.xxl }} />
      </View>
    </Screen>
  );
}
