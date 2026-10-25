import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { Alert, Text, TextInput, View } from 'react-native';

import {
  deleteSchedule,
  duplicateSchedule,
  readStructure,
  saveSchedule,
  setMainSchedule,
  type AcademicYearRow,
  type ScheduleRow,
} from '../../src/database/academicRepo';
import {
  Button,
  Chip,
  Notice,
  PageHeader,
  Screen,
  SectionHeader,
} from '../../src/ui/components';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

/**
 * A schedule's name, and the two things that change its standing: making it the
 * year's main schedule, or copying it into a scenario. Nothing about its courses is
 * edited here — that is the schedule screen's own job.
 */
export default function ScheduleEditor() {
  const { id, yearId, duplicateFrom, saveAs } = useLocalSearchParams<{
    id?: string;
    yearId?: string;
    duplicateFrom?: string;
    saveAs?: string;
  }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [years, setYears] = useState<AcademicYearRow[]>([]);
  const [source, setSource] = useState<ScheduleRow | null>(null);
  const [name, setName] = useState('');
  const [year, setYear] = useState<string | null>(yearId ?? null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const structure = await readStructure(db);
      setYears(structure.years);
      const target = id ?? duplicateFrom;
      const row = target ? (structure.schedules.find((item) => item.id === target) ?? null) : null;
      setSource(row);
      if (row) {
        setName(duplicateFrom ? `${row.name} Copy` : row.name);
        setYear(row.academic_year_id);
      } else if (!year) {
        setYear(
          structure.years.find((item) => item.status === 'current')?.id ??
            structure.years[0]?.id ??
            null,
        );
      }
    })();
    // The route's own params identify what is being edited; they do not change.
  }, [db, id, duplicateFrom, year]);

  const canSave = name.trim().length > 0 && Boolean(year);

  const save = async () => {
    if (!canSave || !year) return;
    setSaving(true);
    try {
      if (duplicateFrom && source) {
        const created = await duplicateSchedule(db, source.id, name.trim());
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
        if (created) router.replace(`/schedule/${created}`);
        else router.back();
        return;
      }
      await saveSchedule(db, { id: id ?? undefined, academicYearId: year, name: name.trim() });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The schedule could not be saved.');
    } finally {
      setSaving(false);
    }
  };

  const makeMain = () => {
    if (!source) return;
    void (async () => {
      await setMainSchedule(db, source.id);
      router.back();
    })();
  };

  const remove = () => {
    if (!source) return;
    Alert.alert('Delete schedule?', 'Its courses stay, but lose their place in this year.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deleteSchedule(db, source.id);
            router.back();
          })();
        },
      },
    ]);
  };

  const title = duplicateFrom ? 'Duplicate' : saveAs ? 'Save as Schedule' : id ? 'Edit Schedule' : 'New Schedule';

  return (
    <Screen scroll edges="top">
      <PageHeader title={title} onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
          <Notice tone="danger" title="Couldn't save" body={error} />
        </View>
      ) : null}

      <SectionHeader>Name</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <TextInput
          value={name}
          onChangeText={setName}
          placeholder="Main"
          placeholderTextColor={palette.inkFaint}
          autoFocus={!id && !duplicateFrom}
          returnKeyType="done"
          style={[
            type.body,
            {
              color: palette.ink,
              backgroundColor: palette.surface,
              borderWidth: hairline,
              borderColor: palette.line,
              borderRadius: radius.md,
              paddingHorizontal: space.lg,
              paddingVertical: space.md,
            },
          ]}
        />
      </View>

      {!id && !duplicateFrom ? (
        <>
          <SectionHeader>Year</SectionHeader>
          <View
            style={{
              flexDirection: 'row',
              flexWrap: 'wrap',
              gap: space.sm,
              paddingHorizontal: space.xl,
            }}
          >
            {years.map((row) => (
              <Chip
                key={row.id}
                label={row.name}
                selected={year === row.id}
                onPress={() => setYear(row.id)}
              />
            ))}
            {years.length === 0 ? (
              <Text style={[type.caption, { color: palette.inkMuted }]}>
                Add an academic year first.
              </Text>
            ) : null}
          </View>
        </>
      ) : null}

      <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl, gap: space.md }}>
        <Button
          title={duplicateFrom ? 'Duplicate' : 'Save'}
          onPress={() => void save()}
          disabled={!canSave}
          loading={saving}
        />
        {id && source && source.is_main !== 1 ? (
          <Button title="Make Main" kind="secondary" onPress={makeMain} />
        ) : null}
        {id && source ? <Button title="Delete" kind="quiet" onPress={remove} /> : null}
      </View>
    </Screen>
  );
}
