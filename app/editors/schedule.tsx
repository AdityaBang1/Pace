import {
  Button as NativeButton,
  TextField,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import { disabled, foregroundStyle } from '@expo/ui/swift-ui/modifiers';
import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';

import {
  deleteSchedule,
  duplicateSchedule,
  readStructure,
  saveSchedule,
  setMainSchedule,
  type AcademicYearRow,
  type ScheduleRow,
} from '../../src/database/academicRepo';
import { NativeScreen, NativeText, Section } from '../../src/ui/native';
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

  const nameRef = useRef<TextFieldRef>(null);

  useEffect(() => {
    void (async () => {
      const structure = await readStructure(db);
      setYears(structure.years);
      const target = id ?? duplicateFrom;
      const row = target ? (structure.schedules.find((item) => item.id === target) ?? null) : null;
      setSource(row);
      if (row) {
        const nextName = duplicateFrom ? `${row.name} Copy` : row.name;
        setName(nextName);
        setYear(row.academic_year_id);
        // The field is mounted with the sheet, so it can be filled once the row is read.
        void nameRef.current?.setText(nextName);
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
    <NativeScreen topInset={false} title={title} onBack={() => router.back()}>
      {error ? (
        <Section>
          <NativeText modifiers={[foregroundStyle(palette.danger)]}>{error}</NativeText>
        </Section>
      ) : null}

      <Section title="Name">
        <TextField
          ref={nameRef}
          placeholder="Main"
          autoFocus={!id && !duplicateFrom}
          onTextChange={setName}
        />
      </Section>

      {!id && !duplicateFrom ? (
        <Section title="Year">
          {years.map((row) => (
            <NativeButton
              key={row.id}
              label={row.name}
              onPress={() => setYear(row.id)}
              systemImage={year === row.id ? 'checkmark' : undefined}
            />
          ))}
          {years.length === 0 ? <NativeText>Add an academic year first.</NativeText> : null}
        </Section>
      ) : null}

      <Section>
        <NativeButton
          label={duplicateFrom ? 'Duplicate' : 'Save'}
          systemImage="checkmark"
          modifiers={[disabled(!canSave || saving)]}
          onPress={() => void save()}
        />
        {id && source && source.is_main !== 1 ? (
          <NativeButton label="Make Main" systemImage="star" onPress={makeMain} />
        ) : null}
        {id && source ? (
          <NativeButton label="Delete" role="destructive" onPress={remove} />
        ) : null}
      </Section>

      {saving ? (
        <Section>
          <NativeText>Saving…</NativeText>
        </Section>
      ) : null}
    </NativeScreen>
  );
}
