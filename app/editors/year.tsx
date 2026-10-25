import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { Alert, StyleSheet, Text, TextInput, View } from 'react-native';

import {
  deleteYear,
  listYears,
  readStructure,
  saveYear,
  saveYearWithSchedule,
  type YearStatus,
} from '../../src/database/academicRepo';
import { Button, Chip, Notice, PageHeader, Screen, SectionHeader } from '../../src/ui/components';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

const STATUSES: { value: YearStatus; label: string }[] = [
  { value: 'past', label: 'Past' },
  { value: 'current', label: 'Current' },
  { value: 'future', label: 'Future' },
];

/** A year the student names themselves: "Grade 11", "2026–27", "Year 1". */
export default function YearEditor() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [name, setName] = useState('');
  const [label, setLabel] = useState('');
  const [grade, setGrade] = useState('');
  const [status, setStatus] = useState<YearStatus>('future');
  const [courseCount, setCourseCount] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      if (!id) return;
      const [years, structure] = await Promise.all([listYears(db), readStructure(db)]);
      const year = years.find((row) => row.id === id);
      if (!year) return;
      setName(year.name);
      setLabel(year.school_year_label ?? '');
      setGrade(year.grade_level ?? '');
      setStatus(year.status);
      const scheduleIds = structure.schedules
        .filter((schedule) => schedule.academic_year_id === id)
        .map((schedule) => schedule.id);
      setCourseCount(
        // Only used to make the delete confirmation honest.
        (await db.getFirstAsync<{ count: number }>(
          `SELECT COUNT(*) AS count FROM courses WHERE schedule_id IN (${scheduleIds
            .map(() => '?')
            .join(',') || "''"})`,
          scheduleIds,
        ))?.count ?? 0,
      );
    })();
  }, [db, id]);

  const valid = name.trim().length > 0;

  const save = async () => {
    if (!valid) return;
    setSaving(true);
    try {
      const input = {
        id,
        name: name.trim(),
        schoolYearLabel: label.trim() || null,
        gradeLevel: grade.trim() || null,
        status,
      };
      if (id) await saveYear(db, input);
      else await saveYearWithSchedule(db, input);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The year could not be saved.');
      setSaving(false);
    }
  };

  const remove = () => {
    if (!id) return;
    Alert.alert(
      'Delete year?',
      courseCount
        ? `Its ${courseCount} course${courseCount === 1 ? '' : 's'} will be kept and placed in an imported year.`
        : 'This year has no courses.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            void (async () => {
              await deleteYear(db, id);
              router.back();
            })();
          },
        },
      ],
    );
  };

  return (
    <Screen scroll edges="none">
      <PageHeader title={id ? 'Edit Year' : 'New Year'} onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
          <Notice tone="danger" body={error} />
        </View>
      ) : null}

      <View style={{ paddingHorizontal: space.xl, gap: space.md, paddingTop: space.sm }}>
        <Field label="Name" value={name} onChange={setName} placeholder="Grade 11" />
        <Field label="School year" value={label} onChange={setLabel} placeholder="2026-2027" />
        <Field label="Grade level" value={grade} onChange={setGrade} placeholder="11" />
      </View>

      <SectionHeader>STATUS</SectionHeader>
      <View style={{ flexDirection: 'row', gap: space.sm, paddingHorizontal: space.xl }}>
        {STATUSES.map((option) => (
          <Chip
            key={option.value}
            label={option.label}
            selected={status === option.value}
            onPress={() => setStatus(option.value)}
          />
        ))}
      </View>

      <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl, gap: space.md }}>
        <Button title="Save" onPress={() => void save()} disabled={!valid} loading={saving} />
        {id ? <Button title="Delete" kind="quiet" onPress={remove} /> : null}
      </View>
    </Screen>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const { palette } = useTheme();
  return (
    <View>
      <Text style={[type.caption, { color: palette.inkFaint, marginBottom: 4 }]}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor={palette.inkFaint}
        style={[
          styles.input,
          type.body,
          { color: palette.ink, borderColor: palette.line, backgroundColor: palette.surface },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  input: {
    minHeight: 44,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    borderRadius: radius.md,
    borderWidth: hairline,
  },
});
