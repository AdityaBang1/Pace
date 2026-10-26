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
  deleteYear,
  listYears,
  readStructure,
  saveYear,
  saveYearWithSchedule,
  type YearStatus,
} from '../../src/database/academicRepo';
import { NativeScreen, NativeText, Section } from '../../src/ui/native';
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

  const nameRef = useRef<TextFieldRef>(null);
  const labelRef = useRef<TextFieldRef>(null);
  const gradeRef = useRef<TextFieldRef>(null);

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
      void nameRef.current?.setText(year.name);
      void labelRef.current?.setText(year.school_year_label ?? '');
      void gradeRef.current?.setText(year.grade_level ?? '');
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
    <NativeScreen
      topInset={false}
      title={id ? 'Edit Year' : 'New Year'}
      onBack={() => router.back()}
    >
      {error ? (
        <Section>
          <NativeText modifiers={[foregroundStyle(palette.danger)]}>{error}</NativeText>
        </Section>
      ) : null}

      <Section title="Name">
        <TextField ref={nameRef} placeholder="Grade 11" onTextChange={setName} />
      </Section>

      <Section title="School year">
        <TextField ref={labelRef} placeholder="2026-2027" onTextChange={setLabel} />
      </Section>

      <Section title="Grade level">
        <TextField ref={gradeRef} placeholder="11" onTextChange={setGrade} />
      </Section>

      <Section title="Status">
        {STATUSES.map((option) => (
          <NativeButton
            key={option.value}
            label={option.label}
            onPress={() => setStatus(option.value)}
            systemImage={status === option.value ? 'checkmark' : undefined}
          />
        ))}
      </Section>

      <Section>
        <NativeButton
          label="Save"
          systemImage="checkmark"
          modifiers={[disabled(!valid || saving)]}
          onPress={() => void save()}
        />
        {id ? <NativeButton label="Delete" role="destructive" onPress={remove} /> : null}
        {saving ? <NativeText>Saving…</NativeText> : null}
      </Section>
    </NativeScreen>
  );
}
