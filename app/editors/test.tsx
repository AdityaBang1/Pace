import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { Alert, StyleSheet, Text, TextInput, View } from 'react-native';

import { readStructure, saveTestScore, deleteTestScore, type TestSection } from '../../src/database/academicRepo';
import { Button, Chip, Notice, PageHeader, Screen, SectionHeader } from '../../src/ui/components';
import { hairline, radius, space, type } from '../../src/ui/theme';
import { useTheme } from '../../src/ui/useTheme';

const KINDS = ['SAT', 'ACT', 'PSAT', 'AP', 'Custom'] as const;

/** A score, typed once. Nothing here is inferred from the record. */
export default function TestEditor() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();

  const [kind, setKind] = useState<string>('SAT');
  const [name, setName] = useState('');
  const [takenOn, setTakenOn] = useState('');
  const [total, setTotal] = useState('');
  const [max, setMax] = useState('');
  const [sections, setSections] = useState<TestSection[]>([
    { name: '', score: '' },
    { name: '', score: '' },
    { name: '', score: '' },
  ]);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    void (async () => {
      const structure = await readStructure(db);
      const test = structure.tests.find((row) => row.id === id);
      if (!test) return;
      setKind(test.kind);
      setName(test.name);
      setTakenOn(test.taken_on ?? '');
      setTotal(test.total_score ?? '');
      setMax(test.max_score ?? '');
      setNotes(test.notes ?? '');
      try {
        const parsed: unknown = JSON.parse(test.sections);
        if (Array.isArray(parsed)) {
          const rows = (parsed as TestSection[]).filter((row) => row && typeof row.name === 'string');
          setSections([...rows, ...Array(Math.max(0, 3 - rows.length)).fill({ name: '', score: '' })].slice(0, 3));
        }
      } catch {
        // A score with unreadable sections is still editable; the fields stay empty.
      }
    })();
  }, [db, id]);

  const valid = name.trim().length > 0;

  const save = async () => {
    if (!valid) return;
    setSaving(true);
    try {
      await saveTestScore(db, {
        id,
        kind,
        name: name.trim(),
        takenOn: takenOn.trim() || null,
        totalScore: total.trim() || null,
        maxScore: max.trim() || null,
        sections: sections.filter((row) => row.name.trim() || row.score.trim()),
        notes: notes.trim() || null,
      });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The score could not be saved.');
      setSaving(false);
    }
  };

  const remove = () => {
    if (!id) return;
    Alert.alert('Delete score?', name || 'This score', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deleteTestScore(db, id);
            router.back();
          })();
        },
      },
    ]);
  };

  return (
    <Screen scroll edges="none">
      <PageHeader title={id ? 'Edit Score' : 'New Score'} onBack={() => router.back()} />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.md }}>
          <Notice tone="danger" body={error} />
        </View>
      ) : null}

      <SectionHeader>TYPE</SectionHeader>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm, paddingHorizontal: space.xl }}>
        {KINDS.map((option) => (
          <Chip
            key={option}
            label={option}
            selected={kind === option}
            onPress={() => {
              setKind(option);
              if (!name.trim() || KINDS.includes(name.trim() as (typeof KINDS)[number])) setName(option);
            }}
          />
        ))}
      </View>

      <SectionHeader>SCORE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Field label="Name" value={name} onChange={setName} placeholder="SAT" />
        <Field label="Date" value={takenOn} onChange={setTakenOn} placeholder="2026-10-05" />
        <View style={{ flexDirection: 'row', gap: space.md }}>
          <View style={{ flex: 1 }}>
            <Field label="Total" value={total} onChange={setTotal} keyboard="number-pad" />
          </View>
          <View style={{ flex: 1 }}>
            <Field label="Out of" value={max} onChange={setMax} keyboard="number-pad" />
          </View>
        </View>
      </View>

      <SectionHeader>SECTIONS</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        {sections.map((section, index) => (
          <View key={`section-${index}`} style={{ flexDirection: 'row', gap: space.md }}>
            <View style={{ flex: 2 }}>
              <Field
                label={index === 0 ? 'Section' : ' '}
                value={section.name}
                onChange={(value) =>
                  setSections(sections.map((row, at) => (at === index ? { ...row, name: value } : row)))
                }
                placeholder="Math"
              />
            </View>
            <View style={{ flex: 1 }}>
              <Field
                label={index === 0 ? 'Score' : ' '}
                value={section.score}
                onChange={(value) =>
                  setSections(sections.map((row, at) => (at === index ? { ...row, score: value } : row)))
                }
                keyboard="number-pad"
              />
            </View>
          </View>
        ))}
      </View>

      <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg, gap: space.md }}>
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
  keyboard,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  keyboard?: 'default' | 'number-pad';
}) {
  const { palette } = useTheme();
  return (
    <View>
      <Text style={[type.caption, { color: palette.inkFaint, marginBottom: 4 }]} numberOfLines={1}>
        {label}
      </Text>
      <TextInput
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor={palette.inkFaint}
        keyboardType={keyboard === 'number-pad' ? 'number-pad' : 'default'}
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
