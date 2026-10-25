import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';

import {
  Button as NativeButton,
  Picker,
  Section,
  Text as NativeText,
  TextField,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import { disabled, tag } from '@expo/ui/swift-ui/modifiers';

import {
  deleteTestScore,
  readStructure,
  saveTestScore,
  type TestSection,
} from '../../src/database/academicRepo';
import { NativeScreen } from '../../src/ui/native';

const KINDS = ['SAT', 'ACT', 'PSAT', 'AP', 'Custom'] as const;

/** A score, typed once. Nothing here is inferred from the record. */
export default function TestEditor() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const db = useSQLiteContext();
  const router = useRouter();

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
  const [hydrated, setHydrated] = useState(false);

  const nameRef = useRef<TextFieldRef>(null);
  const dateRef = useRef<TextFieldRef>(null);
  const totalRef = useRef<TextFieldRef>(null);
  const maxRef = useRef<TextFieldRef>(null);
  const notesRef = useRef<TextFieldRef>(null);
  const sectionRefs = [
    useRef<TextFieldRef>(null),
    useRef<TextFieldRef>(null),
    useRef<TextFieldRef>(null),
  ];
  const scoreRefs = [
    useRef<TextFieldRef>(null),
    useRef<TextFieldRef>(null),
    useRef<TextFieldRef>(null),
  ];

  useEffect(() => {
    if (!id) {
      setHydrated(true);
      return;
    }
    void (async () => {
      const structure = await readStructure(db);
      const test = structure.tests.find((row) => row.id === id);
      if (!test) {
        setHydrated(true);
        return;
      }
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
      setHydrated(true);
    })();
  }, [db, id]);

  // Push the loaded values into the native fields once; after that the fields report
  // every keystroke and the state above stays the source of truth for saving.
  useEffect(() => {
    if (!hydrated) return;
    void nameRef.current?.setText(name);
    void dateRef.current?.setText(takenOn);
    void totalRef.current?.setText(total);
    void maxRef.current?.setText(max);
    void notesRef.current?.setText(notes);
    sections.forEach((row, index) => {
      void sectionRefs[index].current?.setText(row.name);
      void scoreRefs[index].current?.setText(row.score);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated]);

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

  const updateSection = (index: number, patch: Partial<TestSection>) => {
    setSections((current) => current.map((row, at) => (at === index ? { ...row, ...patch } : row)));
  };

  return (
    <NativeScreen topInset={false} title={id ? 'Edit Score' : 'New Score'} onBack={() => router.back()}>
      {error ? (
        <Section title="Error">
          <NativeText>{error}</NativeText>
        </Section>
      ) : null}

      <Section title="Type">
        <Picker
          label="Kind"
          selection={kind}
          onSelectionChange={(value) => {
            const next = String(value);
            setKind(next);
            const trimmed = name.trim();
            if (!trimmed || (KINDS as readonly string[]).includes(trimmed)) {
              setName(next);
              void nameRef.current?.setText(next);
            }
          }}
        >
          {KINDS.map((option) => (
            <NativeText key={option} modifiers={[tag(option)]}>
              {option}
            </NativeText>
          ))}
        </Picker>
      </Section>

      <Section title="Score">
        <TextField ref={nameRef} placeholder="SAT" onTextChange={setName} />
        <TextField ref={dateRef} placeholder="2026-10-05" onTextChange={setTakenOn} />
        <TextField ref={totalRef} placeholder="Total" onTextChange={setTotal} />
        <TextField ref={maxRef} placeholder="Out of" onTextChange={setMax} />
      </Section>

      {sections.map((section, index) => (
        <Section key={`section-${index}`} title={`Section ${index + 1}`}>
          <TextField
            ref={sectionRefs[index]}
            placeholder="Math"
            onTextChange={(value) => updateSection(index, { name: value })}
          />
          <TextField
            ref={scoreRefs[index]}
            placeholder="Score"
            onTextChange={(value) => updateSection(index, { score: value })}
          />
        </Section>
      ))}

      <Section title="Notes">
        <TextField ref={notesRef} placeholder="Optional" onTextChange={setNotes} />
      </Section>

      <Section>
        <NativeButton
          label={saving ? 'Saving…' : 'Save'}
          onPress={() => void save()}
          modifiers={[disabled(!valid || saving)]}
        />
      </Section>

      {id ? (
        <Section>
          <NativeButton label="Delete Score" role="destructive" systemImage="trash" onPress={remove} />
        </Section>
      ) : null}
    </NativeScreen>
  );
}
