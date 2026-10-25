import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';

import {
  Button as NativeButton,
  Picker,
  Section,
  Text as NativeText,
  TextField,
  Toggle,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import { disabled, tag } from '@expo/ui/swift-ui/modifiers';

import {
  deletePathway,
  ensureDefaults,
  readStructure,
  savePathway,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { NativeScreen } from '../../src/ui/native';

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

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [name, setName] = useState('');
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [active, setActive] = useState(false);
  const [firstPathway, setFirstPathway] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  const nameRef = useRef<TextFieldRef>(null);

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
      setHydrated(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not read its local database.');
    }
  }, [db, id]);

  useEffect(() => {
    void load();
  }, [load]);

  // A SwiftUI field owns its text; the value loaded from the database is pushed into
  // it once, then reported back on every change.
  useEffect(() => {
    if (!hydrated) return;
    void nameRef.current?.setText(name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated]);

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
    <NativeScreen
      topInset={false}
      title={id ? 'Pathway' : 'New Pathway'}
      onBack={() => router.back()}
    >
      {error ? (
        <Section title="Error">
          <NativeText>{error}</NativeText>
          <NativeButton label="Try Again" systemImage="arrow.clockwise" onPress={() => void load()} />
        </Section>
      ) : null}

      <Section title="Name">
        <TextField ref={nameRef} placeholder="Main" autoFocus={!id} onTextChange={setName} />
      </Section>

      <Section title="Pathway" footer={<NativeText>Home and Pace report on the active pathway.</NativeText>}>
        <Toggle
          label={firstPathway ? 'Active · the only pathway' : 'Active'}
          isOn={active || firstPathway}
          onIsOnChange={(next) => {
            if (firstPathway) return;
            setActive(next);
          }}
        />
      </Section>

      {years.length ? (
        <Section title="Years" footer={<NativeText>Main keeps that year's own schedule.</NativeText>}>
          {years.map((year) => {
            const yearSchedules = schedules.filter((schedule) => schedule.academic_year_id === year.id);
            const chosen = selections[year.id] ?? 'main';
            return (
              <Picker
                key={year.id}
                label={year.name}
                selection={chosen}
                onSelectionChange={(value) => {
                  setSelections((current) => {
                    const next = { ...current };
                    if (value === 'main' || value == null) delete next[year.id];
                    else next[year.id] = String(value);
                    return next;
                  });
                }}
              >
                <NativeText modifiers={[tag('main')]}>Main</NativeText>
                {yearSchedules.map((schedule) => (
                  <NativeText key={schedule.id} modifiers={[tag(schedule.id)]}>
                    {schedule.name}
                  </NativeText>
                ))}
              </Picker>
            );
          })}
        </Section>
      ) : (
        <Section title="Years">
          <NativeText>Import a transcript or add a year first.</NativeText>
        </Section>
      )}

      <Section>
        <NativeButton
          label={saving ? 'Saving…' : 'Save'}
          onPress={() => void save()}
          modifiers={[disabled(!name.trim() || saving)]}
        />
      </Section>

      {id ? (
        <Section>
          <NativeButton label="Delete Pathway" role="destructive" systemImage="trash" onPress={remove} />
        </Section>
      ) : null}
    </NativeScreen>
  );
}
