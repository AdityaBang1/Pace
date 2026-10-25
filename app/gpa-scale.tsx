import { Button as NativeButton, LabeledContent, Section, TextField, type TextFieldRef } from '@expo/ui/swift-ui';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';

import { readGpaScaleValue, writeGpaScaleValue } from '../src/database/repo';
import { DEFAULT_SCALE, LETTERS, LEVELS, parseScale, type GpaScale } from '../src/domain/gpa';
import { NativeScreen } from '../src/ui/native';

type LevelKey = 'regular' | 'honors' | 'ap';

/**
 * The points a letter is worth, per course level.
 *
 * It is a form of numbers, so it is a system form: one section per level, one row
 * per letter. The fields are native text fields — prefilled from the stored scale
 * through their ref, because a SwiftUI field owns its own text — and the value the
 * screen saves is the same parsed object as before.
 */
export default function GpaScaleScreen() {
  const db = useSQLiteContext();
  const router = useRouter();
  const [scale, setScale] = useState<GpaScale>(DEFAULT_SCALE);
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const fields = useRef<Record<string, TextFieldRef | null>>({});
  const keyOf = (level: LevelKey, letter: string) => `${level}-${letter}`;

  const writeFields = (next: GpaScale) => {
    for (const level of LEVELS) {
      for (const letter of LETTERS) {
        const field = fields.current[keyOf(level.value as LevelKey, letter)];
        void field?.setText(String(next[level.value as LevelKey][letter] ?? 0));
      }
    }
  };

  useEffect(() => {
    void (async () => {
      const stored = parseScale(await readGpaScaleValue(db)) ?? DEFAULT_SCALE;
      setScale(stored);
      setLoaded(true);
    })();
  }, [db]);

  // A field cannot be prefilled before it exists, so the stored scale is written
  // into the fields once they have mounted.
  useEffect(() => {
    if (!loaded) return;
    writeFields(scale);
  }, [loaded]);

  const setPoint = (level: LevelKey, letter: string, text: string) => {
    const value = Number(text.replace(',', '.'));
    setScale((current) => ({
      ...current,
      [level]: {
        ...current[level],
        [letter]: Number.isFinite(value) ? value : 0,
      },
    }));
  };

  const save = async (next: GpaScale) => {
    if (saving) return;
    setSaving(true);
    await writeGpaScaleValue(db, JSON.stringify(next));
    setSaving(false);
    router.back();
  };

  return (
    <NativeScreen title="GPA Scale" onBack={() => router.back()} topInset={false}>
      {LEVELS.map((level) => (
        <Section key={level.value} title={level.label}>
          {LETTERS.map((letter) => (
            <LabeledContent key={letter} label={letter}>
              <TextField
                ref={(instance) => {
                  fields.current[keyOf(level.value as LevelKey, letter)] = instance;
                }}
                placeholder="0"
                onTextChange={(text) => setPoint(level.value as LevelKey, letter, text)}
              />
            </LabeledContent>
          ))}
        </Section>
      ))}

      <Section>
        <NativeButton label={saving ? 'Saving…' : 'Save'} onPress={() => void save(scale)} />
        <NativeButton label="Use Default" onPress={() => void save(DEFAULT_SCALE)} />
      </Section>
    </NativeScreen>
  );
}
