import { Button as NativeButton, LabeledContent, Section, TextField, type TextFieldRef } from '@expo/ui/swift-ui';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';

import { getSetting, setSetting } from '../src/database/repo';
import { CREDIT_TARGET_KEY, GPA_TARGET_KEY } from '../src/domain/pace';
import { NativeScreen, NativeText } from '../src/ui/native';

/**
 * Targets are the student's own numbers. Pace compares a projection against them
 * and reports the difference; it never supplies a target of its own, which is why
 * both fields start empty and an empty field means "no target", not zero.
 */
export default function Targets() {
  const db = useSQLiteContext();
  const router = useRouter();
  const [gpa, setGpa] = useState('');
  const [credits, setCredits] = useState('');
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const gpaField = useRef<TextFieldRef>(null);
  const creditField = useRef<TextFieldRef>(null);

  useEffect(() => {
    let alive = true;
    void (async () => {
      const [storedGpa, storedCredits] = await Promise.all([
        getSetting(db, GPA_TARGET_KEY, ''),
        getSetting(db, CREDIT_TARGET_KEY, ''),
      ]);
      if (!alive) return;
      setGpa(storedGpa);
      setCredits(storedCredits);
      setLoaded(true);
    })();
    return () => {
      alive = false;
    };
  }, [db]);

  // The fields own their text, so a stored value is written into them once they exist.
  useEffect(() => {
    if (!loaded) return;
    void gpaField.current?.setText(gpa);
    void creditField.current?.setText(credits);
  }, [loaded]);

  const save = async () => {
    setSaving(true);
    try {
      await setSetting(db, GPA_TARGET_KEY, gpa.trim());
      await setSetting(db, CREDIT_TARGET_KEY, credits.trim());
      router.back();
    } finally {
      setSaving(false);
    }
  };

  const hasTarget = gpa.trim() !== '' || credits.trim() !== '';

  return (
    <NativeScreen
      title="Targets"
      onBack={() => router.back()}
      topInset={false}
    >
      <Section title="GPA">
        <LabeledContent label="Weighted">
          <TextField
            ref={gpaField}
            placeholder="4.20"
            onTextChange={setGpa}
          />
        </LabeledContent>
      </Section>

      <Section title="Credits">
        <LabeledContent label="Total">
          <TextField ref={creditField} placeholder="24" onTextChange={setCredits} />
        </LabeledContent>
      </Section>

      <Section
        footer={
          <NativeText>
            {hasTarget ? 'Pace compares your path to these.' : 'No targets set.'}
          </NativeText>
        }
      >
        <NativeButton label={saving ? 'Saving…' : 'Save'} onPress={() => void save()} />
      </Section>
    </NativeScreen>
  );
}
