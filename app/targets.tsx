import { useSQLiteContext } from 'expo-sqlite';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Text, TextInput, View } from 'react-native';

import { getSetting, setSetting } from '../src/database/repo';
import { CREDIT_TARGET_KEY, GPA_TARGET_KEY } from '../src/domain/pace';
import { Button, PageHeader, SectionHeader, Screen } from '../src/ui/components';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

/**
 * Targets are the student's own numbers. Pace compares a projection against them
 * and reports the difference; it never supplies a target of its own, which is why
 * both fields start empty and an empty field means "no target", not zero.
 */
export default function Targets() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [gpa, setGpa] = useState('');
  const [credits, setCredits] = useState('');
  const [saving, setSaving] = useState(false);

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
    })();
    return () => {
      alive = false;
    };
  }, [db]);

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

  const input = {
    color: palette.ink,
    borderColor: palette.line,
    backgroundColor: palette.surfaceSunken,
    borderWidth: hairline,
    borderRadius: radius.md,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    minHeight: 44,
  } as const;

  return (
    <Screen scroll edges="none">
      <PageHeader title="Targets" onBack={() => router.back()} />
      <SectionHeader>GPA</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <TextInput
          value={gpa}
          onChangeText={setGpa}
          placeholder="4.20"
          placeholderTextColor={palette.inkFaint}
          keyboardType="decimal-pad"
          accessibilityLabel="Weighted GPA target"
          style={[type.number, input]}
        />
      </View>
      <SectionHeader>Credits</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <TextInput
          value={credits}
          onChangeText={setCredits}
          placeholder="24"
          placeholderTextColor={palette.inkFaint}
          keyboardType="decimal-pad"
          accessibilityLabel="Credit target"
          style={[type.number, input]}
        />
      </View>
      <View style={{ paddingHorizontal: space.xl, paddingTop: space.xl }}>
        <Button title="Save" onPress={() => void save()} loading={saving} />
      </View>
      <View style={{ paddingHorizontal: space.xl, paddingTop: space.lg }}>
        <Text style={[type.caption, { color: palette.inkFaint }]}>
          {gpa.trim() || credits.trim() ? 'Pace compares your path to these.' : 'No targets set.'}
        </Text>
      </View>
    </Screen>
  );
}
