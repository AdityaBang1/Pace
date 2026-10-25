import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';

import { readGpaScaleValue, writeGpaScaleValue } from '../src/database/repo';
import {
  DEFAULT_SCALE,
  LETTERS,
  LEVELS,
  parseScale,
  type GpaScale,
} from '../src/domain/gpa';
import { Button, Card, Divider, PageHeader, Screen } from '../src/ui/components';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

type LevelKey = keyof GpaScale;

export default function GpaScaleScreen() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const [scale, setScale] = useState<GpaScale>(DEFAULT_SCALE);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void (async () => {
      setScale(parseScale(await readGpaScaleValue(db)) ?? DEFAULT_SCALE);
    })();
  }, [db]);

  const setPoint = (level: LevelKey, letter: string, text: string) => {
    const value = Number(text.replace(',', '.'));
    setScale({
      ...scale,
      [level]: {
        ...scale[level],
        [letter]: Number.isFinite(value) ? value : 0,
      },
    });
  };

  const save = async (next: GpaScale) => {
    setSaving(true);
    await writeGpaScaleValue(db, JSON.stringify(next));
    setSaving(false);
    router.back();
  };

  return (
    <Screen scroll>
      <PageHeader title="GPA Scale" onBack={() => router.back()} />

      {LEVELS.map((level) => (
        <View key={level.value} style={{ paddingHorizontal: space.xl, marginBottom: space.lg }}>
          <View style={{ flexDirection: 'row', alignItems: 'baseline', marginBottom: space.sm }}>
            <Text style={[type.label, { color: palette.inkFaint }]}>{level.label}</Text>
          </View>
          <Card padded={false}>
            {LETTERS.map((letter, index) => (
              <View key={letter}>
                {index > 0 ? <Divider inset={space.lg} /> : null}
                <View style={styles.row}>
                  <Text style={[type.body, { color: palette.ink }]}>{letter}</Text>
                  <TextInput
                    value={String(scale[level.value as LevelKey][letter] ?? 0)}
                    onChangeText={(text) => setPoint(level.value as LevelKey, letter, text)}
                    keyboardType="decimal-pad"
                    style={[
                      type.number,
                      styles.input,
                      {
                        color: palette.ink,
                        borderColor: palette.line,
                        backgroundColor: palette.surfaceSunken,
                      },
                    ]}
                  />
                </View>
              </View>
            ))}
          </Card>
        </View>
      ))}

      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Button title="Save" onPress={() => void save(scale)} loading={saving} />
        <Button title="Use Default" kind="quiet" onPress={() => void save(DEFAULT_SCALE)} />
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: space.sm,
    paddingHorizontal: space.lg,
    minHeight: 44,
  },
  input: {
    minWidth: 72,
    textAlign: 'right',
    paddingVertical: 6,
    paddingHorizontal: space.sm,
    borderRadius: radius.sm,
    borderWidth: hairline,
  },
});
