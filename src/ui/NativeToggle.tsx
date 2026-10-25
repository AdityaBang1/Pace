/**
 * The Android (and web) counterpart of the SwiftUI toggle.
 *
 * Same props, same behaviour, rendered with React Native's own `Switch` so the
 * Settings screen works everywhere without importing @expo/ui on a platform that
 * does not have it.
 */
import { Switch, Text, View } from 'react-native';

import { space, type } from './theme';
import { useTheme } from './useTheme';

export const hasNativeToggle = false;

export function NativeToggle({
  label,
  value,
  onValueChange,
}: {
  label: string;
  value: boolean;
  onValueChange: (next: boolean) => void;
}) {
  const { palette } = useTheme();
  return (
    <View
      style={{
        minHeight: 44,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: space.md,
      }}
    >
      <Text style={[type.body, { color: palette.ink, flexShrink: 1 }]}>{label}</Text>
      <Switch
        value={value}
        onValueChange={onValueChange}
        trackColor={{ true: palette.accent, false: palette.lineStrong }}
        accessibilityLabel={label}
      />
    </View>
  );
}
