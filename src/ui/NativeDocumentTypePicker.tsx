/**
 * The Android (and web) counterpart of the SwiftUI picker: the same chips the
 * review screen used before, so the control exists on every platform.
 */
import { Text, View } from 'react-native';

import { Chip } from './components';
import { space, type } from './theme';
import { useTheme } from './useTheme';

export const hasNativeTypePicker = false;

export function NativeDocumentTypePicker({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { label: string; value: string }[];
  onChange: (value: string) => void;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ gap: space.sm }}>
      <Text style={[type.label, { color: palette.inkFaint }]}>{label}</Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
        {options.map((option) => (
          <Chip
            key={option.value}
            label={option.label}
            tone={option.value === value ? 'accent' : 'neutral'}
            selected={option.value === value}
            onPress={() => option.value !== value && onChange(option.value)}
          />
        ))}
      </View>
    </View>
  );
}
