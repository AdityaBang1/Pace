/**
 * Document type as a native SwiftUI menu.
 *
 * The gate's label is a proposal like everything else, and the person holding the
 * page is allowed to disagree. A menu picker is the native way to offer that, and
 * it stays a single, quiet control instead of a row of chips.
 */
import { Host, Picker, Text } from '@expo/ui/swift-ui';
import { pickerStyle, tag } from '@expo/ui/swift-ui/modifiers';

import { useTheme } from './useTheme';

export const hasNativeTypePicker = true;

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
    <Host style={{ height: 40, alignSelf: 'stretch' }} seedColor={palette.accent}>
      <Picker
        label={label}
        selection={value}
        onSelectionChange={(next) => {
          if (typeof next === 'string' && next !== value) onChange(next);
        }}
        modifiers={[pickerStyle('menu')]}
      >
        {options.map((option) => (
          <Text key={option.value} modifiers={[tag(option.value)]}>
            {option.label}
          </Text>
        ))}
      </Picker>
    </Host>
  );
}
