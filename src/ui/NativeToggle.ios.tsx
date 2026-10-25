/**
 * A real SwiftUI `Toggle` through @expo/ui.
 *
 * SwiftUI lays this out, not Yoga: the `Host` is the React Native view and its
 * style is the only place React Native sizing belongs. The height is explicit
 * because `matchContents` must not be used on the same axis as a scroll container,
 * and Settings is a scroll view.
 */
import { Host, Toggle } from '@expo/ui/swift-ui';

import { useTheme } from './useTheme';

export const hasNativeToggle = true;

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
    <Host style={{ height: 44, alignSelf: 'stretch' }} seedColor={palette.accent}>
      <Toggle label={label} isOn={value} onIsOnChange={onValueChange} />
    </Host>
  );
}
