import { NativeTabs } from 'expo-router/native-tabs';

import { useTheme } from '../../src/ui/useTheme';

/**
 * The three destinations. Native, system-drawn tabs: the platform owns the bar's
 * height, blur, press behaviour and accessibility, so the app looks like a phone
 * app rather than a web page with a bar drawn on it.
 *
 * The bar is told the app's one colour. Left alone it tints the selected tab with
 * the system blue, which is the only blue in an app made of paper, ink and green —
 * and it reads as a control borrowed from somewhere else.
 */
export default function TabsLayout() {
  const { palette } = useTheme();
  return (
    <NativeTabs
      tintColor={palette.accent}
      iconColor={{ default: palette.inkFaint, selected: palette.accent }}
      backgroundColor={palette.background}
    >
      <NativeTabs.Trigger name="index">
        <NativeTabs.Trigger.Icon sf="house.fill" />
        <NativeTabs.Trigger.Label>Home</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="academics">
        <NativeTabs.Trigger.Icon sf="books.vertical.fill" />
        <NativeTabs.Trigger.Label>Academics</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="plan">
        <NativeTabs.Trigger.Icon sf="map.fill" />
        <NativeTabs.Trigger.Label>Plan</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}
