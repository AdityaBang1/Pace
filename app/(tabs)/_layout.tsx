import { NativeTabs } from 'expo-router/native-tabs';

/**
 * The three destinations. Native, system-drawn tabs: the platform owns the bar's
 * height, blur, press behaviour and accessibility, so the app looks like a phone
 * app rather than a web page with a bar drawn on it.
 */
export default function TabsLayout() {
  return (
    <NativeTabs>
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
