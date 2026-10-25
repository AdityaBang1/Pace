import { Platform } from 'react-native';

import { SettingsPortable } from '../src/features/settings/SettingsPortable';

/**
 * iOS gets the SwiftUI form; everywhere else the portable screen draws the same
 * rows with the app's own primitives. The platform decides the drawing, never the
 * behaviour: both call `useSettings`.
 */
export default function Settings() {
  if (Platform.OS === 'ios') {
    const { SettingsNative } = require('../src/features/settings/SettingsNative.ios') as typeof import('../src/features/settings/SettingsNative.ios');
    return <SettingsNative />;
  }
  return <SettingsPortable />;
}
