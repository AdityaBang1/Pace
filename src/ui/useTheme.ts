import { useColorScheme } from 'react-native';

import { dark, light, type Palette } from './theme';

export interface Theme {
  palette: Palette;
  dark: boolean;
}

/**
 * System appearance, no override.
 *
 * A manual light/dark switch was deliberately not added: the app follows the
 * phone, which is what a native iOS app is expected to do, and one less setting
 * is one less way for the review screen to look wrong.
 */
export function useTheme(): Theme {
  const scheme = useColorScheme();
  const isDark = scheme === 'dark';
  return { palette: isDark ? dark : light, dark: isDark };
}
