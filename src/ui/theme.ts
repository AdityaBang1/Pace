/**
 * Pace's visual language: paper, ink, and one green.
 *
 * The product reads academic documents, so the surface imitates the material
 * (warm off-white, hairline rules, tabular numerals) rather than a dashboard.
 * There are no gradients and no glow: confidence is carried by weight and
 * position, not by colour effects.
 */
import { Platform, TextStyle } from 'react-native';

export interface Palette {
  background: string;
  surface: string;
  surfaceSunken: string;
  ink: string;
  inkMuted: string;
  inkFaint: string;
  line: string;
  lineStrong: string;
  accent: string;
  accentInk: string;
  accentSoft: string;
  warn: string;
  warnSoft: string;
  danger: string;
  dangerSoft: string;
  high: string;
  medium: string;
  low: string;
  overlay: string;
}

/** Warm paper in light, near-black slate in dark: the same hierarchy, inverted. */
export const light: Palette = {
  background: '#FBFAF7',
  surface: '#FFFFFF',
  surfaceSunken: '#F3F0EA',
  ink: '#15171A',
  inkMuted: '#63686E',
  inkFaint: '#93999F',
  line: '#E7E2D9',
  lineStrong: '#D6D0C4',
  accent: '#2E5B4C',
  accentInk: '#FFFFFF',
  accentSoft: '#E4EDE8',
  warn: '#96601A',
  warnSoft: '#F7EDDD',
  danger: '#9C3B2E',
  dangerSoft: '#F8E7E3',
  high: '#2E5B4C',
  medium: '#8A6413',
  low: '#8C4A3F',
  overlay: 'rgba(16, 17, 20, 0.42)',
};

export const dark: Palette = {
  background: '#0E0F11',
  surface: '#17191C',
  surfaceSunken: '#121316',
  ink: '#F1EFEA',
  inkMuted: '#A0A5AB',
  inkFaint: '#71767C',
  line: '#262A2D',
  lineStrong: '#33383C',
  accent: '#8FBFAE',
  accentInk: '#10231D',
  accentSoft: '#1B2A25',
  warn: '#D9A667',
  warnSoft: '#2A2217',
  danger: '#E09284',
  dangerSoft: '#2C1D1A',
  high: '#8FBFAE',
  medium: '#D9A667',
  low: '#E09284',
  overlay: 'rgba(0, 0, 0, 0.55)',
};

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
} as const;

export const space = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 44,
} as const;

/** System font, tuned per platform, with optical size where the OS supports it. */
export const type = {
  display: {
    fontSize: 34,
    lineHeight: 40,
    fontWeight: '600',
    letterSpacing: -0.9,
  } as TextStyle,
  title: {
    fontSize: 24,
    lineHeight: 30,
    fontWeight: '600',
    letterSpacing: -0.5,
  } as TextStyle,
  heading: {
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '600',
    letterSpacing: -0.2,
  } as TextStyle,
  body: {
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '400',
  } as TextStyle,
  bodyStrong: {
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '600',
  } as TextStyle,
  label: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '600',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  } as TextStyle,
  caption: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '400',
  } as TextStyle,
  number: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
  } as TextStyle,
  mono: {
    fontSize: 13,
    lineHeight: 19,
    fontWeight: '400',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
  } as TextStyle,
};

/** Hairlines stay hairlines on every screen density. */
export const hairline = Platform.select({ ios: 0.5, default: 1 }) as number;
