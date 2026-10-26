/**
 * The scaffold for a screen drawn by SwiftUI.
 *
 * A settings-shaped screen should be a system form: grouped sections, system rows,
 * system switches and buttons. The title and the way back stay ours, because the
 * app's root stack has no native navigation bar; everything below them is SwiftUI,
 * hosted in a form that fills the rest of the screen.
 *
 * Plain `@expo/ui/swift-ui` primitives are re-exported so a screen imports one
 * module rather than two.
 */
import {
  Form,
  Host,
  Section,
  Text as NativeText,
  LabeledContent,
} from '@expo/ui/swift-ui';
import {
  background,
  listRowBackground,
  listSectionSpacing,
  scrollContentBackground,
  tint,
} from '@expo/ui/swift-ui/modifiers';
import { ReactNode } from 'react';
import { View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { NativeHeader, type HeaderIconAction, type HeaderPrimaryAction } from './header';
import { useTheme } from './useTheme';

export { Form, Host, Section, NativeText, LabeledContent };
export { NativeHeader } from './header';
export type { HeaderIconAction, HeaderPrimaryAction } from './header';

export function NativeScreen({
  title,
  subtitle,
  onBack,
  actions,
  primary,
  children,
  topInset = true,
}: {
  title?: string;
  subtitle?: string;
  onBack?: () => void;
  /** Trailing glyph actions, drawn as one system pill. */
  actions?: HeaderIconAction[];
  /** The one filled action a form sheet carries, usually Save. */
  primary?: HeaderPrimaryAction;
  children: ReactNode;
  /**
   * A pushed screen owns the status-bar inset; a sheet does not, because the sheet
   * already begins below the status bar and adding it again leaves a hand's width
   * of empty paper above the title.
   */
  topInset?: boolean;
}) {
  const { palette } = useTheme();
  const insets = useSafeAreaInsets();
  return (
    <View
      style={{
        flex: 1,
        backgroundColor: palette.background,
        paddingTop: topInset ? insets.top : 0,
      }}
    >
      {title ? (
        <NativeHeader
          title={title}
          subtitle={subtitle}
          onBack={onBack}
          actions={actions}
          primary={primary}
        />
      ) : null}
      {/*
        The form is SwiftUI's, but it is not left as the stock grouped list: the
        system paints a cool grey page, blue controls and blue switches, and that
        is the generic look this app is not. The paper colour comes from our
        palette, rows sit on the surface colour, sections are spaced by the same
        rhythm the hand-drawn screens used, and every control that takes a tint —
        switch, button, picker checkmark — takes the one green.
      */}
      <Host style={{ flex: 1 }} useViewportSizeMeasurement seedColor={palette.accent}>
        <Form
          modifiers={[
            scrollContentBackground('hidden'),
            background(palette.background),
            listRowBackground(palette.surface),
            tint(palette.accent),
            listSectionSpacing(20),
          ]}
        >
          {children}
        </Form>
      </Host>
    </View>
  );
}

/**
 * A value on the right of a row. SwiftUI needs its own `Text`, not a React Native
 * one; a string is wrapped here so a screen can pass the common case directly.
 */
export function Value({ children }: { children: ReactNode | string }) {
  if (typeof children === 'string' || typeof children === 'number') {
    return <NativeText>{String(children)}</NativeText>;
  }
  return <>{children}</>;
}
