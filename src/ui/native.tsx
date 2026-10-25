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
  Button as SystemButton,
  Form,
  Host,
  HStack,
  Image as SystemImage,
  Section,
  Text as NativeText,
  LabeledContent,
} from '@expo/ui/swift-ui';
import {
  accessibilityLabel,
  buttonStyle,
  glassEffect,
  padding,
} from '@expo/ui/swift-ui/modifiers';
import { ReactNode } from 'react';
import { View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { PageHeader } from './components';
import { useTheme } from './useTheme';

export { Form, Host, Section, NativeText, LabeledContent };

/**
 * The screen's top-trailing actions, drawn by SwiftUI as one glass pill.
 *
 * The buttons are real system buttons — the platform owns their press highlight,
 * their accessibility traits and their sizing — and they share a single capsule so
 * two small glyphs read as one control rather than two loose icons floating in the
 * status bar. `Host` is asked to match its content so the row is as wide as the
 * pill and nothing wider.
 */
export function HeaderPill({
  actions,
}: {
  actions: { systemName: string; label: string; onPress: () => void }[];
}) {
  const { palette } = useTheme();
  return (
    <Host matchContents alignment="trailing" seedColor={palette.accent}>
      <HStack
        spacing={0}
        modifiers={[
          padding({ vertical: 3, horizontal: 3 }),
          glassEffect({ shape: 'capsule', glass: { variant: 'regular', interactive: true } }),
        ]}
      >
        {actions.map((action) => (
          <SystemButton
            key={action.label}
            onPress={action.onPress}
            modifiers={[
              buttonStyle('borderless'),
              padding({ horizontal: 11, vertical: 8 }),
              accessibilityLabel(action.label),
            ]}
          >
            <SystemImage systemName={action.systemName as never} size={17} />
          </SystemButton>
        ))}
      </HStack>
    </Host>
  );
}

export function NativeScreen({
  title,
  subtitle,
  onBack,
  right,
  children,
  topInset = true,
}: {
  title?: string;
  subtitle?: string;
  onBack?: () => void;
  right?: ReactNode;
  children: ReactNode;
  /**
   * A pushed screen owns the status-bar inset; a tab root must not add it, because
   * the native tab controller already places its content below the status bar.
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
        <PageHeader title={title} subtitle={subtitle} onBack={onBack} right={right} />
      ) : null}
      <Host style={{ flex: 1 }} useViewportSizeMeasurement>
        <Form>{children}</Form>
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
