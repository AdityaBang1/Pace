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
import { Form, Host, Section, Text as NativeText, LabeledContent } from '@expo/ui/swift-ui';
import { ReactNode } from 'react';
import { View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { PageHeader } from './components';
import { useTheme } from './useTheme';

export { Form, Host, Section, NativeText, LabeledContent };

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
