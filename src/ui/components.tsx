/**
 * The small set of primitives every screen is built from.
 *
 * Keeping them in one file is intentional: the review screen, course detail and
 * home all need to agree on what a rule, a label and a value look like, and a
 * design that lives in nine places drifts by the third screen.
 */
import * as Haptics from 'expo-haptics';
import { ReactNode } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleProp,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import type { ConfidenceBand } from '../engine/types';
import { PressableScale } from './press';
import { hairline, radius, space, type } from './theme';
import { useTheme } from './useTheme';

export function Screen({
  children,
  scroll = false,
  contentStyle,
  edges = 'all',
}: {
  children: ReactNode;
  scroll?: boolean;
  contentStyle?: StyleProp<ViewStyle>;
  edges?: 'all' | 'bottom' | 'top' | 'none';
}) {
  const { palette } = useTheme();
  const insets = useSafeAreaInsets();
  const padding = {
    paddingTop: edges === 'all' || edges === 'top' ? insets.top : 0,
    paddingBottom: edges === 'all' || edges === 'bottom' ? insets.bottom : 0,
  };
  if (scroll) {
    return (
      <ScrollView
        style={{ flex: 1, backgroundColor: palette.background }}
        contentContainerStyle={[padding, { paddingBottom: (padding.paddingBottom || 0) + space.xxl }, contentStyle]}
        keyboardShouldPersistTaps="handled"
      >
        {children}
      </ScrollView>
    );
  }
  return (
    <View style={[{ flex: 1, backgroundColor: palette.background }, padding, contentStyle]}>{children}</View>
  );
}

export function PageHeader({
  title,
  subtitle,
  onBack,
  right,
}: {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  right?: ReactNode;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ paddingHorizontal: space.xl, paddingTop: space.sm, paddingBottom: space.md }}>
      <View style={styles.headerRow}>
        {onBack ? <IconButton glyph="‹" label="Back" onPress={onBack} size={30} /> : null}
        {right ? <View style={{ marginLeft: 'auto' }}>{right}</View> : null}
      </View>
      <Text style={[type.display, { color: palette.ink }]} accessibilityRole="header">
        {title}
      </Text>
      {subtitle ? (
        <Text style={[type.body, { color: palette.inkMuted, marginTop: space.xs }]}>{subtitle}</Text>
      ) : null}
    </View>
  );
}

/**
 * A tappable glyph button. The hit area is 44pt regardless of the glyph size,
 * which is the smallest target that stays comfortable one-handed.
 */
export function IconButton({
  glyph,
  label,
  onPress,
  size = 22,
  tone = 'ink',
}: {
  glyph: string;
  label: string;
  onPress: () => void;
  size?: number;
  tone?: 'ink' | 'accent';
}) {
  const { palette } = useTheme();
  const color = tone === 'accent' ? palette.accent : palette.ink;
  return (
    <PressableScale
      accessibilityLabel={label}
      hitSlop={8}
      onPress={onPress}
      style={{ minWidth: 44, minHeight: 44, justifyContent: 'center' }}
    >
      <Text style={{ color, fontSize: size, lineHeight: size * 1.15, fontWeight: '500' }}>{glyph}</Text>
    </PressableScale>
  );
}

export function SectionHeader({ children, right }: { children: string; right?: ReactNode }) {
  const { palette } = useTheme();
  return (
    <View style={styles.sectionHeader}>
      <Text style={[type.label, { color: palette.inkFaint }]} accessibilityRole="header">
        {children}
      </Text>
      {right ? <View style={{ marginLeft: 'auto' }}>{right}</View> : null}
    </View>
  );
}

export function Card({
  children,
  style,
  padded = true,
}: {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  padded?: boolean;
}) {
  const { palette } = useTheme();
  return (
    <View
      style={[
        {
          backgroundColor: palette.surface,
          borderRadius: radius.lg,
          borderWidth: hairline,
          borderColor: palette.line,
        },
        padded && { padding: space.lg },
        style,
      ]}
    >
      {children}
    </View>
  );
}

export function Divider({ inset = 0 }: { inset?: number }) {
  const { palette } = useTheme();
  return <View style={{ height: hairline, backgroundColor: palette.line, marginLeft: inset }} />;
}

/** One row of a card: label on the left, value on the right. */
export function FieldRow({
  label,
  value,
  accessory,
  onPress,
  emphasis = false,
}: {
  label: string;
  value: ReactNode;
  accessory?: ReactNode;
  onPress?: () => void;
  emphasis?: boolean;
}) {
  const { palette } = useTheme();
  const content = (
    <View style={styles.fieldRow}>
      <Text style={[type.body, { color: palette.inkMuted, flexShrink: 1 }]}>{label}</Text>
      <View style={styles.fieldValue}>
        {typeof value === 'string' ? (
          <Text
            style={[
              emphasis ? type.bodyStrong : type.body,
              { color: palette.ink, textAlign: 'right' },
            ]}
          >
            {value}
          </Text>
        ) : (
          value
        )}
        {accessory}
      </View>
    </View>
  );
  if (!onPress) return content;
  // A row that opens its source or editor is a tap, not a detent: the sheet that
  // slides up is the feedback, so no haptic and no other movement.
  return (
    <PressableScale accessibilityRole="button" onPress={onPress}>
      {content}
    </PressableScale>
  );
}

export function Button({
  title,
  onPress,
  kind = 'primary',
  disabled = false,
  loading = false,
  compact = false,
}: {
  title: string;
  onPress: () => void;
  kind?: 'primary' | 'secondary' | 'quiet';
  disabled?: boolean;
  loading?: boolean;
  compact?: boolean;
}) {
  const { palette } = useTheme();
  const background =
    kind === 'primary' ? palette.accent : kind === 'secondary' ? palette.surfaceSunken : 'transparent';
  const foreground = kind === 'primary' ? palette.accentInk : palette.ink;
  return (
    <PressableScale
      accessibilityState={{ disabled: disabled || loading, busy: loading }}
      // A compact button is 36pt tall, so the touch area is grown to 44 instead.
      hitSlop={compact ? 4 : undefined}
      disabled={disabled || loading}
      onPress={() => {
        // Only the committing action has weight. Secondary and quiet buttons are
        // navigation and retries, which happen often enough that a tap each time
        // would turn the haptics off for the user.
        if (kind === 'primary') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
        onPress();
      }}
      style={{
        backgroundColor: background,
        opacity: disabled ? 0.4 : 1,
        paddingVertical: compact ? space.sm : space.lg,
        paddingHorizontal: space.xl,
        borderRadius: radius.lg,
        borderWidth: kind === 'quiet' ? hairline : 0,
        borderColor: palette.line,
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: compact ? 36 : 52,
      }}
    >
      {loading ? (
        <ActivityIndicator color={foreground} />
      ) : (
        <Text style={[type.bodyStrong, { color: foreground }]}>{title}</Text>
      )}
    </PressableScale>
  );
}

export function Chip({
  label,
  tone = 'neutral',
  onPress,
  selected = false,
}: {
  label: string;
  tone?: 'neutral' | 'accent' | 'warn' | 'danger';
  onPress?: () => void;
  selected?: boolean;
}) {
  const { palette } = useTheme();
  const map = {
    neutral: { bg: palette.surfaceSunken, fg: palette.inkMuted },
    accent: { bg: palette.accentSoft, fg: palette.accent },
    warn: { bg: palette.warnSoft, fg: palette.warn },
    danger: { bg: palette.dangerSoft, fg: palette.danger },
  } as const;
  const { bg, fg } = map[tone];
  const body = (
    <View
      style={{
        backgroundColor: bg,
        borderRadius: radius.sm,
        paddingVertical: 5,
        paddingHorizontal: 9,
        borderWidth: selected ? 1 : 0,
        borderColor: palette.accent,
      }}
    >
      <Text style={[type.caption, { color: fg, fontWeight: '600' }]}>{label}</Text>
    </View>
  );
  if (!onPress) return body;
  // Chips act as a segmented control (document type, day filter), which is the one
  // place the design language allows a selection tick.
  return (
    <PressableScale
      accessibilityState={{ selected }}
      // The chip is visually 26pt tall; the target is the 44pt the platform asks for.
      hitSlop={10}
      onPress={() => {
        Haptics.selectionAsync().catch(() => {});
        onPress();
      }}
    >
      {body}
    </PressableScale>
  );
}

/**
 * A confidence marker.
 *
 * Rule-extracted values have no probability, and the product must not invent one,
 * so `value === null` renders "From text" instead of a percentage. Model values
 * render as a band, never as a raw score.
 */
export function ConfidenceMark({
  band,
  confidence,
  source,
  compact = false,
}: {
  band: ConfidenceBand | null;
  confidence?: number | null;
  source: 'rule' | 'model';
  compact?: boolean;
}) {
  const { palette } = useTheme();
  if (source === 'rule' || band === null) {
    return (
      <View style={styles.mark}>
        <View style={[styles.dot, { backgroundColor: palette.inkFaint }]} />
        <Text style={[type.caption, { color: palette.inkFaint }]}>
          {compact ? 'From text' : 'Read from the page'}
        </Text>
      </View>
    );
  }
  const color = band === 'high' ? palette.high : band === 'medium' ? palette.medium : palette.low;
  const text =
    band === 'high' ? 'High confidence' : band === 'medium' ? 'Medium confidence' : 'Needs review';
  return (
    <View style={styles.mark} accessibilityLabel={text}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={[type.caption, { color }]}>
        {text}
        {confidence != null && !compact ? ` · ${Math.round(confidence * 100)}%` : ''}
      </Text>
    </View>
  );
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action?: ReactNode;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ padding: space.xl, alignItems: 'flex-start', gap: space.sm }}>
      <Text style={[type.heading, { color: palette.ink }]}>{title}</Text>
      <Text style={[type.body, { color: palette.inkMuted }]}>{body}</Text>
      {action ? <View style={{ marginTop: space.sm }}>{action}</View> : null}
    </View>
  );
}

export function Notice({
  tone,
  title,
  body,
}: {
  tone: 'warn' | 'danger' | 'accent';
  title?: string;
  body: string;
}) {
  const { palette } = useTheme();
  const map = {
    warn: { bg: palette.warnSoft, fg: palette.warn },
    danger: { bg: palette.dangerSoft, fg: palette.danger },
    accent: { bg: palette.accentSoft, fg: palette.accent },
  } as const;
  const { bg, fg } = map[tone];
  return (
    <View style={{ backgroundColor: bg, borderRadius: radius.md, padding: space.md, gap: 2 }}>
      {title ? <Text style={[type.bodyStrong, { color: fg }]}>{title}</Text> : null}
      <Text style={[type.caption, { color: fg }]}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 36,
    marginBottom: space.xs,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: space.xl,
    marginTop: space.xl,
    marginBottom: space.sm,
  },
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.md,
    paddingVertical: space.sm,
    minHeight: 40,
  },
  fieldValue: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    // Take the space the label leaves, rather than sizing to content: a long
    // value (a transcript's course title) would otherwise overflow the row and
    // draw over the label, which shrink-to-fit cannot prevent.
    flex: 1,
    flexShrink: 1,
    justifyContent: 'flex-end',
  },
  mark: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
});
