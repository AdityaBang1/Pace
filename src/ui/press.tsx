/**
 * Press feedback, on the UI thread, as a CSS transition.
 *
 * A press happens tens of times an hour, so the gate allows it only as something
 * the user barely notices: 120ms and a 3% scale. There is no gesture and no shared
 * value here — React flips one style value, the transition runs in Reanimated on the
 * UI thread, and `transform` never re-runs layout.
 *
 * Reduced motion keeps the press visible without the movement: the scale stays at 1
 * and the same 120ms is spent on a small opacity dip instead.
 */
import { Pressable, type AccessibilityRole, type AccessibilityState, type StyleProp, type ViewStyle } from 'react-native';
import Animated, { useReducedMotion } from 'react-native-reanimated';
import type { ReactNode } from 'react';

import { EASE_OUT_CSS, PRESS_MS, PRESS_SCALE } from './motion';

export function PressableScale({
  children,
  onPress,
  onLongPress,
  disabled = false,
  hitSlop,
  accessibilityLabel,
  accessibilityRole = 'button',
  accessibilityState,
  style,
}: {
  children: ReactNode;
  onPress?: () => void;
  onLongPress?: () => void;
  disabled?: boolean;
  hitSlop?: number;
  accessibilityLabel?: string;
  accessibilityRole?: AccessibilityRole;
  accessibilityState?: AccessibilityState;
  /** The visual box — background, padding, radius. The label scales with it. */
  style?: StyleProp<ViewStyle>;
}) {
  const reduced = useReducedMotion();
  return (
    <Pressable
      accessibilityRole={accessibilityRole}
      accessibilityLabel={accessibilityLabel}
      accessibilityState={disabled ? { ...accessibilityState, disabled: true } : accessibilityState}
      disabled={disabled}
      hitSlop={hitSlop}
      pressRetentionOffset={16}
      onPress={onPress}
      onLongPress={onLongPress}
    >
      {({ pressed }) => (
        <Animated.View
          style={[
            style,
            {
              transform: [{ scale: pressed && !disabled && !reduced ? PRESS_SCALE : 1 }],
              // Only claim `opacity` under reduced motion; otherwise it would
              // overwrite a caller's own dimming (a disabled button, say).
              ...(reduced && pressed && !disabled ? { opacity: 0.72 } : null),
              transitionProperty: reduced ? 'opacity' : 'transform',
              transitionDuration: `${PRESS_MS}ms`,
              transitionTimingFunction: EASE_OUT_CSS,
            },
          ]}
        >
          {children}
        </Animated.View>
      )}
    </Pressable>
  );
}
