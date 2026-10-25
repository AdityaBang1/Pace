/**
 * The two curves and the one duration this app uses for feedback.
 *
 * Everything here is deliberately small. The rest of the motion in Pace belongs to
 * the platform — the native stack's push, the modal sheet, the picker — and is not
 * rebuilt in JavaScript.
 */
import { Easing } from 'react-native-reanimated';

/** Strong ease-out: starts fast, which is when the user is looking. */
export const EASE_OUT = Easing.bezier(0.23, 1, 0.32, 1);

/** On-screen movement between two known positions. */
export const EASE_IN_OUT = Easing.bezier(0.77, 0, 0.175, 1);

/**
 * The CSS-transition form of `EASE_OUT`, for the press recipe.
 *
 * Reanimated's CSS transitions take the curve as a string; keeping both forms
 * beside each other stops the two from drifting apart.
 */
export const EASE_OUT_CSS = 'cubic-bezier(0.23, 1, 0.32, 1)';

/** Press feedback ceiling: 120ms and a 3% scale, per the frequency gate. */
export const PRESS_MS = 120;
export const PRESS_SCALE = 0.97;

/** A change the user did not cause stays under this. */
export const REVEAL_MS = 160;
