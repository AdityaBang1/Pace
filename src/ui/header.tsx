/**
 * The screen header, drawn by SwiftUI.
 *
 * It used to be React Native text with SwiftUI buttons bolted into the corners,
 * which is why the controls sat above the words they belonged to and why the back
 * chevron never quite matched the system's. The whole header is one SwiftUI tree
 * now, in one `Host`: title, subtitle, back button and actions share a single
 * layout, so their baselines and their spacing are the platform's business.
 *
 * Two shapes, matching iOS:
 *
 * - A destination with a way back gets an *inline* header: 44pt glass back button,
 *   the title centred between the edges, actions trailing. That is the shape a
 *   sheet and a pushed screen already have in every system app.
 * - A tab root has nowhere to go back to, so it gets the *large* title on the
 *   leading edge with the actions trailing, which is the shape the rest of iOS
 *   uses for a top-level destination.
 *
 * Every control is a system `Button` inside a glass shape or a bordered-prominent
 * style, at least 44pt on each side: the platform owns the press highlight, the
 * disabled state, the accessibility traits and the material, and the target is the
 * size the Human Interface Guidelines ask for.
 */
import {
  Button as SystemButton,
  HStack,
  Host,
  Image as SystemImage,
  Spacer,
  Text as SystemText,
  VStack,
  ZStack,
} from '@expo/ui/swift-ui';
import {
  accessibilityLabel,
  background,
  buttonStyle,
  disabled,
  font,
  foregroundStyle,
  frame,
  glassEffect,
  lineLimit,
  multilineTextAlignment,
  padding,
  tint,
} from '@expo/ui/swift-ui/modifiers';

import { useTheme } from './useTheme';

export interface HeaderIconAction {
  systemName: string;
  label: string;
  onPress: () => void;
}

export interface HeaderPrimaryAction {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}

/** A 44pt glass back circle: the platform chevron, at the HIG's target size. */
function BackButton({ onPress }: { onPress: () => void }) {
  return (
    <SystemButton
      onPress={onPress}
      modifiers={[
        buttonStyle('borderless'),
        padding({ horizontal: 13, vertical: 13 }),
        glassEffect({ shape: 'circle', glass: { variant: 'regular', interactive: true } }),
        accessibilityLabel('Back'),
      ]}
    >
      <SystemImage systemName="chevron.left" size={18} />
    </SystemButton>
  );
}

/** The trailing glyph actions, sharing one glass capsule. */
function ActionPill({ actions }: { actions: HeaderIconAction[] }) {
  return (
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
            padding({ horizontal: 12, vertical: 10 }),
            accessibilityLabel(action.label),
          ]}
        >
          <SystemImage systemName={action.systemName as never} size={17} />
        </SystemButton>
      ))}
    </HStack>
  );
}

/** The one filled header action, usually Save. */
function PrimaryButton({ primary }: { primary: HeaderPrimaryAction }) {
  const { palette } = useTheme();
  return (
    <SystemButton
      onPress={primary.onPress}
      modifiers={[
        buttonStyle('borderedProminent'),
        tint(palette.accent),
        disabled(primary.disabled ?? false),
        accessibilityLabel(primary.label),
      ]}
    >
      <SystemText modifiers={[font({ textStyle: 'subheadline', weight: 'semibold' })]}>
        {primary.label}
      </SystemText>
    </SystemButton>
  );
}

export function NativeHeader({
  title,
  subtitle,
  onBack,
  actions,
  primary,
}: {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  actions?: HeaderIconAction[];
  primary?: HeaderPrimaryAction;
}) {
  const { palette } = useTheme();
  const inline = onBack != null;

  return (
    <Host
      matchContents={{ vertical: true }}
      style={{ width: '100%' }}
      seedColor={palette.accent}
    >
      <VStack
        alignment={inline ? 'center' : 'leading'}
        spacing={3}
        modifiers={[
          frame({ maxWidth: 9999 }),
          padding({ horizontal: inline ? 12 : 20, vertical: inline ? 8 : 6 }),
          background(palette.background),
        ]}
      >
        {inline ? (
          /*
            Centred the way a navigation bar is centred: the title sits in a stack
            of its own across the whole width, and the controls are laid over it at
            the two edges. A title that shrinks between a back button and nothing
            else ends up left of the middle, which is what it looked like before.
            The side padding keeps a long title from running under the controls.
          */
          <ZStack modifiers={[frame({ maxWidth: 9999 })]}>
            <SystemText
              modifiers={[
                font({ textStyle: 'headline', weight: 'semibold' }),
                foregroundStyle(palette.ink),
                frame({ maxWidth: 9999, alignment: 'center' }),
                padding({ horizontal: 56 }),
                multilineTextAlignment('center'),
                lineLimit(1),
              ]}
            >
              {title}
            </SystemText>
            <HStack spacing={8} alignment="center" modifiers={[frame({ maxWidth: 9999 })]}>
              {onBack ? <BackButton onPress={onBack} /> : <Spacer />}
              <Spacer />
              {actions && actions.length ? <ActionPill actions={actions} /> : null}
              {primary ? <PrimaryButton primary={primary} /> : null}
            </HStack>
          </ZStack>
        ) : (
          <HStack spacing={10} alignment="center" modifiers={[frame({ maxWidth: 9999 })]}>
            <SystemText
              modifiers={[
                font({ textStyle: 'largeTitle', weight: 'bold' }),
                foregroundStyle(palette.ink),
                frame({ maxWidth: 9999, alignment: 'leading' }),
                multilineTextAlignment('leading'),
                lineLimit(2),
              ]}
            >
              {title}
            </SystemText>
            {actions && actions.length ? <ActionPill actions={actions} /> : null}
            {primary ? <PrimaryButton primary={primary} /> : null}
          </HStack>
        )}
        {subtitle ? (
          <SystemText
            modifiers={[
              font({ textStyle: 'footnote' }),
              foregroundStyle(palette.inkMuted),
              frame({ maxWidth: 9999, alignment: inline ? 'center' : 'leading' }),
              multilineTextAlignment(inline ? 'center' : 'leading'),
            ]}
          >
            {subtitle}
          </SystemText>
        ) : null}
      </VStack>
    </Host>
  );
}
