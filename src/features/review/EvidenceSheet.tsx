/**
 * "Where did this come from?"
 *
 * The answer is the document's own line, quoted, with the value marked — and the
 * highlighted region of the photograph when Vision's geometry reached us. It is
 * deliberately a sheet, not a route: the reviewer should be able to check a source
 * and come straight back without losing their place in the list.
 */
import { useEffect, useState } from 'react';
import { Image, Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Button, Card } from '../../ui/components';
import { hairline, radius, space, type } from '../../ui/theme';
import { useTheme } from '../../ui/useTheme';

export interface SourceLine {
  label: string;
  quote: string;
  page: number | null;
  bbox: { x: number; y: number; width: number; height: number } | null;
  confidence: number | null;
  confidenceSource: 'rule' | 'model';
  method: string;
  image: { uri: string; width: number; height: number } | null;
}

/** Draws the value inside its line, without assuming the value is unique. */
function QuotedLine({ quote, value }: { quote: string; value: string }) {
  const { palette } = useTheme();
  const trimmed = value.trim();
  const at = trimmed ? quote.toLowerCase().indexOf(trimmed.toLowerCase()) : -1;
  if (at === -1) {
    return <Text style={[type.body, { color: palette.ink, lineHeight: 26 }]}>{quote}</Text>;
  }
  return (
    <Text style={[type.body, { color: palette.ink, lineHeight: 26 }]}>
      {quote.slice(0, at)}
      <Text style={{ backgroundColor: palette.accentSoft, color: palette.ink, fontWeight: '600' }}>
        {quote.slice(at, at + trimmed.length)}
      </Text>
      {quote.slice(at + trimmed.length)}
    </Text>
  );
}

/**
 * The photograph with the source line marked.
 *
 * The box is measured, not guessed: the image is drawn `contain`-style inside a
 * 4:3 viewport, and the normalised box is projected onto the drawn rect, so the
 * highlight lands on the line for portrait and landscape pages alike.
 */
function PageHighlight({ image, bbox }: { image: SourceLine['image']; bbox: SourceLine['bbox'] }) {
  const { palette } = useTheme();
  // The course screen reads its page back from SQLite, which stores the URI but
  // not the pixel size; the aspect ratio is measured once from the file instead of
  // silently dropping the highlight.
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (!image) {
      setSize(null);
      return;
    }
    if (image.width > 0 && image.height > 0) {
      setSize({ width: image.width, height: image.height });
      return;
    }
    let cancelled = false;
    Image.getSize(
      image.uri,
      (width, height) => {
        if (!cancelled) setSize({ width, height });
      },
      () => {
        if (!cancelled) setSize(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [image?.uri, image?.width, image?.height]);

  if (!image || !bbox || !size || !size.width || !size.height) return null;

  const aspect = size.width / size.height;
  return (
    <View
      style={{
        borderRadius: radius.md,
        overflow: 'hidden',
        backgroundColor: palette.surfaceSunken,
        marginTop: space.md,
      }}
    >
      <View style={{ width: '100%', aspectRatio: aspect }}>
        <Image source={{ uri: image.uri }} style={{ width: '100%', height: '100%' }} resizeMode="contain" />
        <View
          pointerEvents="none"
          style={{
            position: 'absolute',
            left: `${bbox.x * 100}%`,
            top: `${bbox.y * 100}%`,
            width: `${bbox.width * 100}%`,
            height: `${Math.max(bbox.height * 100, 2)}%`,
            backgroundColor: palette.accentSoft,
            opacity: 0.45,
            borderWidth: 1,
            borderColor: palette.accent,
            borderRadius: 3,
          }}
        />
      </View>
    </View>
  );
}

export function EvidenceSheet({
  visible,
  source,
  onClose,
}: {
  visible: boolean;
  source: SourceLine | null;
  onClose: () => void;
}) {
  const { palette } = useTheme();
  const insets = useSafeAreaInsets();

  return (
    <Modal visible={visible && source != null} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={{ flex: 1, backgroundColor: palette.background, paddingTop: space.lg }}>
        <View style={[styles.header, { borderBottomColor: palette.line }]}>
          <Text style={[type.heading, { color: palette.ink }]}>Source</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Close source"
            onPress={onClose}
            hitSlop={12}
            style={({ pressed }) => ({ opacity: pressed ? 0.5 : 1, padding: space.sm })}
          >
            <Text style={[type.caption, { color: palette.accent, fontWeight: '600' }]}>Done</Text>
          </Pressable>
        </View>

        {source ? (
          <View style={{ paddingHorizontal: space.xl }}>
            <Text style={[type.caption, { color: palette.inkMuted }]}>{source.label}</Text>
            <Card style={{ marginTop: space.sm, paddingVertical: space.lg }}>
              {source.quote ? (
                <QuotedLine quote={source.quote} value={source.label} />
              ) : (
                <Text style={[type.caption, { color: palette.inkMuted }]}>
                  The source line for this value was not kept.
                </Text>
              )}
            </Card>

            <PageHighlight image={source.image} bbox={source.bbox} />

            <View style={{ marginTop: space.lg, gap: 6 }}>
              {source.page != null ? (
                <Text style={[type.caption, { color: palette.inkMuted }]}>Page {source.page + 1}</Text>
              ) : null}
              <Text style={[type.caption, { color: palette.inkMuted }]}>
                {source.method}
                {source.confidence != null ? ` · ${Math.round(source.confidence * 100)}% confidence` : ''}
              </Text>
              <Text style={[type.caption, { color: palette.inkFaint }]}>
                {source.confidenceSource === 'model'
                  ? 'Confidence comes from a model and is not calibrated for your school.'
                  : 'Read from the page with a pattern, not a probability.'}
              </Text>
            </View>
          </View>
        ) : null}

        <View style={{ marginTop: 'auto', padding: space.xl, paddingBottom: insets.bottom + space.lg }}>
          <Button title="Done" kind="secondary" onPress={onClose} />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: space.xl,
    paddingBottom: space.md,
    borderBottomWidth: hairline,
  },
});
