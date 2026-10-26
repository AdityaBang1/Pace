/**
 * Where a value came from, drawn as part of the form that shows the value.
 *
 * The answer is the document's own line, quoted, with the page it was printed on
 * and how the reader found it. It used to be a React Native modal over the
 * screen; as a section it belongs to the same SwiftUI form as the values, opens
 * where the reviewer already is, and closes without a second presentation layer.
 */
import { Button as NativeButton, LabeledContent, Section, Text as NativeText } from '@expo/ui/swift-ui';
import { font, foregroundStyle } from '@expo/ui/swift-ui/modifiers';

import { Value } from './native';
import { useTheme } from './useTheme';

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

export function SourceSection({ source, onClose }: { source: SourceLine; onClose: () => void }) {
  const { palette } = useTheme();
  return (
    <Section title={`Source · ${source.label}`} footer={<NativeText>{source.method}</NativeText>}>
      <NativeText
        modifiers={[font({ design: 'serif', size: 16 }), foregroundStyle(palette.ink)]}
      >
        {`“${source.quote}”`}
      </NativeText>
      <LabeledContent label="Page">
        <Value>{source.page != null ? String(source.page + 1) : 'not recorded'}</Value>
      </LabeledContent>
      <LabeledContent label="Confidence">
        <Value>
          {source.confidence != null
            ? `${Math.round(source.confidence * 100)}% · ${source.confidenceSource === 'model' ? 'model' : 'rule'}`
            : 'not measured'}
        </Value>
      </LabeledContent>
      <NativeButton label="Done" systemImage="checkmark" onPress={onClose} />
    </Section>
  );
}
