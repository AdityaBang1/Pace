import { Button as NativeButton, Section, Text as NativeText } from '@expo/ui/swift-ui';
import { disabled, font, foregroundStyle } from '@expo/ui/swift-ui/modifiers';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Share } from 'react-native';

import { isOcrAvailable } from '../modules/pace-vision';
import { loadEngine, releaseEngine } from '../src/engine/native';
import { readManifestSummary, type ManifestSummary } from '../src/engine/native/manifest';
import { getLastRun, type LastRun } from '../src/features/scan/lastRun';
import { LabeledContent, NativeScreen, Value } from '../src/ui/native';
import { useTheme } from '../src/ui/useTheme';

/**
 * What the reader is, and what it did last.
 *
 * A development screen, and a native form like the rest: the manifest is a table
 * of facts, the last run is the same table for one document, and the only actions
 * are reloading the reader and sending the text it was given somewhere a person
 * can read it. Nothing here is a chart.
 */
export default function Debug() {
  const router = useRouter();
  const { palette } = useTheme();
  const [manifest, setManifest] = useState<ManifestSummary | null>(null);
  const [last, setLast] = useState<LastRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadMs, setLoadMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void readManifestSummary().then(setManifest);
    setLast(getLastRun());
  }, []);

  const reload = async () => {
    setLoading(true);
    setError(null);
    const started = Date.now();
    try {
      releaseEngine();
      await loadEngine();
      setLoadMs(Date.now() - started);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'the reader did not load');
    } finally {
      setLoading(false);
    }
  };

  const mono = [font({ design: 'monospaced', size: 12 }), foregroundStyle(palette.inkMuted)];
  const caption = [font({ textStyle: 'footnote' }), foregroundStyle(palette.inkMuted)];

  return (
    <NativeScreen
      title="Diagnostics"
      subtitle="Development only — not shown in the consumer build"
      onBack={() => router.back()}
    >
      <Section
        title="Reader"
        footer={
          error ? (
            <NativeText modifiers={[foregroundStyle(palette.danger)]}>{`The reader did not load: ${error}`}</NativeText>
          ) : undefined
        }
      >
        <LabeledContent label="OCR module">
          <Value>{isOcrAvailable() ? 'available' : 'not linked'}</Value>
        </LabeledContent>
        <LabeledContent label="Engine">
          <Value>{manifest?.engineVersion ?? '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Schema">
          <Value>{manifest?.schemaVersion ?? '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Section model">
          <Value>{manifest?.models.section.id ?? '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Context model">
          <Value>{manifest?.models.context.id ?? '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Doc-type classes">
          <Value>{manifest?.models.documentType.classes.join(', ') ?? '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Matcher threshold">
          <Value>{manifest ? String(manifest.models.entityResolution.threshold) : '—'}</Value>
        </LabeledContent>
        <LabeledContent label="Load time">
          <Value>{loadMs != null ? `${loadMs} ms` : 'not measured this session'}</Value>
        </LabeledContent>
        <NativeButton
          label={loading ? 'Loading…' : 'Reload the reader'}
          systemImage="arrow.clockwise"
          modifiers={[disabled(loading)]}
          onPress={() => void reload()}
        />
      </Section>

      {!last ? (
        <Section title="Last run">
          <NativeText modifiers={caption}>No document has been read since the app started.</NativeText>
        </Section>
      ) : (
        <>
          <Section title="Last run">
            <LabeledContent label="Proposed type">
              <Value>
                {last.documentType
                  ? `${last.documentType.label} · ${Math.round(last.documentType.confidence * 100)}%`
                  : 'unavailable'}
              </Value>
            </LabeledContent>
            <LabeledContent label="Characters">
              <Value>{last.characters.toLocaleString()}</Value>
            </LabeledContent>
            <LabeledContent label="Sections">
              <Value>{String(last.sections.length)}</Value>
            </LabeledContent>
            <LabeledContent label="Extracted">
              <Value>{`${last.counts.grading} grading · ${last.counts.meetings} meetings · ${last.counts.policies} policies`}</Value>
            </LabeledContent>
            <LabeledContent label="Evidence rows">
              <Value>{String(last.counts.evidence)}</Value>
            </LabeledContent>
          </Section>

          {last.documentType?.top.length ? (
            <Section title="Document type">
              {last.documentType.top.slice(0, 4).map((row) => (
                <LabeledContent key={`${row.label}-${row.probability}`} label={row.label}>
                  <Value>{`${(row.probability * 100).toFixed(1)}%`}</Value>
                </LabeledContent>
              ))}
            </Section>
          ) : null}

          <Section title="Timings">
            {Object.entries(last.timings).map(([key, value]) => (
              <LabeledContent key={key} label={key}>
                <Value>{`${Math.round(value)} ms`}</Value>
              </LabeledContent>
            ))}
          </Section>

          {last.extraction ? (
            <Section
              title="Extraction"
              footer={
                <NativeText modifiers={caption}>
                  The exact text the reader was given, so a page that read badly can be sent somewhere it can be
                  diagnosed.
                </NativeText>
              }
            >
              <NativeButton
                label="Share the extracted text"
                systemImage="square.and.arrow.up"
                onPress={() => void Share.share({ message: last.extraction?.text ?? '' })}
              />
              {last.extraction.pages.length ? (
                last.extraction.pages.map((page) => (
                  <LabeledContent
                    key={page.page}
                    label={`page ${page.page + 1} · ${page.method === 'text_layer' ? 'text layer' : 'OCR'}`}
                  >
                    <Value>
                      {`${page.characters} chars${
                        page.confidence != null ? ` · ${Math.round(page.confidence * 100)}%` : ''
                      }`}
                    </Value>
                  </LabeledContent>
                ))
              ) : (
                <NativeText modifiers={caption}>a plain-text file, no pages</NativeText>
              )}
              <NativeText modifiers={mono}>{last.extraction.text.slice(0, 2000)}</NativeText>
            </Section>
          ) : null}

          {last.sections.length ? (
            <Section title="Sections">
              {last.sections.map((section) => (
                <NativeText
                  key={`${section.heading}-${section.confidence}`}
                  modifiers={caption}
                >
                  {`${section.heading} — ${section.label} (${Math.round(section.confidence * 100)}%)`}
                </NativeText>
              ))}
            </Section>
          ) : null}

          {last.warnings.length ? (
            <Section title="Warnings">
              {last.warnings.map((warning) => (
                <NativeText key={warning} modifiers={[font({ textStyle: 'footnote' }), foregroundStyle(palette.warn)]}>
                  {warning}
                </NativeText>
              ))}
            </Section>
          ) : null}
        </>
      )}
    </NativeScreen>
  );
}
