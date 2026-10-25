import { useRouter } from 'expo-router';
import { isOcrAvailable } from '../modules/pace-vision';
import { useEffect, useState } from 'react';
import { Text, View } from 'react-native';

import { loadEngine, releaseEngine } from '../src/engine/native';
import { readManifestSummary, type ManifestSummary } from '../src/engine/native/manifest';
import { getLastRun, type LastRun } from '../src/features/scan/lastRun';
import { Button, Card, Divider, FieldRow, Notice, PageHeader, Screen, SectionHeader } from '../src/ui/components';
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

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

  return (
    <Screen scroll>
      <PageHeader
        title="Diagnostics"
        subtitle="Development only — not shown in the consumer build"
        onBack={() => router.back()}
      />

      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow label="OCR module" value={isOcrAvailable() ? 'available' : 'not linked'} />
            <Divider />
            <FieldRow label="Engine" value={manifest?.engineVersion ?? '—'} />
            <Divider />
            <FieldRow label="Schema" value={manifest?.schemaVersion ?? '—'} />
            <Divider />
            <FieldRow label="Section model" value={manifest?.models.section.id ?? '—'} />
            <Divider />
            <FieldRow label="Context model" value={manifest?.models.context.id ?? '—'} />
            <Divider />
            <FieldRow label="Doc-type classes" value={manifest?.models.documentType.classes.join(', ') ?? '—'} />
            <Divider />
            <FieldRow label="Matcher threshold" value={manifest ? String(manifest.models.entityResolution.threshold) : '—'} />
            <Divider />
            <FieldRow label="Load time" value={loadMs != null ? `${loadMs} ms` : 'not measured this session'} />
          </View>
        </Card>

        {error ? <Notice tone="danger" title="The reader did not load" body={error} /> : null}
        <Button title={loading ? 'Loading…' : 'Reload the reader'} kind="secondary" onPress={() => void reload()} loading={loading} />
      </View>

      <SectionHeader>LAST RUN</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        {!last ? (
          <Text style={[type.caption, { color: palette.inkMuted }]}>
            No document has been read since the app started.
          </Text>
        ) : (
          <>
            <Card padded={false}>
              <View style={{ padding: space.lg }}>
                <FieldRow
                  label="Proposed type"
                  value={
                    last.documentType
                      ? `${last.documentType.label} · ${Math.round(last.documentType.confidence * 100)}%`
                      : 'unavailable'
                  }
                />
                <Divider />
                <FieldRow label="Characters" value={last.characters.toLocaleString()} />
                <Divider />
                <FieldRow label="Sections" value={String(last.sections.length)} />
                <Divider />
                <FieldRow
                  label="Extracted"
                  value={`${last.counts.grading} grading · ${last.counts.meetings} meetings · ${last.counts.policies} policies`}
                />
                <Divider />
                <FieldRow label="Evidence rows" value={String(last.counts.evidence)} />
              </View>
            </Card>

            <Card>
              <Text style={[type.label, { color: palette.inkFaint }]}>DOCUMENT TYPE</Text>
              {last.documentType?.top.slice(0, 4).map((row) => (
                <FieldRow
                  key={`${row.label}-${row.probability}`}
                  label={row.label}
                  value={`${(row.probability * 100).toFixed(1)}%`}
                />
              )) ?? null}
            </Card>

            <Card>
              <Text style={[type.label, { color: palette.inkFaint }]}>TIMINGS</Text>
              {Object.entries(last.timings).map(([key, value]) => (
                <FieldRow key={key} label={key} value={`${Math.round(value)} ms`} />
              ))}
            </Card>

            {last.extraction ? (
              <Card>
                <Text style={[type.label, { color: palette.inkFaint }]}>EXTRACTION</Text>
                <View style={{ marginTop: space.sm, gap: 6 }}>
                  {last.extraction.pages.length ? (
                    last.extraction.pages.map((page) => (
                      <FieldRow
                        key={page.page}
                        label={`page ${page.page + 1} · ${page.method === 'text_layer' ? 'text layer' : 'OCR'}`}
                        value={`${page.characters} chars${
                          page.confidence != null ? ` · ${Math.round(page.confidence * 100)}%` : ''
                        }`}
                      />
                    ))
                  ) : (
                    <Text style={[type.caption, { color: palette.inkMuted }]}>a plain-text file, no pages</Text>
                  )}
                </View>
                <Text
                  style={[type.mono, { color: palette.inkMuted, marginTop: space.md }]}
                  selectable
                >
                  {last.extraction.text.slice(0, 2000)}
                </Text>
              </Card>
            ) : null}

            <Card>
              <Text style={[type.label, { color: palette.inkFaint }]}>SECTIONS</Text>
              <View style={{ marginTop: space.sm, gap: 6 }}>
                {last.sections.map((section) => (
                  <Text key={`${section.heading}-${section.confidence}`} style={[type.caption, { color: palette.inkMuted }]}>
                    {section.heading} — {section.label} ({Math.round(section.confidence * 100)}%)
                  </Text>
                ))}
              </View>
            </Card>

            {last.warnings.length ? (
              <Card>
                <Text style={[type.label, { color: palette.inkFaint }]}>WARNINGS</Text>
                <View style={{ marginTop: space.sm, gap: 6 }}>
                  {last.warnings.map((warning) => (
                    <Text key={warning} style={[type.caption, { color: palette.warn }]}>
                      {warning}
                    </Text>
                  ))}
                </View>
              </Card>
            ) : null}
          </>
        )}
      </View>
    </Screen>
  );
}
