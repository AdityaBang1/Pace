import Constants from 'expo-constants';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useState } from 'react';
import { Alert, Text, View } from 'react-native';

import {
  KEEP_SOURCE_KEY,
  courseCount,
  allSourceUris,
  deleteAllData,
  keepSourceEnabled,
  readGpaScaleValue,
  setSetting,
  storedDocumentCount,
} from '../src/database/repo';
import { parseScale } from '../src/domain/gpa';
import { deleteAllStoredSources } from '../src/features/scan/storage';
import { NativeToggle } from '../src/ui/NativeToggle';
import { readManifestSummary, type ManifestSummary } from '../src/engine/native/manifest';
import { Button, Card, Divider, FieldRow, PageHeader, Screen, SectionHeader } from '../src/ui/components';
import { space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

export default function Settings() {
  const router = useRouter();
  const db = useSQLiteContext();
  const { palette } = useTheme();
  const [manifest, setManifest] = useState<ManifestSummary | null>(null);
  const [counts, setCounts] = useState({ courses: 0, documents: 0 });
  const [keepSource, setKeepSource] = useState(true);
  const [gpaScale, setGpaScale] = useState('Not set');

  useEffect(() => {
    void readManifestSummary().then(setManifest);
  }, []);

  useEffect(() => {
    void (async () => {
      setCounts({ courses: await courseCount(db), documents: await storedDocumentCount(db) });
      setKeepSource(await keepSourceEnabled(db));
      setGpaScale(parseScale(await readGpaScaleValue(db)) ? 'Custom' : 'Not set');
    })();
  }, [db]);

  const confirmDelete = () => {
    Alert.alert(
      'Delete all Pace data?',
      'Every course, imported document and source line is removed from this iPhone. This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete everything',
          style: 'destructive',
          onPress: () => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
            void (async () => {
              // The files are gathered before the rows that point at them are
              // deleted; afterwards nothing knows where they are.
              const uris = await allSourceUris(db);
              await deleteAllData(db);
              await deleteAllStoredSources(uris);
              setCounts({ courses: 0, documents: 0 });
              router.back();
            })();
          },
        },
      ],
    );
  };

  const appVersion = Constants.expoConfig?.version ?? '1.0.0';

  return (
    <Screen scroll>
      <PageHeader title="Settings" subtitle={`Pace ${appVersion}`} onBack={() => router.back()} />

      <SectionHeader>PRIVACY</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg, gap: space.sm }}>
            <Text style={[type.body, { color: palette.ink }]}>Documents are processed locally</Text>
            <Text style={[type.caption, { color: palette.inkMuted, lineHeight: 20 }]}>
              Text recognition uses Apple Vision. The reader that understands the document is a model bundled with the
              app and runs on this device. Your pages are never uploaded.
            </Text>
            <View style={{ height: space.sm }} />
            <FieldRow label="Remote AI calls" value="0" />
            <Divider />
            <FieldRow label="Document uploads" value="0" />
            <Divider />
            <FieldRow label="Accounts required" value="0" />
            <Divider />
            <FieldRow label="Documents stored" value={String(counts.documents)} />
            <Divider />
            <FieldRow label="Courses stored" value={String(counts.courses)} />
          </View>
        </Card>
      </View>

      <SectionHeader>READER</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow label="Engine" value={manifest?.engineVersion ?? 'loading'} />
            <Divider />
            <FieldRow label="Record schema" value={manifest?.schemaVersion ?? 'pace.academic-ir/0.3'} />
            <Divider />
            <FieldRow label="Section model" value={manifest?.models.section.id ?? 'field-classifier-minilm'} />
            <Divider />
            <FieldRow
              label="Parameters"
              value={manifest ? `${(manifest.models.section.parameters / 1_000_000).toFixed(1)}M` : '—'}
            />
            <Divider />
            <FieldRow label="Runtime" value="ONNX Runtime · on device" />
            <Divider />
            <FieldRow label="Context window" value={manifest ? `${manifest.models.section.max_length} tokens` : '—'} />
          </View>
        </Card>
        {manifest?.why ? (
          <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.sm }]}>
            Frozen {manifest.frozenOn ?? ''}: {manifest.why}
          </Text>
        ) : null}
      </View>

      <SectionHeader>STORAGE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card>
          <NativeToggle
            label="Keep the page file with an import"
            value={keepSource}
            onValueChange={(next) => {
              setKeepSource(next);
              void setSetting(db, KEEP_SOURCE_KEY, next ? 'true' : 'false');
            }}
          />
          <Text style={[type.caption, { color: palette.inkMuted, marginTop: space.sm }]}>
            Keeps the photographed page so the source sheet can highlight the line a value came from. Turning this
            off saves space; the quoted source line is kept either way, and a PDF's pages are never stored as images.
          </Text>
        </Card>
      </View>

      <SectionHeader>GPA SCALE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow
              label="Scale"
              value={gpaScale}
              onPress={() => router.push('/gpa-scale')}
              accessory={<Text style={{ color: palette.inkFaint, fontSize: 18 }}>›</Text>}
            />
          </View>
        </Card>
      </View>

      <SectionHeader>DATA</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Text style={[type.caption, { color: palette.inkMuted }]}>
          Undoing a single import is done from the course it created. Deleting everything here removes the stored
          documents and their source lines too.
        </Text>
        <Button title="Delete all data" kind="quiet" onPress={confirmDelete} />
      </View>

      {__DEV__ ? (
        <>
          <SectionHeader>DEVELOPMENT</SectionHeader>
          <View style={{ paddingHorizontal: space.xl }}>
            <Button title="Open diagnostics" kind="secondary" onPress={() => router.push('/debug')} />
          </View>
        </>
      ) : null}

      <View style={{ padding: space.xl, paddingTop: space.xxl }}>
        <Text style={[type.caption, { color: palette.inkFaint }]}>
          Pace turns academic documents into structured academic data. It is not a chat assistant and it does not send
          your documents anywhere.
        </Text>
      </View>
    </Screen>
  );
}
