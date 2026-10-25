import * as DocumentPicker from 'expo-document-picker';
import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useRef, useState } from 'react';
import { ActivityIndicator, Alert, Linking, Text, View } from 'react-native';
import Animated, { FadeIn, FadeInDown, useReducedMotion } from 'react-native-reanimated';

import { keepSourceEnabled } from '../src/database/repo';
import { useImportDraft } from '../src/features/import/ImportContext';
import { ScanError, scanFile, scanText, type ScanProgress, type ScanStage } from '../src/features/scan/pipeline';
import { Button, Card, Notice, PageHeader, Screen } from '../src/ui/components';
import { EASE_OUT, REVEAL_MS } from '../src/ui/motion';
import { PressableScale } from '../src/ui/press';
import { hairline, radius, space, type } from '../src/ui/theme';
import { useTheme } from '../src/ui/useTheme';

type Phase = ScanStage | 'checking';

const STAGES: { key: Phase; label: string }[] = [
  { key: 'preparing', label: 'Preparing the reader' },
  { key: 'reading', label: 'Reading the text' },
  { key: 'understanding', label: 'Understanding the document' },
  { key: 'checking', label: 'Checking the result' },
];

/** "Take photo" needs the camera; a denied permission has to say where to fix it. */
async function ensureCamera(): Promise<boolean> {
  const permission = await ImagePicker.requestCameraPermissionsAsync();
  if (permission.granted) return true;
  Alert.alert(
    'Camera access is off',
    'Pace needs the camera to photograph a document. You can turn it on in Settings, or choose a photo instead.',
    [
      { text: 'Not now', style: 'cancel' },
      { text: 'Open Settings', onPress: () => void Linking.openSettings() },
    ],
  );
  return false;
}

export default function Scan() {
  const router = useRouter();
  const { palette } = useTheme();
  const db = useSQLiteContext();
  const reduceMotion = useReducedMotion();
  const { begin } = useImportDraft();
  const [phase, setPhase] = useState<Phase | null>(null);
  const [error, setError] = useState<{ message: string; hint: string } | null>(null);

  const busy = phase != null;
  // The phase only reaches the button on the next render, so a fast double tap
  // would otherwise start two scans (and two camera sheets).
  const inFlight = useRef(false);

  const handleProgress = (progress: ScanProgress) => setPhase(progress.stage);

  const run = async (task: () => Promise<void>) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setError(null);
    try {
      await task();
    } catch (cause) {
      const message = cause instanceof ScanError ? cause.hint : 'Something went wrong while reading the document.';
      setError({
        message: cause instanceof Error ? cause.message : 'The document could not be read.',
        hint: message,
      });
      setPhase(null);
    } finally {
      inFlight.current = false;
    }
  };

  const runFile = async (uri: string, kind: 'photo' | 'library' | 'file', label: string) => {
    setPhase('preparing');
    // Read the policy at scan time rather than from state: a tap that lands before
    // the preference finishes loading would otherwise keep a page the user turned
    // off (the old default was `true`).
    const keepSource = await keepSourceEnabled(db);
    const outcome = await scanFile({ uri, kind, label, keepSource, onProgress: handleProgress });
    setPhase('checking');
    await begin({
      engine: outcome.engine,
      result: outcome.result,
      source: outcome.source,
      evidenceMeta: outcome.evidenceMeta,
      image: outcome.image,
      pageCount: outcome.pageCount,
    });
    router.replace('/review');
  };

  const takePhoto = () =>
    run(async () => {
      if (!(await ensureCamera())) return;
      const result = await ImagePicker.launchCameraAsync({ quality: 1, allowsEditing: false, exif: false });
      if (result.canceled || !result.assets?.length) return;
      await runFile(result.assets[0].uri, 'photo', 'Photographed document');
    });

  const choosePhoto = () =>
    run(async () => {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 1,
        allowsEditing: false,
        exif: false,
      });
      if (result.canceled || !result.assets?.length) return;
      await runFile(result.assets[0].uri, 'library', result.assets[0].fileName ?? 'Chosen photo');
    });

  const chooseFile = () =>
    run(async () => {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'image/*', 'text/plain'],
        copyToCacheDirectory: true,
      });
      if (result.canceled || !result.assets?.length) return;
      const asset = result.assets[0];
      const isPDF = /\.pdf$/i.test(asset.name) || asset.mimeType === 'application/pdf';
      const isText = /\.(txt|md|markdown|csv)$/i.test(asset.name) || asset.mimeType?.startsWith('text/');

      if (isPDF || asset.mimeType?.startsWith('image/')) {
        await runFile(asset.uri, 'file', asset.name);
        return;
      }

      if (isText) {
        setPhase('preparing');
        const text = await new File(asset.uri).text();
        const outcome = await scanText({
          text,
          label: asset.name,
          onProgress: handleProgress,
        });
        setPhase('checking');
        await begin({
          engine: outcome.engine,
          result: outcome.result,
          source: outcome.source,
          evidenceMeta: outcome.evidenceMeta,
          image: outcome.image,
          pageCount: outcome.pageCount,
        });
        router.replace('/review');
        return;
      }

      throw new ScanError(
        `Pace cannot read ${asset.name}.`,
        'Choose a PDF, a photo, or a plain-text file.',
      );
    });

  if (phase) {
    const activeIndex = STAGES.findIndex((stage) => stage.key === phase);
    return (
      <Screen>
        <View style={{ flex: 1, justifyContent: 'center', paddingHorizontal: space.xl }}>
          <Animated.View entering={reduceMotion ? undefined : FadeIn.duration(200).easing(EASE_OUT)}>
            <Text style={[type.label, { color: palette.inkFaint, marginBottom: space.sm }]}>READING</Text>
            <Text style={[type.title, { color: palette.ink, marginBottom: space.xl }]}>One moment</Text>
            <View style={{ gap: space.lg }}>
              {STAGES.map((stage, index) => {
                const done = index < activeIndex;
                const current = index === activeIndex;
                return (
                  <View key={stage.key} style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
                    <View style={{ width: 20, alignItems: 'center' }}>
                      {done ? (
                        <Animated.View entering={reduceMotion ? undefined : FadeIn.duration(REVEAL_MS)}>
                          <Text style={{ color: palette.accent, fontSize: 16 }}>✓</Text>
                        </Animated.View>
                      ) : current ? (
                        <ActivityIndicator color={palette.inkMuted} />
                      ) : (
                        <View
                          style={{
                            width: 7,
                            height: 7,
                            borderRadius: 4,
                            borderWidth: hairline,
                            borderColor: palette.lineStrong,
                          }}
                        />
                      )}
                    </View>
                    <Text
                      style={[
                        type.body,
                        { color: done || current ? palette.ink : palette.inkFaint },
                        current && { fontWeight: '600' },
                      ]}
                    >
                      {stage.label}
                    </Text>
                  </View>
                );
              })}
            </View>
            <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.xxl }]}>
              The model runs on this device. Large documents take a few seconds longer than a single page.
            </Text>
          </Animated.View>
        </View>
      </Screen>
    );
  }

  return (
    <Screen scroll>
      <PageHeader
        title="Add a document"
        subtitle="A syllabus, a course catalog, or a transcript — a PDF or a photo. Pace reads the course, its grading breakdown and its meetings, or every course row on a transcript, on this iPhone."
        onBack={() => router.back()}
      />

      {error ? (
        <View style={{ paddingHorizontal: space.xl, paddingBottom: space.lg }}>
          <Notice tone="danger" title={error.message} body={error.hint} />
        </View>
      ) : null}

      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Animated.View entering={reduceMotion ? undefined : FadeInDown.duration(200).easing(EASE_OUT)}>
          <PressableScale
            accessibilityLabel="Take a photo of a document"
            onPress={() => void takePhoto()}
            style={{
              backgroundColor: palette.ink,
              borderRadius: radius.lg,
              padding: space.xl,
            }}
          >
            <Text style={[type.heading, { color: palette.background }]}>Take a photo</Text>
            <Text style={[type.caption, { color: palette.background, opacity: 0.7, marginTop: 4 }]}>
              Lay the page flat and fill the frame.
            </Text>
          </PressableScale>
        </Animated.View>

        <Option
          title="Choose a photo"
          subtitle="From your library, if you already photographed it."
          onPress={() => void choosePhoto()}
        />
        <Option
          title="Choose a file"
          subtitle="A PDF, a photo, or a plain-text export."
          onPress={() => void chooseFile()}
        />
      </View>

      <View style={{ padding: space.xl, paddingTop: space.xxl }}>
        <Card>
          <Text style={[type.label, { color: palette.inkFaint }]}>WHAT HAPPENS</Text>
          <Text style={[type.caption, { color: palette.inkMuted, marginTop: space.sm, lineHeight: 20 }]}>
            Apple Vision reads the text, page by page for a PDF, then Pace's own reader — running entirely on this
            iPhone — proposes the course, grading weights, meeting times and policies, or, for a transcript, every
            course row with its term, mark and credit. You review every value before anything is saved.
          </Text>
        </Card>
      </View>

      <View style={{ paddingHorizontal: space.xl }}>
        <Button title="Cancel" kind="quiet" onPress={() => router.back()} />
      </View>
    </Screen>
  );
}

function Option({
  title,
  subtitle,
  onPress,
}: {
  title: string;
  subtitle: string;
  onPress: () => void;
}) {
  const { palette } = useTheme();
  return (
    <PressableScale
      accessibilityLabel={title}
      onPress={onPress}
      style={{
        backgroundColor: palette.surface,
        borderRadius: radius.lg,
        borderWidth: hairline,
        borderColor: palette.line,
        padding: space.lg,
        flexDirection: 'row',
        alignItems: 'center',
      }}
    >
      <View style={{ flex: 1, paddingRight: space.md }}>
        <Text style={[type.bodyStrong, { color: palette.ink }]}>{title}</Text>
        <Text style={[type.caption, { color: palette.inkMuted, marginTop: 2 }]}>{subtitle}</Text>
      </View>
      <Text style={{ color: palette.inkFaint, fontSize: 18 }}>›</Text>
    </PressableScale>
  );
}
