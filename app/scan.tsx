import { Button as NativeButton, Image as NativeImage, Section, Text as NativeText } from '@expo/ui/swift-ui';
import { aspectRatio, disabled, font, foregroundStyle, frame, padding, resizable } from '@expo/ui/swift-ui/modifiers';
import * as DocumentPicker from 'expo-document-picker';
import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';
import { Alert, Linking } from 'react-native';

import { renderPage, type PaceVisionPreview } from '../modules/pace-vision';
import { keepSourceEnabled } from '../src/database/repo';
import { useImportDraft } from '../src/features/import/ImportContext';
import { ScanError, scanFile, scanText, type ScanProgress, type ScanStage } from '../src/features/scan/pipeline';
import { LabeledContent, NativeScreen, Value } from '../src/ui/native';
import { useTheme } from '../src/ui/useTheme';

type Phase = ScanStage | 'checking';

const STAGES: { key: Phase; label: string }[] = [
  { key: 'preparing', label: 'Preparing the reader' },
  { key: 'reading', label: 'Reading the text' },
  { key: 'understanding', label: 'Understanding the document' },
  { key: 'checking', label: 'Checking the result' },
];

/** What was picked, held until the person says it is the document they meant. */
interface Pending {
  uri: string;
  kind: 'photo' | 'library' | 'file';
  label: string;
  /** Absent for a plain-text file, which has no page to show. */
  preview: PaceVisionPreview | null;
  previewFailed: boolean;
}

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

function baseName(label: string): string {
  const parts = label.split('/');
  return parts[parts.length - 1] || label;
}

/**
 * Add a document: choose it, look at it, and only then read it.
 *
 * The confirmation is not ceremony. A PDF is opened at whatever page the picker
 * remembers, a library holds six photographs of the same syllabus, and reading one
 * costs the model load plus a second a page — so the page is drawn first and the
 * reader does not run until the person has said "that one". Everything on this
 * screen is a system control: buttons, rows, the page itself.
 */
export default function Scan() {
  const router = useRouter();
  const { palette } = useTheme();
  const db = useSQLiteContext();
  const { begin } = useImportDraft();
  const [phase, setPhase] = useState<Phase | null>(null);
  const [error, setError] = useState<{ message: string; hint: string } | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const busy = phase != null;
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

  const runFile = async (file: { uri: string; kind: 'photo' | 'library' | 'file'; label: string }) => {
    setPending(null);
    setPhase('preparing');
    const keepSource = await keepSourceEnabled(db);
    const outcome = await scanFile({
      uri: file.uri,
      kind: file.kind,
      label: file.label,
      keepSource,
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
  };

  /** Hold a picked file and draw its first page before anything is read. */
  const propose = async (file: { uri: string; kind: 'photo' | 'library' | 'file'; label: string }) => {
    setError(null);
    setPending({ ...file, preview: null, previewFailed: false });
    setPreviewing(true);
    try {
      const preview = await renderPage(file.uri, 0);
      setPending((current) => (current && current.uri === file.uri ? { ...current, preview } : current));
    } catch {
      // A page that cannot be drawn is not a page that cannot be read: the
      // confirmation says so and still offers to read the file.
      setPending((current) => (current && current.uri === file.uri ? { ...current, previewFailed: true } : current));
    } finally {
      setPreviewing(false);
    }
  };

  const takePhoto = () =>
    run(async () => {
      if (!(await ensureCamera())) return;
      const result = await ImagePicker.launchCameraAsync({ quality: 1, allowsEditing: false, exif: false });
      if (result.canceled || !result.assets?.length) return;
      await propose({ uri: result.assets[0].uri, kind: 'photo', label: 'Photographed document' });
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
      await propose({
        uri: result.assets[0].uri,
        kind: 'library',
        label: result.assets[0].fileName ?? 'Chosen photo',
      });
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
        await propose({ uri: asset.uri, kind: 'file', label: asset.name });
        return;
      }

      if (isText) {
        setPhase('preparing');
        const text = await new File(asset.uri).text();
        const outcome = await scanText({ text, label: asset.name, onProgress: handleProgress });
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

  const activeIndex = phase ? STAGES.findIndex((stage) => stage.key === phase) : -1;

  useEffect(() => {
    // A picker can be dismissed while a page is still being drawn; nothing to do
    // then, and the state it would have written belongs to a document that is gone.
    return () => setPreviewing(false);
  }, []);

  return (
    <NativeScreen
      topInset={false}
      title="Add a document"
      subtitle="Read on this iPhone. Nothing leaves it."
      onBack={() => router.back()}
    >
      {error ? (
        <Section title={error.message} footer={<NativeText>{error.hint}</NativeText>}>
          <NativeButton label="Try again" systemImage="arrow.clockwise" onPress={() => setError(null)} />
        </Section>
      ) : null}

      {pending ? (
        <>
          <Section
            title="Is this the right document?"
            footer={
              <NativeText>
                {pending.previewFailed
                  ? 'This file has no page Pace can draw. It can still be read.'
                  : 'Reading happens on this iPhone. Nothing leaves it.'}
              </NativeText>
            }
          >
            {pending.preview ? (
              <NativeImage
                uiImage={pending.preview.uri}
                modifiers={[resizable(), aspectRatio({ contentMode: 'fit' }), frame({ maxHeight: 340 })]}
              />
            ) : (
              <NativeText modifiers={[foregroundStyle(palette.inkMuted)]}>
                {previewing ? 'Drawing the page…' : 'No preview'}
              </NativeText>
            )}
            <LabeledContent label="File">
              <Value>{baseName(pending.label)}</Value>
            </LabeledContent>
            {pending.preview ? (
              <LabeledContent label="Pages">
                <Value>{String(pending.preview.pages)}</Value>
              </LabeledContent>
            ) : null}
          </Section>

          <Section>
            <NativeButton
              label="Read this document"
              systemImage="text.viewfinder"
              modifiers={[disabled(previewing)]}
              onPress={() => void run(() => runFile(pending))}
            />
            <NativeButton label="Choose another" systemImage="arrow.uturn.backward" onPress={() => setPending(null)} />
          </Section>
        </>
      ) : busy ? (
        <Section
          title="Reading"
          footer={
            <NativeText>
              The model runs on this device. Large documents take a few seconds longer than a single page.
            </NativeText>
          }
        >
          {STAGES.map((stage, index) => (
            <LabeledContent
              key={stage.key}
              label={stage.label}
            >
              <NativeText
                modifiers={[
                  foregroundStyle(index <= activeIndex ? palette.accent : palette.inkFaint),
                  font({ textStyle: 'body', weight: index === activeIndex ? 'semibold' : 'regular' }),
                ]}
              >
                {index < activeIndex ? '✓' : index === activeIndex ? '…' : ''}
              </NativeText>
            </LabeledContent>
          ))}
        </Section>
      ) : (
        <>
          <Section
            title="Add"
            footer={
              <NativeText>
                A syllabus, a course catalog, or a transcript — a PDF or a photo. Pace reads the course, its grading
                breakdown and its meetings, or every course row on a transcript, on this iPhone.
              </NativeText>
            }
          >
            <NativeButton
              label="Take a photo"
              systemImage="camera"
              onPress={() => void takePhoto()}
            />
            <NativeButton
              label="Choose a photo"
              systemImage="photo.on.rectangle"
              onPress={() => void choosePhoto()}
            />
            <NativeButton
              label="Choose a file"
              systemImage="doc"
              onPress={() => void chooseFile()}
            />
          </Section>

          <Section title="What happens">
            <NativeText modifiers={[foregroundStyle(palette.inkMuted), padding({ bottom: 6 })]}>
              Apple Vision reads the text, page by page for a PDF, then Pace's own reader proposes the course, its
              grading weights, meeting times and policies — or, for a transcript, every course row with its term,
              mark and credit. You review every value before anything is saved.
            </NativeText>
          </Section>
        </>
      )}
    </NativeScreen>
  );
}
