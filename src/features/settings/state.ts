/**
 * What the Settings screen knows and does, independent of how it is drawn.
 *
 * Two renderings exist: the SwiftUI form on iOS and the portable one elsewhere.
 * They share this hook so a preference cannot be written one way on one platform
 * and another way on the other.
 */
import Constants from 'expo-constants';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useState } from 'react';
import { Alert } from 'react-native';

import {
  KEEP_SOURCE_KEY,
  allSourceUris,
  courseCount,
  deleteAllData,
  keepSourceEnabled,
  readGpaScaleValue,
  setSetting,
  storedDocumentCount,
} from '../../database/repo';
import { readManifestSummary, type ManifestSummary } from '../../engine/native/manifest';
import { parseScale } from '../../domain/gpa';
import { deleteAllStoredSources } from '../scan/storage';

export interface SettingsState {
  manifest: ManifestSummary | null;
  courses: number;
  documents: number;
  keepSource: boolean;
  gpaScale: string;
  version: string;
  setKeepSource: (next: boolean) => void;
  confirmDelete: () => void;
}

export function useSettings(): SettingsState {
  const db = useSQLiteContext();
  const router = useRouter();
  const [manifest, setManifest] = useState<ManifestSummary | null>(null);
  const [counts, setCounts] = useState({ courses: 0, documents: 0 });
  const [keepSource, setKeepSourceState] = useState(true);
  const [gpaScale, setGpaScale] = useState('Not set');

  useEffect(() => {
    void readManifestSummary().then(setManifest);
  }, []);

  useEffect(() => {
    void (async () => {
      setCounts({ courses: await courseCount(db), documents: await storedDocumentCount(db) });
      setKeepSourceState(await keepSourceEnabled(db));
      setGpaScale(parseScale(await readGpaScaleValue(db)) ? 'Custom' : 'Not set');
    })();
  }, [db]);

  const setKeepSource = useCallback(
    (next: boolean) => {
      setKeepSourceState(next);
      void setSetting(db, KEEP_SOURCE_KEY, next ? 'true' : 'false');
    },
    [db],
  );

  const confirmDelete = useCallback(() => {
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
  }, [db, router]);

  return {
    manifest,
    courses: counts.courses,
    documents: counts.documents,
    keepSource,
    gpaScale,
    version: Constants.expoConfig?.version ?? '1.0.0',
    setKeepSource,
    confirmDelete,
  };
}
