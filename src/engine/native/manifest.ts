/**
 * The manifest without the model bundle.
 *
 * Settings needs to show which reader is frozen, and doing that through
 * `loadEngine()` would copy 180 MB of graphs just to print a version string. The
 * manifest is an 11 KB text asset, so it is read directly.
 */
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';

export interface ManifestSummary {
  engineVersion: string;
  schemaVersion: string;
  frozenOn?: string;
  why?: string;
  models: {
    section: { id: string; file: string; parameters: number; max_length: number };
    context: { id: string; file: string };
    documentType: { id: string; classes: string[] };
    entityResolution: { id: string; threshold: number };
  };
}

export async function readManifestSummary(): Promise<ManifestSummary | null> {
  try {
    const asset = Asset.fromModule(require('../../../assets/models/manifest.model'));
    if (!asset.localUri) await asset.downloadAsync();
    if (!asset.localUri) return null;
    const text = await new File(asset.localUri).text();
    return JSON.parse(text) as ManifestSummary;
  } catch {
    return null;
  }
}
