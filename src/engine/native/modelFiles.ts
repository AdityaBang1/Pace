/**
 * Getting the model bundle onto the phone.
 *
 * Metro ships the artifacts inside the app bundle, but ONNX Runtime and the
 * tokenizer want ordinary filesystem paths, so on first launch every artifact is
 * copied once into `documentDirectory/models`. A completion marker records the
 * byte length of every artifact, and the copy is skipped only when the marker and
 * all those lengths still agree: an app killed halfway through the 90 MB copy
 * leaves a truncated file that exists, and trusting `exists` would then load a
 * corrupt graph on every later launch.
 *
 * Only `ensureModelFiles()` touches the filesystem; everything else in the app
 * reads through `readText` / `readBinary` / `modelUri`, which is exactly the
 * surface the Node parity harness implements too.
 */
import { Asset } from 'expo-asset';
import { Directory, File, Paths } from 'expo-file-system';

/**
 * Every artifact the engine loads, by name inside `assets/models`.
 *
 * A literal `require` per file is deliberate: Metro resolves static requires at
 * bundle time, so a computed path would ship an empty bundle and fail on device
 * with a missing asset rather than at build time.
 */
const BUNDLED: Record<string, number> = {
  'manifest.model': require('../../../assets/models/manifest.model'),
  'rules.model': require('../../../assets/models/rules.model'),
  'doc-type.model': require('../../../assets/models/doc-type.model'),
  'doc-type.bin': require('../../../assets/models/doc-type.bin'),
  'matcher.model': require('../../../assets/models/matcher.model'),
  'section-model.onnx': require('../../../assets/models/section-model.onnx'),
  'context-model.onnx': require('../../../assets/models/context-model.onnx'),
  'tokenizer/vocab.model': require('../../../assets/models/tokenizer/vocab.model'),
};

export const MODEL_FILE_NAMES = Object.keys(BUNDLED);

function modelDirectory(): Directory {
  return new Directory(Paths.document, 'models');
}

function destination(name: string): File {
  const parts = name.split('/');
  const fileName = parts.pop() as string;
  const directory = parts.reduce((current, part) => new Directory(current, part), modelDirectory());
  if (!directory.exists) directory.create({ intermediates: true, idempotent: true });
  return new File(directory, fileName);
}

export interface PrepareProgress {
  file: string;
  index: number;
  total: number;
}

const MARKER_NAME = '.ready.model';

interface ReadyMarker {
  version: 1;
  sizes: Record<string, number>;
}

let prepared: Promise<void> | null = null;

/**
 * Copy the bundle into private storage. Safe to call many times; the work runs
 * once per process and concurrent callers await the same promise.
 */
export function ensureModelFiles(onProgress?: (progress: PrepareProgress) => void): Promise<void> {
  if (!prepared) {
    prepared = copyAll(onProgress).catch((error) => {
      // A failed copy must not poison later attempts (e.g. low disk space that
      // the user then clears), so the memo is dropped before rethrowing.
      prepared = null;
      throw error;
    });
  }
  return prepared;
}

async function readMarker(directory: Directory): Promise<ReadyMarker | null> {
  try {
    const file = new File(directory, MARKER_NAME);
    if (!file.exists) return null;
    const parsed = JSON.parse(await file.text()) as Partial<ReadyMarker>;
    if (parsed.version !== 1 || !parsed.sizes || typeof parsed.sizes !== 'object') return null;
    return parsed as ReadyMarker;
  } catch {
    // An unreadable marker is the same as no marker: copy again.
    return null;
  }
}

async function matches(marker: ReadyMarker): Promise<boolean> {
  for (const name of MODEL_FILE_NAMES) {
    const expected = marker.sizes[name];
    if (typeof expected !== 'number') return false;
    const target = destination(name);
    if (!target.exists || target.size !== expected) return false;
  }
  return true;
}

async function copyAll(onProgress?: (progress: PrepareProgress) => void): Promise<void> {
  const directory = modelDirectory();
  if (!directory.exists) directory.create({ intermediates: true, idempotent: true });

  const marker = await readMarker(directory);
  if (marker && (await matches(marker))) return;

  const names = MODEL_FILE_NAMES;
  const sizes: Record<string, number> = {};
  for (let index = 0; index < names.length; index += 1) {
    const name = names[index];
    const target = destination(name);
    onProgress?.({ file: name, index, total: names.length });
    const asset = Asset.fromModule(BUNDLED[name]);
    if (!asset.localUri) await asset.downloadAsync();
    const localUri = asset.localUri;
    if (!localUri) throw new Error(`bundled asset ${name} has no local file`);
    const source = new File(localUri);
    // A file of the wrong length is a leftover from an interrupted copy, not a
    // model: overwrite it rather than loading it.
    if (!target.exists || target.size !== source.size) {
      await source.copy(target, { overwrite: true });
    }
    if (target.size !== source.size) {
      throw new Error(`model file ${name} copied ${target.size} of ${source.size} bytes`);
    }
    sizes[name] = target.size;
  }

  // Written last, and only after every length checked out: its presence is what
  // later launches trust. Create it explicitly because SDK 58 documents `write`
  // as writing an existing File; relying on implicit creation makes the first
  // model preparation path platform-dependent.
  const ready = new File(directory, MARKER_NAME);
  if (!ready.exists) ready.create();
  await ready.write(JSON.stringify({ version: 1, sizes } satisfies ReadyMarker));
}

/** Path for a model file once `ensureModelFiles` has run. */
export function modelPath(name: string): string {
  return destination(name).uri;
}

export async function readModelText(name: string): Promise<string> {
  await ensureModelFiles();
  return destination(name).text();
}

export async function readModelBinary(name: string): Promise<Uint8Array> {
  await ensureModelFiles();
  return destination(name).bytes();
}
