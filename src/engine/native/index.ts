/**
 * The phone-side wiring of the reader: bundled artifacts on disk, ONNX Runtime
 * for the graph. Both adapters are created once per process.
 */
import { PaceEngine, type EngineAssets } from '../PaceEngine';

import {
  ensureModelFiles,
  modelPath,
  readModelBinary,
  readModelText,
  type PrepareProgress,
} from './modelFiles';
import { createNativeSession } from './session';

export const nativeAssets: EngineAssets = {
  readText: readModelText,
  readBinary: readModelBinary,
  modelUri: async (name) => modelPath(name),
};

let loaded: Promise<PaceEngine> | null = null;

export interface InitializeHooks {
  onFile?: (progress: PrepareProgress) => void;
}

/**
 * Load the engine, copying the model bundle on first run.
 *
 * Initialization is lazy so app start is not blocked: the home screen draws
 * immediately, and the first scan pays for the copy plus one graph load.
 */
export function loadEngine(hooks: InitializeHooks = {}): Promise<PaceEngine> {
  if (!loaded) {
    loaded = (async () => {
      await ensureModelFiles(hooks.onFile);
      return PaceEngine.create(nativeAssets, createNativeSession);
    })().catch((error) => {
      loaded = null;
      throw error;
    });
  }
  return loaded;
}

/** Drop the cached engine (debug screen: forces a fresh load and re-timing). */
export function releaseEngine(): void {
  loaded = null;
}

export { ensureModelFiles, MODEL_FILE_NAMES } from './modelFiles';
