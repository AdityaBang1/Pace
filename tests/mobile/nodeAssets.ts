/**
 * The developer-machine adapters: the same engine, reading the same artifacts
 * from disk and running the same ONNX graph through `onnxruntime-node`.
 *
 * The point of this file is that it is the *only* thing that differs between the
 * harness and the phone. If the harness needed engine changes to run, it would be
 * testing a different engine.
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';

import type { EngineAssets } from '../../src/engine/PaceEngine';
import type { SessionFactory, SessionLike } from '../../src/engine/runtime';

const ROOT = path.resolve(__dirname, '../..');
const MODELS = path.join(ROOT, 'assets', 'models');

export const assets: EngineAssets = {
  async readText(name) {
    return fs.readFile(path.join(MODELS, name), 'utf8');
  },
  async readBinary(name) {
    const buffer = await fs.readFile(path.join(MODELS, name));
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  },
  async modelUri(name) {
    return path.join(MODELS, name);
  },
};

export const createSession: SessionFactory = async (modelPath: string): Promise<SessionLike> => {
  const session = await ort.InferenceSession.create(modelPath);
  return {
    async run(inputs, dims) {
      const feeds = {
        input_ids: new ort.Tensor('int64', inputs.input_ids, dims),
        attention_mask: new ort.Tensor('int64', inputs.attention_mask, dims),
      };
      const output = await session.run(feeds);
      const logits = output.logits;
      return { logits: logits.data as Float32Array };
    },
  };
};

export async function readFixture(name: string): Promise<string> {
  return fs.readFile(path.join(ROOT, 'fixtures', name), 'utf8');
}

export async function readJson<T>(...parts: string[]): Promise<T> {
  return JSON.parse(await fs.readFile(path.join(ROOT, ...parts), 'utf8')) as T;
}

export const root = ROOT;
