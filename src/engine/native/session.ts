/**
 * ONNX Runtime on the phone, behind the engine's runtime boundary.
 *
 * The engine speaks in `SessionLike`; this file is the only place that knows the
 * runtime is `onnxruntime-react-native`. Swapping in ExecuTorch, a Core ML bridge
 * or an int8 graph later means replacing this file, not the reader.
 */
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

import type { SessionFactory, SessionLike } from '../runtime';

/** ORT's native bridge opens bare paths; the JS side hands it `file://` URIs. */
function asPath(uri: string): string {
  return uri.startsWith('file://') ? decodeURI(uri.slice('file://'.length)) : uri;
}

export const createNativeSession: SessionFactory = async (modelPath: string): Promise<SessionLike> => {
  const session = await InferenceSession.create(asPath(modelPath), {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });
  return {
    async run(inputs, dims) {
      const outputs = await session.run({
        input_ids: new Tensor('int64', inputs.input_ids, dims),
        attention_mask: new Tensor('int64', inputs.attention_mask, dims),
      });
      const logits = outputs.logits;
      if (!logits) throw new Error('model returned no logits');
      return { logits: logits.data as Float32Array };
    },
    async release() {
      // Frees the native session; `release` is the ORT API, not `dispose`.
      await session.release();
    },
  };
};
