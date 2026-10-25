/**
 * The runtime boundary.
 *
 * The engine never imports `onnxruntime-react-native` directly. It takes a
 * factory, so the exact same engine code runs in a Node parity harness on the
 * developer machine and on the phone, and so a future runtime (ExecuTorch,
 * Core ML, a smaller int8 graph) is a new factory rather than a rewrite.
 */

export interface SessionOutputs {
  logits: Float32Array;
}

export interface SessionLike {
  run(inputs: Record<string, BigInt64Array>, dims: [number, number]): Promise<SessionOutputs>;
  release?(): Promise<void>;
}

export type SessionFactory = (modelPath: string) => Promise<SessionLike>;
