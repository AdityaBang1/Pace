/**
 * The section classifier, running locally through ONNX Runtime.
 *
 * One text in, one label and a calibrated probability out. Temperature comes
 * from the checkpoint (trained with it, exported with it) and is applied here,
 * because the graph returns logits.
 */

import { WordPieceTokenizer } from './tokenizer';
import type { SessionLike } from './runtime';

export interface SectionModelMeta {
  labels: string[];
  tasks: Record<string, string>;
  maxLength: number;
  temperature: number;
}

export interface SectionPrediction {
  label: string;
  task: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export class SectionClassifier {
  constructor(
    private readonly session: SessionLike,
    private readonly tokenizer: WordPieceTokenizer,
    private readonly meta: SectionModelMeta,
  ) {}

  /** Softmax over temperature-scaled logits, like `FieldEngine.probabilities`. */
  private softmax(logits: Float32Array): number[] {
    const temperature = Math.max(this.meta.temperature, 1e-6);
    let max = -Infinity;
    for (const value of logits) max = Math.max(max, value / temperature);
    let total = 0;
    const out: number[] = [];
    for (const value of logits) {
      const scaled = Math.exp(value / temperature - max);
      out.push(scaled);
      total += scaled;
    }
    return out.map((value) => value / total);
  }

  async probabilities(texts: string[]): Promise<number[][]> {
    if (!texts.length) return [];
    // `padding=True` in the Python reader pads to the longest item in the batch;
    // batching by longest text keeps the arithmetic identical and the tensors
    // smaller than a fixed 256-token window.
    const encoded = texts.map((text) => this.tokenizer.encode(text.trim() ? text : ' ', this.meta.maxLength));
    const width = Math.max(...encoded.map((row) => row.ids.length));
    const ids = new BigInt64Array(encoded.length * width);
    const mask = new BigInt64Array(encoded.length * width);
    encoded.forEach((row, index) => {
      const padded = this.tokenizer.encode(texts[index].trim() ? texts[index] : ' ', width);
      ids.set(padded.ids, index * width);
      mask.set(padded.mask, index * width);
    });
    const output = await this.session.run({ input_ids: ids, attention_mask: mask }, [encoded.length, width]);
    const logits = output.logits;
    return encoded.map((_, row) =>
      this.softmax(logits.subarray(row * this.meta.labels.length, (row + 1) * this.meta.labels.length)),
    );
  }

  async classify(text: string): Promise<SectionPrediction> {
    const [row] = await this.probabilities([text]);
    let best = 0;
    row.forEach((value, index) => {
      if (value > row[best]) best = index;
    });
    const probabilities: Record<string, number> = {};
    this.meta.labels.forEach((label, index) => {
      probabilities[label] = row[index];
    });
    const label = this.meta.labels[best];
    return {
      label,
      task: this.meta.tasks[label] ?? 'other',
      confidence: row[best],
      probabilities,
    };
  }
}
