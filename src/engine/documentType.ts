/**
 * The document gate: is this a syllabus, a catalogue, a schedule, or not an
 * academic document at all?
 *
 * scikit-learn's char_wb TF-IDF has no ONNX converter, so the fitted pipeline is
 * flattened into vocabulary + IDFs + per-fold logistic weights + Platt curves
 * (`assets/models/doc-type.model` and `doc-type.bin`). This is the reference
 * reader for those numbers; `scripts/export_mobile_assets.py` checks it against
 * sklearn itself and records the max probability difference (1e-8).
 *
 * The priority here is not elegance, it is arithmetic that matches the artifact
 * the Python reader was measured with.
 */

import type { DocumentType } from './types';

interface Block {
  name: string;
  analyzer: string;
  ngram_min: number;
  ngram_max: number;
  lowercase: boolean;
  sublinear_tf: boolean;
  token_pattern: string | null;
  terms: string[];
  idf: number[];
  weight_offset: number;
  weight_width: number;
}

interface Fold {
  intercept: number[];
  calibration: [number, number][];
  class_order: number[];
}

export interface DocumentTypeModel {
  classes: string[];
  threshold: number;
  blocks: Block[];
  folds: Fold[];
  weights: { file: string; dtype: string; shape: number[] };
}

export interface DocumentTypeWeights {
  data: Float32Array;
  shape: number[];
}

const WORD_TOKEN = /[\p{L}\p{N}_]{2,}/gu;

function ngrams(text: string, block: Block): string[] {
  const low = block.lowercase ? text.toLowerCase() : text;
  const out: string[] = [];
  if (block.analyzer === 'word') {
    const tokens = low.match(WORD_TOKEN) ?? [];
    for (let size = block.ngram_min; size <= block.ngram_max; size += 1) {
      for (let start = 0; start + size <= tokens.length; start += 1) {
        out.push(tokens.slice(start, start + size).join(' '));
      }
    }
    return out;
  }
  for (const word of low.split(/\s+/)) {
    if (!word) continue;
    const padded = ` ${word} `;
    const characters = Array.from(padded);
    for (let size = block.ngram_min; size <= block.ngram_max; size += 1) {
      for (let start = 0; start + size <= characters.length; start += 1) {
        out.push(characters.slice(start, start + size).join(''));
      }
    }
  }
  return out;
}

/** `_prepare` in pace.doc_type: collapse whitespace, trim. */
export function prepareDocument(text: string): string {
  return (text || '').replace(/\s+/g, ' ').trim();
}

export class DocumentTypeClassifier {
  private readonly indices: Map<string, number>[];

  private readonly rowWidth: number;

  constructor(
    private readonly model: DocumentTypeModel,
    private readonly weights: DocumentTypeWeights,
  ) {
    this.indices = model.blocks.map(
      (block) => new Map(block.terms.map((term, index) => [term, index] as const)),
    );
    this.rowWidth = model.weights.shape[1] ?? weights.shape[1];
  }

  /** The sparse TF-IDF vector, concatenated across the fitted blocks. */
  private vectorize(text: string): Float32Array {
    const prepared = prepareDocument(text);
    const width = this.model.blocks.reduce((total, block) => total + block.weight_width, 0);
    const vector = new Float32Array(width);
    this.model.blocks.forEach((block, blockIndex) => {
      const index = this.indices[blockIndex];
      const counts = new Map<number, number>();
      for (const gram of ngrams(prepared, block)) {
        const position = index.get(gram);
        if (position === undefined) continue;
        counts.set(position, (counts.get(position) ?? 0) + 1);
      }
      const start = block.weight_offset;
      let square = 0;
      for (const [position, count] of counts) {
        const value = block.sublinear_tf ? 1 + Math.log(count) : count;
        const weighted = value * block.idf[position];
        vector[start + position] = weighted;
        square += weighted * weighted;
      }
      const norm = Math.sqrt(square);
      if (norm > 0) {
        for (const position of counts.keys()) vector[start + position] /= norm;
      }
    });
    return vector;
  }

  probabilities(text: string): Record<string, number> {
    const vector = this.vectorize(text);
    const totals = new Array(this.model.classes.length).fill(0);
    const { data } = this.weights;
    const rowWidth = this.rowWidth;
    let row = 0;
    for (const fold of this.model.folds) {
      const rows = fold.intercept.length;
      const calibrated = new Array(rows).fill(0);
      for (let position = 0; position < rows; position += 1) {
        let decision = fold.intercept[position];
        const base = (row + position) * rowWidth;
        for (let column = 0; column < rowWidth; column += 1) {
          const weight = data[base + column];
          if (weight !== 0 && vector[column] !== 0) decision += weight * vector[column];
        }
        const [a, b] = fold.calibration[position];
        // sklearn: expit(-(a * decision + b)).
        calibrated[position] = 1 / (1 + Math.exp(a * decision + b));
      }
      const total = calibrated.reduce((sum, value) => sum + value, 0);
      for (let position = 0; position < rows; position += 1) {
        const target = fold.class_order[position] ?? position;
        totals[target] += total > 0 ? calibrated[position] / total : 1 / rows;
      }
      row += rows;
    }
    const count = this.model.folds.length || 1;
    const out: Record<string, number> = {};
    this.model.classes.forEach((name, index) => {
      out[name] = totals[index] / count;
    });
    return out;
  }

  classify(text: string): DocumentType {
    const probabilities = this.probabilities(text);
    const ranked = Object.entries(probabilities)
      .map(([label, value]) => ({ label, probability: value }))
      .sort((left, right) => right.probability - left.probability);
    const [label, confidence] = [ranked[0].label, ranked[0].probability];
    const academic = label !== 'other' && confidence >= this.model.threshold;
    return {
      label: academic ? label : label === 'other' ? 'other' : 'unknown',
      proposedLabel: label,
      confidence: Math.round(confidence * 10000) / 10000,
      academic,
      abstained: !academic,
      top: ranked.slice(0, 3).map((entry) => ({
        label: entry.label,
        probability: Math.round(entry.probability * 10000) / 10000,
      })),
    };
  }
}
