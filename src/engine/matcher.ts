/**
 * "Is this the same course?" -- the pairwise matcher, ported from
 * `pace.entity_resolution`.
 *
 * Measured: precision 0.9969, recall 0.917, F1 0.9553 on 2,954 held-out pairs,
 * with 5 false merges (0.42%). The decision policy is part of that measurement,
 * so it is part of the port: a level or name-number conflict vetoes, and two
 * identical names with different catalogue numbers return *uncertain* rather
 * than merging. The app never merges automatically; it proposes.
 *
 * The feature set below is a line-by-line port, including the details that are
 * easy to get wrong: the catalogue-code pattern needs two to four digits (so
 * "CHEM 1A" is a title, not a code), TF-IDF runs over shared character n-grams
 * with raw term counts, and "Advanced Placement" must not also count as the
 * level "advanced".
 */

export interface MatcherModel {
  features: string[];
  coefficients: number[];
  intercept: number;
  threshold: number;
}

export type MatchDecision = 'same' | 'different' | 'uncertain';

export interface MatchResult {
  left: string;
  right: string;
  probability: number;
  decision: MatchDecision;
  confidence: number;
}

const CODE_SOURCE = '\\b([A-Z]{2,6})\\s?-?\\s?(\\d{2,4}[A-Z]?)\\b';
const CODE = new RegExp(CODE_SOURCE, 'g');
const MIT_CODE = /^(\d{1,2})\.(\d{2,3}[A-Z]?)/;

const LEVEL_WORDS: Record<string, string> = {
  'advanced placement': 'ap',
  ap: 'ap',
  honors: 'honors',
  honours: 'honors',
  advanced: 'advanced',
  accelerated: 'accelerated',
  regular: 'regular',
  remedial: 'remedial',
  'college prep': 'regular',
  cp: 'regular',
};

const ROMAN: Record<string, number> = { i: 1, ii: 2, iii: 3, iv: 4, v: 5 };
const SECTION_LETTERS = ['a', 'b', 'c', 'ab', 'bc'];

const ABBREVIATIONS: Record<string, string> = {
  intro: 'introduction',
  introduct: 'introduction',
  chem: 'chemistry',
  bio: 'biology',
  calc: 'calculus',
  phys: 'physics',
  econ: 'economics',
  psych: 'psychology',
  stats: 'statistics',
  stat: 'statistics',
  math: 'mathematics',
  eng: 'english',
  engr: 'engineering',
  cs: 'computer science',
  cogsci: 'cognitive science',
  ap: 'advanced placement',
  hon: 'honors',
  honours: 'honors',
  sem: 'seminar',
  lab: 'laboratory',
  dept: 'department',
};

function expand(text: string): string {
  return text
    .split(/\s+/)
    .map((word) => ABBREVIATIONS[word.toLowerCase().replace(/^[,.]|[,.]$/g, '')] ?? word)
    .join(' ');
}

/** `_norm`: catalogue codes and punctuation out, abbreviations expanded. */
export function normalizeTitle(text: string): string {
  let value = text.replace(new RegExp(CODE_SOURCE, 'g'), ' ');
  value = value.replace(/\b\d{1,2}\.\d{2,3}[a-z]?\b/gi, ' ');
  value = expand(value.toLowerCase());
  // Python's `_norm` ends with `re.sub(r"[^a-z0-9 ]", " ", text).strip()`: the
  // punctuation becomes a space but the spaces are *not* collapsed, so a removed
  // colon leaves two. Collapsing them shortened the string the sequence ratio
  // runs over (24 vs 23 characters) and moved a match probability by 0.003.
  return value.replace(/[^a-z0-9 ]/g, ' ').trim();
}

function tokens(text: string): Set<string> {
  return new Set(normalizeTitle(text).split(' ').filter((token) => token.length > 1));
}

function codeNumbers(text: string): Set<string> {
  return new Set(Array.from(text.toUpperCase().matchAll(new RegExp(CODE_SOURCE, 'g')),
    (match) => match[2]));
}

function contentNumbers(text: string): Set<string> {
  return new Set(normalizeTitle(text).match(/\d+/g) ?? []);
}

function levelSignature(text: string): string[] {
  const folded = text.toLowerCase();
  const marks = new Set<string>();
  for (const [word, canonical] of Object.entries(LEVEL_WORDS)) {
    if (new RegExp(`\\b${escapeRegExp(word)}\\b`).test(folded)) marks.add(canonical);
  }
  for (const match of folded.matchAll(/\b(i{1,3}v?|iv|v)\b/g)) {
    if (match[1] in ROMAN) marks.add(`roman${ROMAN[match[1]]}`);
  }
  for (const letter of SECTION_LETTERS) {
    if (new RegExp(`\\b${letter}\\b`).test(folded)) marks.add(`section${letter}`);
  }
  if (marks.has('ap')) marks.delete('advanced');
  return Array.from(marks).sort();
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function difference(left: Set<string>, right: Set<string>): boolean {
  if (left.size !== right.size) return true;
  for (const value of left) if (!right.has(value)) return true;
  return false;
}

function jaccard(left: Set<string>, right: Set<string>): number {
  let shared = 0;
  for (const token of left) if (right.has(token)) shared += 1;
  const union = left.size + right.size - shared;
  return union ? shared / union : 0;
}

function containment(left: Set<string>, right: Set<string>): number {
  if (!left.size || !right.size) return 0;
  let shared = 0;
  for (const token of left) if (right.has(token)) shared += 1;
  return shared / Math.min(left.size, right.size);
}

/**
 * `difflib.SequenceMatcher.ratio()`, which is *not* an LCS ratio.
 *
 * It finds the longest matching block, recurses into what is left on either
 * side, and scores twice the matched characters over the total. An LCS agrees on
 * most course pairs and then disagrees on exactly the near-miss ones this
 * feature exists for ("MATH 1A: Calculus" vs "MATH 1A Calculus I": 0.9388 vs
 * 0.9583), which moved the probability by ~0.003. Ported so the number is the
 * number the threshold was fitted on.
 */
function sequenceRatio(left: string, right: string): number {
  const a = normalizeTitle(left);
  const b = normalizeTitle(right);
  if (!a.length || !b.length) return 0;
  const positions = new Map<string, number[]>();
  for (let j = 0; j < b.length; j += 1) {
    const list = positions.get(b[j]);
    if (list) list.push(j);
    else positions.set(b[j], [j]);
  }
  let matched = 0;
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop() as [number, number, number, number];
    let besti = alo;
    let bestj = blo;
    let bestsize = 0;
    let previous = new Map<number, number>();
    for (let i = alo; i < ahi; i += 1) {
      const current = new Map<number, number>();
      for (const j of positions.get(a[i]) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const size = (previous.get(j - 1) ?? 0) + 1;
        current.set(j, size);
        if (size > bestsize) {
          besti = i - size + 1;
          bestj = j - size + 1;
          bestsize = size;
        }
      }
      previous = current;
    }
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti -= 1;
      bestj -= 1;
      bestsize += 1;
    }
    while (
      besti + bestsize < ahi &&
      bestj + bestsize < bhi &&
      a[besti + bestsize] === b[bestj + bestsize]
    ) {
      bestsize += 1;
    }
    if (!bestsize) continue;
    matched += bestsize;
    queue.push([alo, besti, blo, bestj]);
    queue.push([besti + bestsize, ahi, bestj + bestsize, bhi]);
  }
  return (2 * matched) / (a.length + b.length);
}

function prefixOverlap(left: Set<string>, right: Set<string>): number {
  if (!left.size || !right.size) return 0;
  const [smaller, larger] = left.size <= right.size ? [left, right] : [right, left];
  let hits = 0;
  for (const token of smaller) {
    if (token.length < 3) continue;
    for (const other of larger) {
      if (other.length >= 3 && (other.startsWith(token) || token.startsWith(other))) {
        hits += 1;
        break;
      }
    }
  }
  // Short tokens ("ab", "bc") cannot abbreviate anything, but they still count
  // against the denominator. Skipping them from both made AB/BC score 1.0.
  return hits / smaller.size;
}

function charGrams(text: string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const word of normalizeTitle(text).split(' ')) {
    if (!word) continue;
    const padded = ` ${word} `;
    for (let size = 3; size <= 5; size += 1) {
      for (let start = 0; start + size <= padded.length; start += 1) {
        const gram = padded.slice(start, start + size);
        counts.set(gram, (counts.get(gram) ?? 0) + 1);
      }
    }
  }
  return counts;
}

/**
 * Character n-gram cosine, fitted across the batch of pairs.
 *
 * `pairs_features` fits one TF-IDF vectoriser over every string in the batch
 * (`min_df=2`), so the feature depends on the *batch*: a gram unique to one
 * title is pruned only when no other title in the batch has it. Reproducing
 * that is not an accident -- a single-pair batch and a five-pair batch give
 * different numbers, and the model was trained on the batch form.
 *
 * The app calls this with one left-hand course and every candidate it might
 * match, which is exactly the shape the model was trained on.
 */
function charCosineMatrix(texts: string[]): number[][] {
  const counts = texts.map(charGrams);
  const documentFrequency = new Map<string, number>();
  for (const row of counts) {
    for (const gram of row.keys()) {
      documentFrequency.set(gram, (documentFrequency.get(gram) ?? 0) + 1);
    }
  }
  const n = texts.length;
  const idf = new Map<string, number>();
  for (const [gram, df] of documentFrequency) {
    if (df < 2) continue;
    idf.set(gram, Math.log((1 + n) / (1 + df)) + 1);
  }
  const vectors = counts.map((row) => {
    const vector = new Map<string, number>();
    let square = 0;
    for (const [gram, count] of row) {
      const weight = idf.get(gram);
      if (weight === undefined) continue;
      const value = count * weight;
      vector.set(gram, value);
      square += value * value;
    }
    const norm = Math.sqrt(square);
    if (norm) for (const [gram, value] of vector) vector.set(gram, value / norm);
    return vector;
  });
  return texts.map((_, i) =>
    texts.map((__, j) => {
      let dot = 0;
      for (const [gram, value] of vectors[i]) {
        const other = vectors[j].get(gram);
        if (other !== undefined) dot += value * other;
      }
      return dot;
    }),
  );
}

/** The 14 features for many pairs, in the order the coefficients expect. */
export function featuresForPairs(pairs: [string, string][]): number[][] {
  const flattened = pairs.flatMap(([left, right]) => [left, right]);
  const cosine = charCosineMatrix(flattened.map(normalizeTitle));
  return pairs.map(([left, right], index) =>
    featuresWithCosine(left, right, cosine[index * 2][index * 2 + 1]),
  );
}

/** The 14 features, in the order the exported coefficients expect. */
export function features(left: string, right: string): number[] {
  return featuresForPairs([[left, right]])[0];
}

function featuresWithCosine(left: string, right: string, cosine: number): number[] {
  const tokenLeft = tokens(left);
  const tokenRight = tokens(right);
  const numberLeft = codeNumbers(left);
  const numberRight = codeNumbers(right);
  const contentLeft = contentNumbers(left);
  const contentRight = contentNumbers(right);
  const levelsLeft = levelSignature(left);
  const levelsRight = levelSignature(right);
  let shared = 0;
  for (const token of tokenLeft) if (tokenRight.has(token)) shared += 1;
  let numberShared = 0;
  for (const number of numberLeft) if (numberRight.has(number)) numberShared += 1;
  const sameLevel = levelsLeft.length > 0 &&
    levelsLeft.length === levelsRight.length &&
    levelsLeft.every((mark) => levelsRight.includes(mark));
  // `word_overlap` is the share of the *raw* word count, not the normalised one:
  // "AP Chem" and "AP Chemistry" share one printed word out of two.
  const leftWords = new Set(left.toLowerCase().split(/\s+/));
  const rightWords = new Set(right.toLowerCase().split(/\s+/));
  let rawShared = 0;
  for (const word of leftWords) if (rightWords.has(word)) rawShared += 1;
  const words = Math.max(1, Math.max(left.split(/\s+/).length, right.split(/\s+/).length));
  return [
    cosine,
    jaccard(tokenLeft, tokenRight),
    containment(tokenLeft, tokenRight),
    sequenceRatio(left, right),
    numberLeft.size && numberRight.size ? (numberShared ? 1 : 0) : -1,
    numberLeft.size && numberRight.size && !numberShared ? 1 : 0,
    levelsLeft.length && levelsRight.length && !sameLevel ? 1 : 0,
    sameLevel ? 1 : 0,
    Math.min(left.length, right.length) / Math.max(left.length, right.length),
    rawShared / words,
    contentLeft.size && contentRight.size && difference(contentLeft, contentRight) ? 1 : 0,
    prefixOverlap(tokenLeft, tokenRight),
    tokenLeft.size === 0 ? 1 : 0,
    tokenRight.size === 0 ? 1 : 0,
  ];
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/** `parse_course_key`: only group ids carry a code; titles do not. */
function parseCourseKey(text: string): [string, string] | null {
  const body = (text.includes(':') ? text.split(':').slice(1).join(':') : text).trim().toUpperCase();
  const found = new RegExp(CODE_SOURCE).exec(body);
  if (found) return [found[1], found[2]];
  const mit = MIT_CODE.exec(body);
  return mit ? [mit[1], mit[2]] : null;
}

export class CourseMatcher {
  constructor(private readonly model: MatcherModel) {}

  probability(left: string, right: string): number {
    return this.probabilitiesForPairs([[left, right]])[0];
  }

  probabilitiesForPairs(pairs: [string, string][]): number[] {
    return featuresForPairs(pairs).map((row) => {
      let score = this.model.intercept;
      for (let index = 0; index < row.length; index += 1) {
        score += row[index] * (this.model.coefficients[index] ?? 0);
      }
      return sigmoid(score);
    });
  }

  decidePairs(pairs: [string, string][]): MatchResult[] {
    const probabilities = this.probabilitiesForPairs(pairs);
    return pairs.map(([left, right], index) => this.policy(left, right, probabilities[index]));
  }

  decide(left: string, right: string): MatchResult {
    return this.decidePairs([[left, right]])[0];
  }

  /**
   * `_veto` + threshold policy. Kept inline so the order of the checks is the
   * order the measurement used: a conflict cannot be outvoted by a string score.
   */
  private policy(left: string, right: string, probability: number): MatchResult {
    const levels = [levelSignature(left), levelSignature(right)];
    const levelConflict = levels[0].length > 0 && levels[1].length > 0 &&
      !(levels[0].length === levels[1].length &&
        levels[0].every((mark) => levels[1].includes(mark)));
    const a = parseCourseKey(left);
    const b = parseCourseKey(right);
    const codesDisagree = Boolean(a && b && a[1] !== b[1]);
    const base = { left, right, probability };
    if (levelConflict || (contentLeftConflict(left, right))) {
      return { ...base, decision: 'different', confidence: Math.min(probability, 1 - probability) };
    }
    if (probability >= this.model.threshold && codesDisagree) {
      return { ...base, decision: 'uncertain', confidence: 0.5 };
    }
    if (probability >= this.model.threshold) return { ...base, decision: 'same', confidence: probability };
    if (probability <= 1 - this.model.threshold) {
      return { ...base, decision: 'different', confidence: 1 - probability };
    }
    return { ...base, decision: 'uncertain', confidence: probability };
  }
}

function contentLeftConflict(left: string, right: string): boolean {
  const a = contentNumbers(left);
  const b = contentNumbers(right);
  return a.size > 0 && b.size > 0 && difference(a, b);
}
