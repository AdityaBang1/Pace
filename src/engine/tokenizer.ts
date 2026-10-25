/**
 * The section reader's tokeniser, ported from HuggingFace's BertTokenizer.
 *
 * A tokeniser that is *almost* right is worse than one that is obviously wrong:
 * the model still returns a confident label, just for different text. So this is
 * a faithful port of `basic_tokenizer` + `wordpiece_tokenizer`, and
 * `tests/mobile/tokenizer_parity.json` holds ids produced by the Python
 * tokeniser for the same strings.
 *
 * The vocabulary is a text asset (`assets/models/tokenizer/vocab.model`), one
 * token per line, index = line number.
 */

export const PAD_ID = 0;
export const CLS_ID = 101;
export const SEP_ID = 102;
export const UNK_ID = 100;

const CONTROL = /[\u0000\u0005\uFFFD]/g;

function isWhitespace(char: string): boolean {
  if (char === ' ' || char === '\t' || char === '\n' || char === '\r') return true;
  return /\s/u.test(char);
}

function isPunctuation(char: string): boolean {
  const code = char.codePointAt(0) ?? 0;
  if (
    (code >= 33 && code <= 47) ||
    (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) ||
    (code >= 123 && code <= 126)
  ) {
    return true;
  }
  return /^\p{P}$/u.test(char);
}

function isChinese(char: string): boolean {
  const code = char.codePointAt(0) ?? 0;
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x20000 && code <= 0x2a6df) ||
    (code >= 0x2a700 && code <= 0x2b73f) ||
    (code >= 0x2b740 && code <= 0x2b81f) ||
    (code >= 0x2b920 && code <= 0x2ceaf) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0x2f800 && code <= 0x2fa1f)
  );
}

/** BertNormalizer: strip control chars, normalise whitespace, lower-case. */
export function normalizeBert(text: string, doLowerCase = true): string {
  let out = text.replace(CONTROL, '');
  out = out.replace(/\s/gu, ' ');
  if (doLowerCase) out = out.toLowerCase();
  return out;
}

function splitOnPunctuation(token: string): string[] {
  const pieces: string[] = [];
  let current = '';
  for (const char of Array.from(token)) {
    if (isPunctuation(char)) {
      if (current) pieces.push(current);
      pieces.push(char);
      current = '';
    } else {
      current += char;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}

export interface TokenizerConfig {
  doLowerCase: boolean;
  tokenizeChineseChars: boolean;
  stripAccents: boolean;
}

export class WordPieceTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly maxTokenChars: number;

  constructor(vocabText: string, private readonly config: TokenizerConfig = {
    doLowerCase: true,
    tokenizeChineseChars: true,
    stripAccents: false,
  }, maxTokenChars = 100) {
    this.vocab = new Map();
    const lines = vocabText.split(/\r?\n/);
    for (let index = 0; index < lines.length; index += 1) {
      const token = lines[index];
      if (token.length === 0) continue;
      this.vocab.set(token, index);
    }
    this.maxTokenChars = maxTokenChars;
  }

  get size(): number {
    return this.vocab.size;
  }

  id(token: string): number {
    return this.vocab.get(token) ?? UNK_ID;
  }

  /** BERT basic tokenizer: clean, split CJK, split punctuation, lower-case. */
  private basicTokenize(text: string): string[] {
    let cleaned = normalizeBert(text, this.config.doLowerCase);
    if (this.config.stripAccents) cleaned = cleaned.normalize('NFD').replace(/\p{Mn}+/gu, '');
    if (this.config.tokenizeChineseChars) {
      let spaced = '';
      for (const char of Array.from(cleaned)) {
        spaced += isChinese(char) ? ` ${char} ` : char;
      }
      cleaned = spaced;
    }
    const tokens: string[] = [];
    for (const word of cleaned.trim().split(/\s+/)) {
      if (!word) continue;
      for (const piece of splitOnPunctuation(word)) tokens.push(piece);
    }
    return tokens;
  }

  private wordPiece(token: string): string[] {
    if (token.length > this.maxTokenChars) return ['[UNK]'];
    const characters = Array.from(token);
    const pieces: string[] = [];
    let start = 0;
    while (start < characters.length) {
      let end = characters.length;
      let found: string | null = null;
      while (start < end) {
        const candidate = (start > 0 ? '##' : '') + characters.slice(start, end).join('');
        if (this.vocab.has(candidate)) {
          found = candidate;
          break;
        }
        end -= 1;
      }
      if (found === null) return ['[UNK]'];
      pieces.push(found);
      start = end;
    }
    return pieces;
  }

  tokenize(text: string): string[] {
    const out: string[] = [];
    for (const token of this.basicTokenize(text)) {
      out.push(...this.wordPiece(token));
    }
    return out;
  }

  /** Token ids without padding: `[CLS] … [SEP]`, truncated like the reader. */
  encodeIds(text: string, maxLength: number): number[] {
    const pieces = this.tokenize(text);
    const ids = [CLS_ID];
    for (const piece of pieces) {
      if (ids.length >= maxLength - 1) break;
      ids.push(this.id(piece));
    }
    ids.push(SEP_ID);
    return ids;
  }

  /**
   * Encode one sequence the way `tokenizer(..., padding='max_length',
   * truncation=True)` does: [CLS] tokens [SEP] padded to ``length`` with PAD.
   */
  encode(text: string, length: number): { ids: bigint[]; mask: bigint[] } {
    const pieces = this.tokenize(text);
    const ids: bigint[] = [BigInt(CLS_ID)];
    for (const piece of pieces) {
      if (ids.length >= length - 1) break;
      ids.push(BigInt(this.id(piece)));
    }
    ids.push(BigInt(SEP_ID));
    const mask: bigint[] = new Array(ids.length).fill(BigInt(1));
    while (ids.length < length) {
      ids.push(BigInt(PAD_ID));
      mask.push(BigInt(0));
    }
    return { ids, mask };
  }
}
