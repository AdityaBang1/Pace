/**
 * The JS side of the OCR boundary.
 *
 * The module is required *optionally*: `expo start` on the web, a JS-only test
 * run, or a build that somehow missed the pod must produce a readable error at
 * scan time, not a crash at import time.
 */
import { requireOptionalNativeModule } from 'expo-modules-core';

import type { PaceVisionDocument, PaceVisionPage } from './PaceVision.types';

interface PaceVisionNativeModule {
  recognizeText(uri: string): Promise<PaceVisionPage>;
  recognizePages(uris: string[]): Promise<PaceVisionPage[]>;
  recognizeDocument(uri: string): Promise<PaceVisionDocument>;
  renderPage(uri: string, page: number): Promise<PaceVisionPreview>;
}

/** One page drawn as an image, for the screen that asks "is this the document?". */
export interface PaceVisionPreview {
  uri: string;
  width: number;
  height: number;
  /** Pages the file holds, so a confirmation can say "page 1 of 12". */
  pages: number;
}

const native = requireOptionalNativeModule<PaceVisionNativeModule>('PaceVision');

export class OcrUnavailableError extends Error {
  constructor(reason: string) {
    super(reason);
    this.name = 'OcrUnavailableError';
  }
}

export function isOcrAvailable(): boolean {
  return native != null;
}

function unavailable(): never {
  throw new OcrUnavailableError(
    'Text recognition needs the Pace development build. Apple Vision is a native module, so it does not run in Expo Go or on the web.',
  );
}

/**
 * Read a file — PDF or photograph — into pages of text plus line geometry.
 *
 * A PDF is rendered and recognised page by page on the device; nothing is sent
 * anywhere, and no PDF JavaScript or network feature is involved.
 */
export async function recognizeDocument(uri: string): Promise<PaceVisionDocument> {
  if (!native) unavailable();
  return native.recognizeDocument(uri);
}

/** Recognise one page of text from an image URI. */
export async function recognizeText(uri: string): Promise<PaceVisionPage> {
  if (!native) unavailable();
  return native.recognizeText(uri);
}

/** Recognise several images in order, each tagged with its page index. */
export async function recognizePages(uris: string[]): Promise<PaceVisionPage[]> {
  if (!native) unavailable();
  return native.recognizePages(uris);
}

/**
 * Draw one page of a document as an image.
 *
 * A PDF is drawn through PDFKit; a photograph is decoded and scaled. Nothing is
 * recognised here — this exists so a page can be shown before it is read.
 */
export async function renderPage(uri: string, page = 0): Promise<PaceVisionPreview> {
  if (!native) unavailable();
  return native.renderPage(uri, page);
}
