import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * onnxruntime-react-native 1.24.3 predates React Native 0.88 / Expo SDK 58.
 *
 * Three upstream issues matter to Pace:
 *
 * 1. iOS: OnnxruntimeModule.mm names RCTCxxBridge, whose declaration was removed
 *    in RN 0.88. The bridge proxy still exposes `runtime`; Microsoft's issue
 *    documents a small compatibility category + respondsToSelector guard that
 *    builds and successfully loads an InferenceSession on Expo 58.
 *
 * 2. Expo autolinking: the package still ships `unimodule.json`. Modern Expo
 *    treats that as an Expo-module marker and can exclude ORT from normal React
 *    Native autolinking even though ORT is a regular RN native module.
 *
 * 3. The podspec leaves `onnxruntime-c` unversioned, so a clean pod install can
 *    pair this 1.24.3 JS/JSI binding with a different native runtime. Pin the pod
 *    to the package version so the submission build is reproducible.
 *
 * npm has not published the upstream fixes in a release newer than 1.24.3 yet.
 * Keep this patch deliberately narrow and fail loudly if the package changes in
 * a way that leaves the known-broken source behind. Once a published ORT release
 * fixes these issues, remove this script and the root postinstall hook.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const ort = path.join(root, 'node_modules', 'onnxruntime-react-native');

if (!fs.existsSync(ort)) {
  console.warn('[pace] onnxruntime-react-native is not installed; compatibility patch skipped.');
  process.exit(0);
}

const packageJsonPath = path.join(ort, 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
console.log(`[pace] checking onnxruntime-react-native ${packageJson.version} compatibility`);

function writeIfChanged(file, before, after) {
  if (before === after) return false;
  fs.writeFileSync(file, after);
  return true;
}

// ---------------------------------------------------------------------------
// iOS / React Native 0.88
// ---------------------------------------------------------------------------
const iosPath = path.join(ort, 'ios', 'OnnxruntimeModule.mm');
if (!fs.existsSync(iosPath)) {
  throw new Error(`[pace] expected ORT iOS source is missing: ${iosPath}`);
}

let ios = fs.readFileSync(iosPath, 'utf8');
const iosOriginal = ios;

if (!ios.includes('@interface RCTBridge (OrtJsiRuntime)')) {
  const anchor = '#import <jsi/jsi.h>';
  if (!ios.includes(anchor)) {
    throw new Error('[pace] ORT iOS source changed: could not find the JSI import anchor.');
  }
  ios = ios.replace(
    anchor,
    `${anchor}\n\n// RN 0.88 no longer declares RCTCxxBridge in RCTBridge+Private.h, but the\n// bridge/proxy still exposes the runtime selector. Keep this category local to\n// ORT until upstream installs its JSI bindings through the RN 0.88 API.\n@interface RCTBridge (OrtJsiRuntime)\n- (void *)runtime;\n@end`,
  );
}

// Match the released block by shape rather than by exact whitespace: upstream
// indents it and leaves a blank line before the `auto jsiRuntime` line, and
// either can change in a patch release without changing what the code does.
const oldInstall =
  /RCTCxxBridge\*\s*cxxBridge\s*=\s*\(\s*RCTCxxBridge\s*\*\s*\)\s*_bridge;\s*if\s*\(\s*cxxBridge\s*==\s*nil\s*\)\s*\{\s*return\s+@false;\s*\}\s*auto\s+jsiRuntime\s*=\s*\(\s*facebook::jsi::Runtime\s*\*\s*\)\s*cxxBridge\.runtime;/;
// Do not gate this on respondsToSelector:. Under the New Architecture the
// bridge handed to a legacy module is an RCTBridgeProxy, which is an NSProxy
// subclass: it implements `runtime`, but it answers
// respondsToSelector:@selector(runtime) with NO, so a selector guard silently
// turns install into a no-op and every ORT call dies with "OrtApi is not
// initialized". Probe the call itself, keep the null check, and let the
// surrounding @try/@catch handle an unexpected bridge.
const newInstall = `if (_bridge == nil) {\n      return @false;\n    }\n    void* rawRuntime = NULL;\n    @try {\n      rawRuntime = [_bridge runtime];\n    } @catch (NSException* exception) {\n      rawRuntime = NULL;\n    }\n    auto jsiRuntime = (facebook::jsi::Runtime*)rawRuntime;`;

if (oldInstall.test(ios)) {
  ios = ios.replace(oldInstall, newInstall);
}

if (oldInstall.test(ios)) {
  throw new Error(
    '[pace] ORT still contains the RN 0.88-incompatible RCTCxxBridge install path after patching.',
  );
}

const iosChanged = writeIfChanged(iosPath, iosOriginal, ios);

// Pin the CocoaPod to the same version as the JS/JSI package. ORT 1.24.3's
// published podspec otherwise resolves whichever onnxruntime-c happens to be
// newest when `pod install` runs.
const podspecPath = path.join(ort, 'onnxruntime-react-native.podspec');
if (!fs.existsSync(podspecPath)) {
  throw new Error(`[pace] expected ORT podspec is missing: ${podspecPath}`);
}
const podspecOriginal = fs.readFileSync(podspecPath, 'utf8');
let podspec = podspecOriginal;
if (podspec.includes('spec.dependency "onnxruntime-c"\n')) {
  podspec = podspec.replace(
    'spec.dependency "onnxruntime-c"\n',
    'spec.dependency "onnxruntime-c", package["version"]\n',
  );
}
if (/spec\.dependency\s+["']onnxruntime-c["']\s*$/.test(podspec)) {
  throw new Error('[pace] ORT podspec still leaves onnxruntime-c unpinned after patching.');
}
const podspecChanged = writeIfChanged(podspecPath, podspecOriginal, podspec);

// ---------------------------------------------------------------------------
// Expo autolinking
// ---------------------------------------------------------------------------
const staleUnimodule = path.join(ort, 'unimodule.json');
let removedUnimodule = false;
if (fs.existsSync(staleUnimodule)) {
  fs.unlinkSync(staleUnimodule);
  removedUnimodule = true;
}

// ---------------------------------------------------------------------------
// Android / Gradle 9 (not needed for the iOS submission, but Expo 58 evaluates
// this when Android is built and ORT 1.24.3 still uses the removed VersionNumber
// helper). Applying the already-upstreamed equivalent keeps the repo portable.
// ---------------------------------------------------------------------------
const androidPath = path.join(ort, 'android', 'build.gradle');
let androidChanged = false;
if (fs.existsSync(androidPath)) {
  const androidOriginal = fs.readFileSync(androidPath, 'utf8');
  let android = androidOriginal;
  const oldCheck = 'if (VersionNumber.parse(REACT_NATIVE_VERSION) < VersionNumber.parse("0.71")) {';
  const newCheck = `def rnVersionParts = REACT_NATIVE_VERSION.split("\\\\.")\n  if (rnVersionParts[0].toInteger() == 0 && rnVersionParts[1].toInteger() < 71) {`;
  if (android.includes(oldCheck)) android = android.replace(oldCheck, newCheck);
  if (android.includes('VersionNumber.parse(REACT_NATIVE_VERSION)')) {
    throw new Error('[pace] ORT Android source still contains the Gradle-9-incompatible VersionNumber check.');
  }
  androidChanged = writeIfChanged(androidPath, androidOriginal, android);
}

console.log(
  `[pace] ORT compatibility ready (ios=${iosChanged ? 'patched' : 'already-ready'}, ` +
    `pod=${podspecChanged ? 'pinned' : 'already-ready'}, ` +
    `autolink=${removedUnimodule ? 'fixed' : 'already-ready'}, ` +
    `android=${androidChanged ? 'patched' : 'already-ready'})`,
);
