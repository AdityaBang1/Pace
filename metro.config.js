// The model artifacts are not images or fonts, so Metro has to be told they are
// assets; otherwise `require('../../assets/models/…')` fails at bundle time.
// Every extension here is one that `assets/models` actually contains.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

for (const extension of ['onnx', 'bin', 'model']) {
  if (!config.resolver.assetExts.includes(extension)) {
    config.resolver.assetExts.push(extension);
  }
}

module.exports = config;
