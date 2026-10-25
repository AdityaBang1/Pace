const { withXcodeProject } = require('expo/config-plugins');

/**
 * React Native's script phases are not sandbox-compatible.
 *
 * `react-native-xcode.sh` writes `ip.txt` into the built `.app` for device
 * builds (the dev client reads it to find Metro over the LAN). With
 * ENABLE_USER_SCRIPT_SANDBOXING = YES that write is denied with EPERM and the
 * whole build fails in "Bundle React Native code and images":
 *
 *   Pace.app/ip.txt: Operation not permitted
 *
 * Simulator builds skip that branch, so only `expo run:ios --device` hits it.
 * React Native expects the setting to be NO; the generated project template
 * sets it to YES. Keep it in a plugin so `expo prebuild` cannot undo it.
 */
const SETTING = 'ENABLE_USER_SCRIPT_SANDBOXING';

module.exports = function withScriptSandboxingOff(config) {
  return withXcodeProject(config, (cfg) => {
    const buildConfigurations = cfg.modResults.pbxXCBuildConfigurationSection();

    // Set explicitly rather than only flipping YES to NO: the generated
    // template adds and drops the key between SDK versions, and "absent"
    // relies on an Xcode default this project should not have to track.
    for (const key of Object.keys(buildConfigurations)) {
      const buildSettings = buildConfigurations[key]?.buildSettings;
      if (buildSettings) {
        buildSettings[SETTING] = 'NO';
      }
    }

    return cfg;
  });
};
