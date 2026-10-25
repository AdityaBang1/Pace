Pod::Spec.new do |s|
  s.name           = 'PaceVision'
  s.version        = '0.1.0'
  s.summary        = 'Apple Vision OCR with line geometry for Pace'
  s.description    = 'Returns recognised text plus normalised bounding boxes so the app can point at the source of a value.'
  s.author         = ''
  s.homepage       = 'https://pace.app'
  s.platforms      = { :ios => '15.1' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }

  s.source_files = "**/*.{h,m,mm,swift}"
end
