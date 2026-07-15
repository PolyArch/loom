// RUN: not loom-synth-config-test %p/unknown_key.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-KEY
// RUN: not loom-synth-config-test %p/removed_section.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-SECTION
// RUN: not loom-synth-config-test %p/unknown_key.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-KEY
// RUN: not loom-synth-config-test %p/removed_section.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-SECTION

// YAML-KEY: error: yaml line 3 column 3: unknown key 'synth.fallback_chain'
// YAML-SECTION: error: yaml line 3 column 3: unknown section 'synth.mcs'
// TOML-KEY: error: toml line 3: unknown key 'synth.fallback_chain'
// TOML-SECTION: error: toml line 3: unknown section 'synth.mcs'
