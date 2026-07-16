// RUN: not loom-synth-config-test %p/duplicate_key.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-KEY
// RUN: not loom-synth-config-test %p/duplicate_section.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-SECTION
// RUN: not loom-synth-config-test %p/duplicate_key.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-KEY
// RUN: not loom-synth-config-test %p/duplicate_section.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-SECTION

// YAML-KEY: error: yaml line 4 column 3: duplicate key 'synth.strategy'
// YAML-SECTION: error: yaml line 4 column 3: duplicate section 'synth.cost'
// TOML-KEY: error: toml line 3: duplicate key 'synth.strategy'
// TOML-SECTION: error: toml line 4: duplicate section 'synth.cost'
