// RUN: not loom-synth-config-test %p/missing_synth_top_level.yaml 2>&1 | FileCheck %s

// A YAML document without a top-level `synth:` mapping is rejected.

// CHECK: error: yaml line 1 column 1: unknown section 'techmap'
