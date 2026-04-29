// RUN: not loom-synth-config-test %p/yaml_parse_error.yaml 2>&1 | FileCheck %s

// Malformed YAML body must surface as a parse-time error so the pass can
// report it as `config_parse_failed`.

// CHECK: error:
