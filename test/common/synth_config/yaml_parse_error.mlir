// RUN: not loom-synth-config-test %p/yaml_parse_error.yaml 2>&1 | FileCheck %s

// Malformed YAML body must surface as a parse-time error so the pass can
// report it as `config_parse_failed`. The captured diagnostic should include
// the YAMLParser line/column tag, proving SourceMgr's diagnostic handler
// routed the parser's error into the returned llvm::Error rather than to
// stderr.

// CHECK: error: yaml:
// CHECK-SAME: YAML:{{[0-9]+}}:{{[0-9]+}}
