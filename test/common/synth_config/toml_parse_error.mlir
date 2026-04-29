// RUN: not loom-synth-config-test %p/toml_parse_error.toml 2>&1 | FileCheck %s

// Malformed TOML body must surface as a parse-time error mirroring the YAML
// parse-error coverage. The TOML parser tags its diagnostics with the
// originating line number so the failing input is easy to locate.

// CHECK: error: toml line {{[0-9]+}}:
