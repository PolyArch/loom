// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping"}' > %t.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json 2>&1 | FileCheck %s

// CHECK: has unsupported schema_version; expected string "2.0"
