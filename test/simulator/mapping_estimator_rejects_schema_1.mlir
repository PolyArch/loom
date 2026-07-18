// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping"}' > %t.v1.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.v1.mapping.json --output %t.v1.estimate.json 2>&1 | FileCheck %s --check-prefix=OLD
// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping"}' > %t.v2.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.v2.mapping.json --output %t.v2.estimate.json 2>&1 | FileCheck %s --check-prefix=OLD

// OLD: has unsupported schema_version; expected string "3.0"
