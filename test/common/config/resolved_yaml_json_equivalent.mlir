// RUN: loom-config-test --resolved-json --loom-accel-profile=%p/resolved_equiv.yaml > %t.yaml.json
// RUN: loom-config-test --resolved-json --loom-accel-profile=%p/resolved_equiv.json > %t.json.json
// RUN: diff %t.yaml.json %t.json.json
// RUN: loom-config-test --resolved-identity --loom-accel-profile=%p/resolved_equiv.yaml > %t.yaml.identity
// RUN: loom-config-test --resolved-identity --loom-accel-profile=%p/resolved_equiv.json > %t.json.identity
// RUN: diff %t.yaml.identity %t.json.identity
// RUN: FileCheck %s < %t.yaml.json
// RUN: FileCheck %s --check-prefix=NO-TECHMAP < %t.yaml.json

// CHECK-DAG: "scope_expansion_limit": 17
// CHECK-DAG: "scope_expansion_limit": 19
// CHECK-DAG: "scope_expansion_limit": 29
// CHECK-DAG: "scope_expansion_limit": 23
// CHECK-DAG: "match_row_attempt_limit": 2048
// CHECK-DAG: "partial_cover_expansion_limit": 4096
// CHECK-DAG: "candidate_evaluation_limit": 32
// CHECK-DAG: "candidate_publication_limit": 8
// CHECK-DAG: "spatial_pnr"
// CHECK-DAG: "system_pnr"
// NO-TECHMAP-NOT: "fabric_techmap"
