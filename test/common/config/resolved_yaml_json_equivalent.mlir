// RUN: loom-config-test --resolved-json %p/resolved_equiv.yaml > %t.yaml.json
// RUN: loom-config-test --resolved-json %p/resolved_equiv.json > %t.json.json
// RUN: diff %t.yaml.json %t.json.json
// RUN: loom-config-test --resolved-identity %p/resolved_equiv.yaml > %t.yaml.identity
// RUN: loom-config-test --resolved-identity %p/resolved_equiv.json > %t.json.identity
// RUN: diff %t.yaml.identity %t.json.identity
// RUN: FileCheck %s < %t.yaml.json
// RUN: FileCheck %s --check-prefix=NO-TECHMAP < %t.yaml.json

// CHECK-DAG: "config_id": "unit.config"
// CHECK-DAG: "addr_bits": 40
// CHECK-DAG: "index_width": 64
// CHECK-DAG: "mem_bus_width": 1024
// CHECK-DAG: "objective_id": "minimize_runtime"
// CHECK-DAG: "objective_id": "minimize_area"
// NO-TECHMAP-NOT: "fabric_techmap"
