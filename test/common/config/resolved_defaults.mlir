// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=JSON
// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=NO-TECHMAP
// RUN: loom-config-test --resolved-fingerprint | FileCheck %s --check-prefix=FINGERPRINT
// RUN: loom-config-test --component-fingerprint --component-view pnr.mapping.v1 | FileCheck %s --check-prefix=COMPONENT

// JSON-DAG: "config_id": "loom.default"
// JSON-DAG: "global"
// JSON-DAG: "addr_bits": 48
// JSON-DAG: "index_width": 32
// JSON-DAG: "mem_bus_width": 32768
// JSON-DAG: "dse"
// JSON-DAG: "ranking_policy": "weighted_sum"
// JSON-DAG: "objective_id": "minimize_runtime"
// NO-TECHMAP-NOT: "fabric_techmap"
// FINGERPRINT: {{^[0-9a-f]{64}$}}
// COMPONENT: {{^[0-9a-f]{64}$}}
