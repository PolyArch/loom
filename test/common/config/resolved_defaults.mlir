// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=JSON
// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=NO-TECHMAP
// RUN: loom-config-test --resolved-identity | FileCheck %s --check-prefix=IDENTITY

// JSON-DAG: "config_id": "loom.default"
// JSON-DAG: "global"
// JSON-DAG: "addr_bits": 48
// JSON-DAG: "index_width": 32
// JSON-DAG: "mem_bus_width": 32768
// JSON-DAG: "dse"
// JSON-DAG: "ranking_policy": "weighted_sum"
// JSON-DAG: "structured_ownership"
// JSON-DAG: "scope_expansion_limit": 64
// JSON-DAG: "tech_mapping"
// JSON-DAG: "match_row_attempt_limit": 65536
// JSON-DAG: "partial_cover_expansion_limit": 262144
// JSON-DAG: "candidate_publication_limit": 16
// JSON-DAG: "objective_id": "minimize_runtime"
// NO-TECHMAP-NOT: "fabric_techmap"
// IDENTITY: {{^[0-9a-f]{64}$}}
