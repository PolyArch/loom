// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=JSON
// RUN: loom-config-test --resolved-json | FileCheck %s --check-prefix=NO-TECHMAP
// RUN: loom-config-test --resolved-identity | FileCheck %s --check-prefix=IDENTITY
// RUN: loom-config-test --resolved-json --loom-accel-profile=quick_explore | FileCheck %s --check-prefix=QUICK
// RUN: loom-config-test --resolved-json --loom-accel-profile=implementation | FileCheck %s --check-prefix=IMPLEMENTATION
// RUN: loom-config-test --resolved-json > %t.canonical.json
// RUN: loom-config-test --resolved-json --loom-accel-profile=%t.canonical.json > %t.reparsed.json
// RUN: diff %t.canonical.json %t.reparsed.json
// RUN: loom-config-test --resolved-identity > %t.identity
// RUN: loom-config-test --resolved-identity --loom-accel-profile=%t.canonical.json > %t.reparsed.identity
// RUN: diff %t.identity %t.reparsed.identity

// JSON-DAG: "hardware_target"
// JSON-DAG: "template_identity": "loom.adg.builtin.default"
// JSON-DAG: "schema_major": 1
// JSON-DAG: "schema_minor": 1
// JSON-DAG: "acc_core_count": 8
// JSON-DAG: "spatial_pe_count": 27
// JSON-DAG: "temporal_pe_count": 9
// JSON-DAG: "memory_capacity_bytes": 262144
// JSON-DAG: "dse"
// JSON-DAG: "structured_ownership"
// JSON-DAG: "scope_expansion_limit": 64
// JSON-DAG: "schedule"
// JSON-DAG: "scope_expansion_limit": 64
// JSON-DAG: "memory_communication"
// JSON-DAG: "scope_expansion_limit": 64
// JSON-DAG: "dataflow_rewrite"
// JSON-DAG: "scope_expansion_limit": 64
// JSON-DAG: "tech_mapping"
// JSON-DAG: "match_row_attempt_limit": 65536
// JSON-DAG: "partial_cover_expansion_limit": 262144
// JSON-DAG: "candidate_publication_limit": 16
// JSON-DAG: "spatial_pnr"
// JSON-DAG: "system_pnr"
// JSON-DAG: "seed_attempt_count": 4
// JSON-DAG: "assignment_attempt_limit_per_seed": 65536
// JSON-DAG: "endpoint_expansion_limit": 262144
// JSON-DAG: "negotiation_iteration_limit": 64
// JSON-DAG: "price_kernel": "multiplicative"
// JSON-DAG: "calibration_proposal_count": 256
// JSON-DAG: "minimum_temperature": 1
// JSON-DAG: "max_region_decisions": 256
// JSON-DAG: "max_solver_calls": 1024
// JSON-DAG: "selected_total_ordering": 0
// JSON-DAG: "selected_search_energy": 2
// JSON-NOT: "config_id"
// JSON-NOT: "global"
// JSON-NOT: "ranking_policy"
// JSON-NOT: "objectives"
// NO-TECHMAP-NOT: "fabric_techmap"
// IDENTITY: {{^[0-9a-f]{64}$}}
// QUICK-DAG: "seed_attempt_count": 2
// QUICK-DAG: "assignment_attempt_limit_per_seed": 16384
// QUICK-DAG: "endpoint_expansion_limit": 65536
// QUICK-DAG: "max_region_decisions": 64
// QUICK-DAG: "max_solver_calls": 128
// IMPLEMENTATION-DAG: "seed_attempt_count": 16
// IMPLEMENTATION-DAG: "assignment_attempt_limit_per_seed": 524288
// IMPLEMENTATION-DAG: "endpoint_expansion_limit": 2097152
// IMPLEMENTATION-DAG: "max_region_decisions": 1024
// IMPLEMENTATION-DAG: "max_solver_calls": 8192
