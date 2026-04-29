// RUN: loom-synth-config-test %p/toml_full.toml | FileCheck %s

// TOML mirror of yaml_full.yaml. The parsed output must match the YAML
// version field-for-field.

// CHECK: strategy=anchor
// CHECK-NEXT: parallelism.cross_group=false
// CHECK-NEXT: parallelism.workers=0
// CHECK-NEXT: coverage_verifier.parallel_match=false
// CHECK-NEXT: fallback_chain.size=2
// CHECK-NEXT: fallback_chain[0]=mcs
// CHECK-NEXT: fallback_chain[1]=incremental
// CHECK-NEXT: cost.mux_penalty=2.000000e+00
// CHECK-NEXT: cost.demux_penalty=2.500000e+00
// CHECK-NEXT: cost.carry_penalty=3.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=true
// CHECK-NEXT: incremental.input_order_heuristic=smallest_first
// CHECK-NEXT: incremental.coverage_verify_each_attempt=false
// CHECK-NEXT: incremental_random.restarts=32
// CHECK-NEXT: incremental_random.seed=123
// CHECK-NEXT: incremental_random.input_order_heuristic=largest_first
// CHECK-NEXT: mcs.timeout_sec=120
// CHECK-NEXT: mcs.branch_workers=4
// CHECK-NEXT: mcs.candidate_cap=500000
// CHECK-NEXT: scc_full_unroll=true
// CHECK-NEXT: subgraph_share_recurse=true
