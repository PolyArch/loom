// RUN: loom-synth-config-test %p/yaml_partial_uses_defaults.yaml | FileCheck %s

// Only `strategy` and one cost weight set; remaining fields keep documented
// defaults.

// CHECK: strategy=incremental
// CHECK-NEXT: parallelism.cross_group=true
// CHECK-NEXT: parallelism.workers=0
// CHECK-NEXT: coverage_verifier.parallel_match=true
// CHECK-NEXT: fallback_chain.size=0
// CHECK-NEXT: cost.mux_penalty=4.000000e+00
// CHECK-NEXT: cost.demux_penalty=1.500000e+00
// CHECK-NEXT: cost.carry_penalty=2.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=false
// CHECK-NEXT: incremental.input_order_heuristic=largest_first
// CHECK-NEXT: incremental.coverage_verify_each_attempt=true
// CHECK-NEXT: incremental_random.restarts=16
// CHECK-NEXT: incremental_random.seed=42
// CHECK-NEXT: incremental_random.input_order_heuristic=random_seeded
// CHECK-NEXT: mcs.timeout_sec=60
// CHECK-NEXT: mcs.branch_workers=8
// CHECK-NEXT: mcs.candidate_cap=1000000
// CHECK-NEXT: scc_full_unroll=false
// CHECK-NEXT: subgraph_share_recurse=false
