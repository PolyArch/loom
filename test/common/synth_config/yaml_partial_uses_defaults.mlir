// RUN: loom-synth-config-test %p/yaml_partial_uses_defaults.yaml | FileCheck %s

// Only one cost weight is set; remaining fields keep canonical defaults.

// CHECK: strategy=anchor
// CHECK-NEXT: parallelism.cross_group=true
// CHECK-NEXT: parallelism.workers=0
// CHECK-NEXT: coverage_verifier.parallel_match=true
// CHECK-NEXT: cost.mux_penalty=4.000000e+00
// CHECK-NEXT: cost.demux_penalty=1.500000e+00
// CHECK-NEXT: cost.carry_penalty=2.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=false
