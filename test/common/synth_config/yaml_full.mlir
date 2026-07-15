// RUN: loom-synth-config-test %p/yaml_full.yaml | FileCheck %s

// Every canonical key in the SynthConfig schema is set.

// CHECK: strategy=anchor
// CHECK-NEXT: parallelism.cross_group=false
// CHECK-NEXT: parallelism.workers=6
// CHECK-NEXT: coverage_verifier.parallel_match=false
// CHECK-NEXT: cost.mux_penalty=2.000000e+00
// CHECK-NEXT: cost.demux_penalty=2.500000e+00
// CHECK-NEXT: cost.carry_penalty=3.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=true
