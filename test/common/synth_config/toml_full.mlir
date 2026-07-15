// RUN: loom-synth-config-test %p/toml_full.toml | FileCheck %s

// TOML mirror of yaml_full.yaml. The parsed output must match the YAML
// version field-for-field.

// CHECK: strategy=anchor
// CHECK-NEXT: parallelism.cross_group=false
// CHECK-NEXT: parallelism.workers=0
// CHECK-NEXT: coverage_verifier.parallel_match=false
// CHECK-NEXT: cost.mux_penalty=2.000000e+00
// CHECK-NEXT: cost.demux_penalty=2.500000e+00
// CHECK-NEXT: cost.carry_penalty=3.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=true
