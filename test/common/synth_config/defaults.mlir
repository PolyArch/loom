// RUN: loom-synth-config-test | FileCheck %s

// With no path supplied, the helper dumps the built-in defaults verbatim.
// The public spec owns the semantic config axes; this regression owns the
// current built-in default values.

// CHECK: strategy=anchor
// CHECK-NEXT: parallelism.cross_group=true
// CHECK-NEXT: parallelism.workers=0
// CHECK-NEXT: coverage_verifier.parallel_match=true
// CHECK-NEXT: cost.mux_penalty=1.500000e+00
// CHECK-NEXT: cost.demux_penalty=1.500000e+00
// CHECK-NEXT: cost.carry_penalty=2.000000e+00
// CHECK-NEXT: anchor.allow_intra_position_mux=false
