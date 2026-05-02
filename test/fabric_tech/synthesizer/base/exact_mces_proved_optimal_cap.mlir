// RUN: loom-synth-base-test --exact-mces-cap-status | FileCheck %s

// A cap-bounded exact MCES query may return the best candidate it kept, but
// that is not a proof that the full search space was exhausted.

// CHECK: exact-mces:
// CHECK-SAME: hit_cap=true
// CHECK-SAME: proved_optimal=false
