// RUN: loom-synth-base-test --capability-tiebreak | FileCheck %s

// CHECK: equal_cost_prefers_less_extra=true
// CHECK-NEXT: lower_cost_precedes_extra_metric=true
