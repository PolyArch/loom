// RUN: loom-synth-base-test --list-strategies | FileCheck %s

// Only strategies that satisfy the canonical capability and coverage gate
// are externally selectable.

// CHECK: anchor
// CHECK-NOT: mcs
// CHECK-NOT: incremental
