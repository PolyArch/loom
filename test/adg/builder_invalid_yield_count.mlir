// RUN: not loom-adg-builder-test --invalid-yield-count --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: not loom-adg-builder-test --invalid-stream-config step --output %t.step.mlir 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: count 0 < %t.step.mlir
// RUN: not loom-adg-builder-test --invalid-stream-config predicate --output %t.predicate.mlir 2>&1 | FileCheck %s --check-prefix=PREDICATE

// CHECK: ADG fu yield value count must match result type count
// STEP: ADG stream capability has invalid fixed step kind
// PREDICATE: ADG fabric.op capability is invalid
