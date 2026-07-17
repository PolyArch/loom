// RUN: not loom-adg-builder-test --invalid-yield-count --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: not loom-adg-builder-test --invalid-stream-config missing --output %t.missing.mlir 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not loom-adg-builder-test --invalid-stream-config generic --output %t.generic.mlir 2>&1 | FileCheck %s --check-prefix=GENERIC
// RUN: not loom-adg-builder-test --invalid-stream-config step --output %t.step.mlir 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: count 0 < %t.step.mlir
// RUN: not loom-adg-builder-test --invalid-stream-config predicate --output %t.predicate.mlir 2>&1 | FileCheck %s --check-prefix=PREDICATE

// CHECK: ADG fu yield value count must match result type count
// MISSING: ADG dataflow.stream requires typed stream configuration
// GENERIC: ADG dataflow.stream configuration cannot use generic hw_params or sw_configs
// STEP: ADG stream configuration has invalid step kind
// PREDICATE: ADG stream configuration has invalid predicate
