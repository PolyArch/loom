// RUN: not loom-adg-builder-test --invalid-yield-count --output %t.hardware.mlir 2>&1 | FileCheck %s

// CHECK: ADG fu yield value count must match result type count
