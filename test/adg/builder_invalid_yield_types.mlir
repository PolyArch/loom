// RUN: not loom-adg-builder-test --invalid-yield-types --output %t.hardware.mlir 2>&1 | FileCheck %s

// CHECK: ADG fu yield type count must match yield value count
