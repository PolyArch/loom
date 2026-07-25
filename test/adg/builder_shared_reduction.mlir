// RUN: not loom-adg-builder-test --shared-reduction --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: not loom-adg-builder-test --shared-reduction --output %t.hardware.mlir 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: test ! -s %t.hardware.mlir

// The shared reduction catalog references resources the normative
// implementation-family registry cannot express: the integer and floating
// divide/remainder operations, dataflow.constant, the LLVM compute intrinsics
// without a registered family, and the TokenControlFu routing schemas.
// Integer absolute value is absent from that list: the catalog lowers it
// through its own compare, select, and subtract resources. Construction fails
// honestly and atomically: the diagnostic names the target and every
// unsupported resource, and no partial Fabric is emitted.

// CHECK: error: ADG target 'shared_reduction_adg' requires {{[0-9]+}} resource(s) with no registered implementation family: arith.divsi has no registered implementation family
// CHECK-DAG: arith.divui, arith.remui has no registered implementation family
// CHECK-DAG: arith.divf, arith.remf has no registered implementation family
// CHECK-DAG: dataflow.constant has no registered implementation family
// CHECK-DAG: llvm.intr.fshl has no registered implementation family
// CHECK-DAG: dataflow.sync has no registered implementation family

// ABSENT-NOT: math.absi
