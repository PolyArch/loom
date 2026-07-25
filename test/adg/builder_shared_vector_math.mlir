// RUN: not loom-adg-builder-test --shared-vector-math --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: test ! -s %t.hardware.mlir

// The shared vector math catalog references resources the normative
// implementation-family registry cannot express: dataflow.constant, the
// integer and floating divide/remainder operations, the elementary math
// functions, the LLVM compute intrinsics without a registered family, and
// the TokenControlFu routing schemas. Construction fails honestly and
// atomically: the diagnostic names the target and every unsupported
// resource, and no partial Fabric is emitted.

// CHECK: error: ADG target 'shared_vector_math_adg' requires {{[0-9]+}} resource(s) with no registered implementation family: dataflow.constant has no registered implementation family
// CHECK-DAG: arith.remsi has no registered implementation family
// CHECK-DAG: math.sqrt has no registered implementation family
// CHECK-DAG: llvm.intr.usub.sat has no registered implementation family
// CHECK-DAG: dataflow.mux has no registered implementation family
// CHECK-DAG: dataflow.sync has no registered implementation family
