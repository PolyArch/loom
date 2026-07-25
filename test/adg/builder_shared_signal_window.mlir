// RUN: not loom-adg-builder-test --shared-signal-window --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: test ! -s %t.hardware.mlir

// The shared signal-window catalog references resources the normative
// implementation-family registry cannot express: dataflow.constant, the
// integer and floating divide/remainder operations, the elementary math
// functions, the LLVM compute intrinsics without a registered family, and
// the TokenControlFu routing schemas. Construction fails honestly and
// atomically: the diagnostic names the target and every unsupported
// resource, and no partial Fabric is emitted.

// CHECK: error: ADG target 'shared_signal_window_adg' requires {{[0-9]+}} resource(s) with no registered implementation family: dataflow.constant has no registered implementation family
// CHECK-DAG: arith.divf has no registered implementation family
// CHECK-DAG: math.cos has no registered implementation family
// CHECK-DAG: llvm.intr.ctlz has no registered implementation family
// CHECK-DAG: dataflow.demux has no registered implementation family
// CHECK-DAG: dataflow.sync has no registered implementation family
