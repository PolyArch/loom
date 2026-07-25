// RUN: not loom-adg-builder-test --heterogeneous-soc --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: test ! -s %t.hardware.mlir

// The heterogeneous SoC composition inlines the reusable shared-reduction
// accelerator template. That template references resources the normative
// implementation-family registry cannot express, so the composition
// propagates the same honest, atomic construction failure rather than
// emitting a fabric.system beside partial Fabric.

// CHECK: error: ADG target 'shared_reduction_adg' requires {{[0-9]+}} resource(s) with no registered implementation family: arith.divsi has no registered implementation family
// CHECK-DAG: dataflow.sync has no registered implementation family
