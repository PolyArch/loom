// RUN: not loom-adg-builder-test --shared-vector-mesh --output %t.hardware.mlir 2>&1 | FileCheck %s
// RUN: test ! -s %t.hardware.mlir

// The shared vector mesh catalog references two resources the normative
// implementation-family registry cannot express: llvm.intr.bswap and
// dataflow.sync. Construction fails honestly and atomically: the diagnostic
// names the target and every unsupported resource, and no partial Fabric is
// emitted.

// CHECK: error: ADG target 'shared_vector_mesh_adg' requires {{[0-9]+}} resource(s) with no registered implementation family: llvm.intr.bswap has no registered implementation family
// CHECK-DAG: dataflow.sync has no registered implementation family
