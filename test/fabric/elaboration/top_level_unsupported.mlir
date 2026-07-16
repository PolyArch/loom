// RUN: not loom --loom-elaborate-fabric-instances %s 2>&1 | FileCheck %s

fabric.module @callee() -> () {
  fabric.yield
}

// CHECK: error: root-local Fabric instance elaboration does not support
// CHECK-SAME: fabric.instantiate @callee directly under builtin.module
// CHECK-SAME: no fabric.module occurrence owner exists
fabric.instantiate @callee() -> ()
