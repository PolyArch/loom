// RUN: not loom --loom-elaborate-fabric-instances %s 2>&1 | FileCheck %s

fabric.module @callee() -> () {
  fabric.yield
}

// CHECK: error: 'fabric.instantiate' op directly under builtin.module is not allowed
fabric.instantiate @callee() -> ()
