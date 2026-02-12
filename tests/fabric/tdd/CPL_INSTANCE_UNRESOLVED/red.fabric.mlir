// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_UNRESOLVED

// fabric.instance references a symbol that does not exist.
fabric.module @top(%a: i32) -> (i32) {
  %out = fabric.instance @nonexistent(%a) : (i32) -> (i32)
  fabric.yield %out : i32
}
