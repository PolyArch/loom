// RUN: loom %s | FileCheck %s

// CHECK: fabric.implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>
builtin.module attributes {
  fabric.implementation_family =
      #fabric.implementation_family<ScalarIntegerAddSub>
} {
}
