// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// An empty module (no func.func at all) is a valid input. The pass emits
// a single `no synth groups` remark and leaves the module untouched.

// CHECK: remark:
// CHECK-SAME: no synth groups
// CHECK: module
module {
}
