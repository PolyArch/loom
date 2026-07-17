// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// The constants differ in value, so the shared physical op owns two complete
// typed modes and the FU exposes two explicit encodings. The control input has
// type `none` on the dataflow side and lifts to `!fabric.bits<0>`.

// CHECK: remark: {{.*}}synth-stat group=const_pair strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_const_pair
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.constant]
// CHECK-SAME: hw_params = [
// CHECK: fabric.yield

func.func @pat_const_dead(%c: none) -> i32
    attributes {loom.synth_group = "const_pair"} {
  %k = dataflow.constant %c {const_value = 3735928559 : i32} : i32
  return %k : i32
}

func.func @pat_const_cafe(%c: none) -> i32
    attributes {loom.synth_group = "const_pair"} {
  %k = dataflow.constant %c {const_value = 3405691582 : i32} : i32
  return %k : i32
}
