// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology, each yielding a
// `dataflow.constant`. The two constants differ in value
// (0xdeadbeef vs 0xcafebabe). Per spec "hw_params policy" the
// synthesized FU's hw_params must surface the observed-value union
// of `const_hex_value` strings so the enumerator's const_hex_value
// axis fan-out covers both inputs. The control input has type
// `none` on the dataflow side and lifts to `!fabric.bits<0>` on the
// fabric side.

// CHECK: remark: {{.*}}synth-stat group=const_pair strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: fabric.module @fu_const_pair
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.constant]
// CHECK-SAME: hw_params = [{const_hex_value = ["0xcafebabe", "0xdeadbeef"]}]
// CHECK: fabric.yield

func.func @pat_const_dead(%c: none) -> i32
    attributes {loom.synth_group = "const_pair"} {
  %r = dataflow.subgraph(%cc = %c : none) -> i32 {
    %k = dataflow.constant %cc {const_value = 3735928559 : i32} : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

func.func @pat_const_cafe(%c: none) -> i32
    attributes {loom.synth_group = "const_pair"} {
  %r = dataflow.subgraph(%cc = %c : none) -> i32 {
    %k = dataflow.constant %cc {const_value = 3405691582 : i32} : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
