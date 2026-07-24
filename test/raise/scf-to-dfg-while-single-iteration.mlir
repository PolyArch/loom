// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph while_single_iteration \
// RUN:   --memref 0=9 --max-event-steps=128 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK: "arg0": [
// CHECK-NEXT: "i32:1"
// CHECK: "dataflow.store": 2
// CHECK: "status": "pass"

dataflow.graph private @while_single_iteration(
    %start: none, %buffer: memref<1xi32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %index = arith.constant 0 : index
  scf.while (%value = %zero) : (i32) -> i32 {
    memref.store %value, %buffer[%index] : memref<1xi32>
    %continue = arith.cmpi slt, %value, %one : i32
    scf.condition(%continue) %value : i32
  } do {
  ^bb0(%value: i32):
    %next = arith.addi %value, %one : i32
    scf.yield %next : i32
  }
  dataflow.graph.return %start : none
}
