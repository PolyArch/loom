// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph independent_loop_memory \
// RUN:   --arg 0=4 --arg 1=9 --memref 2=0,0,0,0 --output %t.active.json
// RUN: FileCheck %s --check-prefix=ACTIVE < %t.active.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph independent_loop_memory \
// RUN:   --arg 0=0 --arg 1=9 --memref 2=0,0,0,0 --output %t.empty.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t.empty.json

// Independent iterations still wait for pre-loop memory effects, and the
// following read waits for every loop store. Empty loops retain their incoming
// frontier without requiring any body completion.
// ACTIVE: "final_memory_state": {
// ACTIVE: "arg2": [
// ACTIVE-NEXT: "i32:9",
// ACTIVE-NEXT: "i32:9",
// ACTIVE-NEXT: "i32:9",
// ACTIVE-NEXT: "i32:9"
// ACTIVE: "final_outputs": [
// ACTIVE-NEXT: "none",
// ACTIVE-NEXT: "i32:9"
// ACTIVE: "status": "pass"
// EMPTY: "final_memory_state": {
// EMPTY: "arg2": [
// EMPTY-NEXT: "i32:17",
// EMPTY-NEXT: "i32:0",
// EMPTY-NEXT: "i32:0",
// EMPTY-NEXT: "i32:0"
// EMPTY: "final_outputs": [
// EMPTY-NEXT: "none",
// EMPTY-NEXT: "i32:17"
// EMPTY: "status": "pass"

module {
  dataflow.graph private @independent_loop_memory(
      %start: none, %upper: index, %value: i32,
      %memory: memref<4xi32>) -> (i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %initial = arith.constant 17 : i32
    memref.store %initial, %memory[%zero] : memref<4xi32>
    scf.for %i = %zero to %upper step %one {
      memref.store %value, %memory[%i] : memref<4xi32>
    }
    %result = memref.load %memory[%zero] : memref<4xi32>
    dataflow.graph.return %start, %result : none, i32
  }
}
