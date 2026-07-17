// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph pointer_offset_load_store --memref 0:4=1.000000e+00,2.000000e+00,3.000000e+00 --output %t.json
// RUN: FileCheck %s --check-prefix=OFFSET < %t.json
// RUN: not loom-dfg-sim %s --graph memref_rejects_offset --memref 0:4=1.000000e+00 --output %t.bad.json 2>&1 | FileCheck %s --check-prefix=MEMREF-ERR

// OFFSET-DAG: "status": "pass"
// OFFSET-DAG: "final_memory_state": {
// OFFSET-DAG: "arg0": [
// OFFSET-DAG: "f32:1",
// OFFSET-DAG: "f32:4",
// OFFSET-DAG: "f32:3"
// MEMREF-ERR: memref argument 0 cannot use a nonzero memory fixture byte offset

module {
  dataflow.graph.func private @pointer_offset_load_store(
      %ctrl: none, %ptr: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xf32>
    %value, %done = dataflow.load %mem[%zero] %ctrl : memref<?xf32>
    %next = arith.addf %value, %value : f32
    llvm.store %next, %ptr {alignment = 4 : i64} : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @memref_rejects_offset(
      %ctrl: none, %mem: memref<?xf32>) -> none
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %ctrl : none
  }
}
