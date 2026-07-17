// RUN: loom-dfg-sim %s --graph calls_external --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph calls_indirect --memref 0=0 --output %t.indirect.json
// RUN: FileCheck %s --check-prefix=INDIRECT < %t.indirect.json

// CHECK-DAG: "status": "unsupported"
// CHECK-DAG: "unsupported op: llvm.call @opaque_callee"

// INDIRECT-DAG: "status": "unsupported"
// INDIRECT-DAG: "unsupported op: llvm.call"
// INDIRECT-NOT: "unsupported op: llvm.call @

module {
  llvm.func @opaque_callee(i32) -> i32

  dataflow.graph.func private @calls_external(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = llvm.call @opaque_callee(%value) : (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @calls_indirect(%ctrl: none, %callee: !llvm.ptr)
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = llvm.call %callee(%value) : !llvm.ptr, (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
