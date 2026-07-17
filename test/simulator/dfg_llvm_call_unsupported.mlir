// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/direct.mlir --graph calls_external --output %t.json 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: not loom-dfg-sim %t.dir/indirect.mlir --graph calls_indirect --memref 0=0 --output %t.indirect.json 2>&1 | FileCheck %s --check-prefix=INDIRECT

// DIRECT: finalized graph contains unregistered actor 'llvm.call'

// INDIRECT: finalized graph contains residual pointer operation 'llvm.call'

//--- direct.mlir
module {
  llvm.func @opaque_callee(i32) -> i32

  dataflow.graph private @calls_external(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = llvm.call @opaque_callee(%value) : (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- indirect.mlir
module {
  dataflow.graph private @calls_indirect(%ctrl: none, %callee: !llvm.ptr)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %result = llvm.call %callee(%value) : !llvm.ptr, (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %result
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
