// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/load.mlir --graph llvm_load_ptr --memref 0=100,102,105 --output %t.ptr.json 2>&1 | FileCheck %s --check-prefix=PTR
// RUN: not loom-dfg-sim %t.dir/gep.mlir --graph llvm_load_gep --memref 0=100,102,105 --output %t.gep.json 2>&1 | FileCheck %s --check-prefix=GEP

// PTR: finalized graph contains residual memory operation 'llvm.load'

// GEP: finalized graph contains residual pointer operation 'llvm.getelementptr'

//--- load.mlir
module {
  dataflow.graph private @llvm_load_ptr(%ctrl: none, %ptr: !llvm.ptr)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data = llvm.load %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    %published:2 = dataflow.sync %ctrl, %data
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- gep.mlir
module {
  dataflow.graph private @llvm_load_gep(%ctrl: none, %ptr: !llvm.ptr)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %next = llvm.getelementptr inbounds|nuw %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %next {alignment = 4 : i64} : !llvm.ptr -> i32
    %published:2 = dataflow.sync %ctrl, %data
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
