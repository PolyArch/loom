// RUN: not loom-dfg-sim %s --graph residual_memref_load --arg 0=0 --memref 1=3 --output %t.memref.json 2>&1 | FileCheck %s --check-prefix=MEMREF
// RUN: not loom-dfg-sim %s --graph residual_llvm_load --memref 0=3 --output %t.llvm.json 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: not loom-pnr-map --dfg-mlir %s --graph residual_llvm_load --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload residual_llvm_load --output %t.llvm.csv --artifact %t.llvm.mapping.json 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: not loom-dfg-sim %s --graph residual_llvm_store --arg 0=3 --memref 1=0 --output %t.store.json 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: not loom-pnr-map --dfg-mlir %s --graph residual_llvm_store --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload residual_llvm_store --output %t.store.csv --artifact %t.store.mapping.json 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: not loom-dfg-sim %s --graph residual_global --arg 0=0 --output %t.global.json 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not loom-dfg-sim %s --graph residual_pointer_arithmetic --arg 0=0 --memref 1=3 --output %t.pointer.json 2>&1 | FileCheck %s --check-prefix=POINTER

// MEMREF: finalized graph contains residual memory operation 'memref.load'
// LLVM: finalized graph contains residual memory operation 'llvm.load'
// STORE: finalized graph contains residual memory operation 'llvm.store'
// GLOBAL: finalized graph contains forbidden memory root 'memref.get_global'
// POINTER: finalized graph contains residual pointer operation 'llvm.getelementptr'

module {
  memref.global "private" @table : memref<1xi32>

  dataflow.graph.func private @residual_memref_load(
      %start: none, %index: index, %memory: memref<?xi32>) -> none
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value = memref.load %memory[%index] : memref<?xi32>
    dataflow.graph.return %start : none
  }

  dataflow.graph.func private @residual_llvm_load(
      %start: none, %memory: !llvm.ptr) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = llvm.load %memory : !llvm.ptr -> i32
    %published:2 = dataflow.sync %start, %value : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @residual_llvm_store(
      %start: none, %value: i32, %memory: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    llvm.store %value, %memory : i32, !llvm.ptr
    dataflow.graph.return %start : none
  }

  dataflow.graph.func private @residual_global(
      %start: none, %index: index) -> none
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %memory = memref.get_global @table : memref<1xi32>
    %value = memref.load %memory[%index] : memref<1xi32>
    dataflow.graph.return %start : none
  }

  dataflow.graph.func private @residual_pointer_arithmetic(
      %start: none, %offset: i32, %memory: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %next = llvm.getelementptr %memory[%offset]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    dataflow.graph.return %start : none
  }
}
