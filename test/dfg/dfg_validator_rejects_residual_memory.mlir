// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/memref.mlir --graph residual_memref_load --arg 0=0 --memref 1=3 --output %t.memref.json 2>&1 | FileCheck %s --check-prefix=MEMREF
// RUN: not loom-dfg-sim %t.dir/global.mlir --graph residual_global --arg 0=0 --output %t.global.json 2>&1 | FileCheck %s --check-prefix=GLOBAL

// MEMREF: finalized graph contains residual memory operation 'memref.load'
// GLOBAL: finalized graph contains forbidden memory root 'memref.get_global'

//--- memref.mlir
module {
  dataflow.graph private @residual_memref_load(
      %start: none, %index: index, %memory: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value = memref.load %memory[%index] : memref<?xi32>
    dataflow.graph.return %start : none
  }
}

//--- global.mlir
module {
  memref.global "private" @table : memref<1xi32>

  dataflow.graph private @residual_global(
      %start: none, %index: index) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %memory = memref.get_global @table : memref<1xi32>
    %value = memref.load %memory[%index] : memref<1xi32>
    dataflow.graph.return %start : none
  }
}
