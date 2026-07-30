// RUN: not loom-dfg-sim %s --graph residual_memory_bridge --memref 0=7 \
// RUN:   --output %t.json 2>&1 | FileCheck %s

// CHECK: finalized graph contains forbidden operation 'builtin.unrealized_conversion_cast'

module {
  dataflow.graph private @residual_memory_bridge(
      %start: none, %memory: memref<?xi32>) -> i32
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %raw = builtin.unrealized_conversion_cast %memory
        : memref<?xi32> to !llvm.ptr
    %view = builtin.unrealized_conversion_cast %raw
        : !llvm.ptr to memref<?xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %value, %done = dataflow.load %view[%index] %start : memref<?xi32>
    dataflow.graph.return %done, %value : none, i32
  }
}
