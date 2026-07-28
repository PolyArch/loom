// RUN: env LOOM_INDEX_WIDTH=64 loom-raise-opt --loom-lower-graph-memory %s \
// RUN:   | FileCheck %s

// A byte-address recurrence whose initial value and update are both aligned
// to the accessed element width is one integer access function. The memory
// capability remains invariant while only the integer offset recurs.

// CHECK-LABEL: dataflow.graph private @byte_offset_recurrence
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg2 : !llvm.ptr to memref<?xf32>
// CHECK: %[[OFFSET:.*]] = dataflow.carry {{.*}} : i64
// CHECK: %[[ELEMENT_OFFSET:.*]] = arith.shrsi %[[OFFSET]], %{{.*}} : i64
// CHECK: %[[ADDRESS:.*]] = arith.index_cast %[[ELEMENT_OFFSET]] : i64 to index
// CHECK: dataflow.load %[[MEM]][%[[ADDRESS]]]
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @byte_offset_recurrence(
      %arg0: none, %arg1: i64, %arg2: !llvm.ptr) -> f32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = arith.constant 0 : i64
    %step = arith.constant 4 : i64
    %initial = arith.constant 0.0 : f32
    %result:2 = scf.while (%sum = %initial, %offset = %zero)
        : (f32, i64) -> (f32, i64) {
      %ptr = llvm.getelementptr inbounds|nuw %arg2[%offset]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %value = llvm.load %ptr : !llvm.ptr -> f32
      %next_sum = arith.addf %sum, %value : f32
      %next_offset = arith.addi %offset, %step : i64
      %more = arith.cmpi ult, %next_offset, %arg1 : i64
      scf.condition(%more) %next_sum, %next_offset : f32, i64
    } do {
    ^bb0(%sum: f32, %offset: i64):
      scf.yield %sum, %offset : f32, i64
    }
    dataflow.graph.return values(%result#0 : f32) streams() memories()
        complete(%arg0 : none)
  }
}
