// RUN: not loom-raise-opt --loom-lower-scf-to-dfg %s 2>&1 | FileCheck %s

// A two-byte recurrence cannot address f32 elements without changing source
// semantics. Publication fails instead of inventing a truncated element view.

// CHECK: memory pointer input remains after graph memory normalization: '!llvm.ptr'
// CHECK-NOT: dataflow.graph private @unaligned_byte_recurrence_graph

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @unaligned_byte_recurrence
      domain(#dataflow.thread_domain<dense>)(%base: !llvm.ptr, %limit: i64)
      ctrl (%ctrl: none) {
    %sum = "loom.spatial_region"(%limit, %base)
        <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%bound: i64, %memory: !llvm.ptr):
        %zero = arith.constant 0 : i64
        %step = arith.constant 2 : i64
        %initial = arith.constant 0.0 : f32
        %result:2 = scf.while (%state = %initial, %offset = %zero)
            : (f32, i64) -> (f32, i64) {
          %ptr = llvm.getelementptr inbounds|nuw %memory[%offset]
              : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %value = llvm.load %ptr : !llvm.ptr -> f32
          %next_state = arith.addf %state, %value : f32
          %next_offset = arith.addi %offset, %step : i64
          %more = arith.cmpi ult, %next_offset, %bound : i64
          scf.condition(%more) %next_state, %next_offset : f32, i64
        } do {
        ^bb0(%state: f32, %offset: i64):
          scf.yield %state, %offset : f32, i64
        }
        "loom.spatial_yield"(%result#0)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "unaligned_byte_recurrence_graph", source_maps = []} :
        (i64, !llvm.ptr) -> f32
    dataflow.thread.yield
  }
}
