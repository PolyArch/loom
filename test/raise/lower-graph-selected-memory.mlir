// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A branch-selected pointer becomes two launch-owned typed memory views. The
// graph selects execution around the loads, not a dynamic memory capability.

// CHECK-LABEL: dataflow.thread private @branch_selected_load
// CHECK-COUNT-2: dataflow.memory.service
// CHECK: dataflow.graph.launch @branch_selected_load_graph
// CHECK-SAME: values(%arg2, %arg3, %arg0, %arg1)
// CHECK-LABEL: dataflow.graph private @branch_selected_load_graph(
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: [[A:%[^, )]+]]: memref<?xf32>, [[B:%[^, )]+]]: memref<?xf32>)
// CHECK-COUNT-2: dataflow.load
// CHECK: dataflow.mux
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: llvm.load

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @branch_selected_load
      domain(#dataflow.thread_domain<dense>)(
          %a: !llvm.ptr, %b: !llvm.ptr, %choose_a: i1, %ordinal: i64)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%choose_a, %ordinal, %a, %b)
        <{operandSegmentSizes = array<i32: 4, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%choose: i1, %index: i64, %a_base: !llvm.ptr,
           %b_base: !llvm.ptr):
        %a_ptr = llvm.getelementptr inbounds %a_base[%index]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %b_ptr = llvm.getelementptr inbounds %b_base[%index]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %selected = scf.if %choose -> (!llvm.ptr) {
          scf.yield %a_ptr : !llvm.ptr
        } else {
          scf.yield %b_ptr : !llvm.ptr
        }
        %loaded = llvm.load %selected : !llvm.ptr -> f32
        "loom.spatial_yield"(%loaded)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "branch_selected_load_graph", source_maps = []} :
        (i1, i64, !llvm.ptr, !llvm.ptr) -> f32
    dataflow.thread.yield
  }
}
