// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// The thread keeps each LLVM pointer as value-plane data and explicitly
// acquires the typed memory service used by the graph.

// CHECK-LABEL: dataflow.thread private @imported_view
// CHECK: %[[SERVICE:.*]] = dataflow.memory.service %arg0 : !llvm.ptr -> memref<?xi32>
// CHECK: dataflow.graph.launch @imported_view_graph
// CHECK-SAME: values(%arg1, %arg0)
// CHECK-SAME: memories(%[[SERVICE]])

// CHECK-LABEL: dataflow.thread private @two_imported_views
// CHECK: dataflow.memory.service %arg0 : !llvm.ptr -> memref<?xi8>
// CHECK: dataflow.memory.service %arg0 : !llvm.ptr -> memref<?xi32>
// CHECK: dataflow.graph.launch @two_imported_views_graph
// CHECK-SAME: values(%arg0)

// CHECK-LABEL: dataflow.graph private @imported_view_graph(
// CHECK-SAME: [[INDEX:%[^, )]+]]: i64, [[BASE:%[^, )]+]]: !llvm.ptr
// CHECK-SAME: [[MEM:%[^, )]+]]: memref<?xi32>)
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[ADDR:.*]] = llvm.getelementptr [[BASE]][[[INDEX]]]
// CHECK: %[[DATA:.*]], %[[DONE:.*]] = dataflow.load [[MEM]][%[[ADDR]]]
// CHECK: dataflow.graph.return

// CHECK-LABEL: dataflow.graph private @two_imported_views_graph(
// CHECK-SAME: [[BASE:%[^, )]+]]: !llvm.ptr
// CHECK-SAME: [[BYTE_MEM:%[^, )]+]]: memref<?xi8>, [[WORD_MEM:%[^, )]+]]: memref<?xi32>)
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: dataflow.load [[BYTE_MEM]]
// CHECK: dataflow.load [[WORD_MEM]]

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>,
  llvm.data_layout = "e-p:64:64"
} {
  dataflow.thread private @imported_view
      domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr, %index: i64)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%index, %pointer)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%offset: i64, %base: !llvm.ptr):
        %address = llvm.getelementptr %base[%offset]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %data = llvm.load %address : !llvm.ptr -> i32
        "loom.spatial_yield"(%data)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
    }) {graph_name = "imported_view_graph", source_maps = []} :
        (i64, !llvm.ptr) -> i32
    dataflow.thread.yield
  }

  dataflow.thread private @two_imported_views
      domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%pointer)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%base: !llvm.ptr):
        %byte = llvm.load %base : !llvm.ptr -> i8
        %word = llvm.load %base : !llvm.ptr -> i32
        %extended = arith.extui %byte : i8 to i32
        %sum = arith.addi %extended, %word : i32
        "loom.spatial_yield"(%sum)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
    }) {graph_name = "two_imported_views_graph", source_maps = []} :
        (!llvm.ptr) -> i32
    dataflow.thread.yield
  }
}
