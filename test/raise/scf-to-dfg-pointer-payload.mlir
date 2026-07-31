// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A pointer stored as descriptor data remains a first-class LLVM pointer.
// Both the descriptor address and the stored pointer obtain exact object
// services, so runtime capture can preserve the payload's object provenance.

// CHECK-COUNT-2: dataflow.memory.service
// CHECK-LABEL: dataflow.graph private @pointer_payload_graph
// CHECK: dataflow.store {{%.*}}[{{%.*}}] {{%.*}} {{%.*}} : memref<?xi64>, !llvm.ptr, !llvm.ptr
// CHECK-NOT: builtin.unrealized_conversion_cast

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @pointer_payload
      domain(#dataflow.thread_domain<dense>)(
          %descriptor: !llvm.ptr, %target: !llvm.ptr) ctrl (%ctrl: none) {
    "loom.spatial_region"(%descriptor, %target)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%descriptor_arg: !llvm.ptr, %target_arg: !llvm.ptr):
        llvm.store %target_arg, %descriptor_arg : !llvm.ptr, !llvm.ptr
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "pointer_payload_graph", source_maps = []} :
        (!llvm.ptr, !llvm.ptr) -> ()
    dataflow.thread.yield
  }
}
