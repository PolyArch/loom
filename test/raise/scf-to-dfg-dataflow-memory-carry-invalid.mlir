// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A loop-carried pointer is first-class graph data. Its object-scoped memory
// service remains invariant while the pointer recurrence advances exactly.

// CHECK: dataflow.memory.service %arg0 : !llvm.ptr -> memref<?xi8>
// CHECK-LABEL: dataflow.graph private @pointer_carry_graph
// CHECK: dataflow.load {{%.*}}[{{%.*}}] {{%.*}} : memref<?xi8>, !llvm.ptr
// CHECK: dataflow.carry {{.*}} : !llvm.ptr
// CHECK: llvm.getelementptr

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @pointer_carry
      domain(#dataflow.thread_domain<dense>)(
          %base: !llvm.ptr, %lower: index, %upper: index, %step: index)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%lower, %upper, %step, %base)
        <{operandSegmentSizes = array<i32: 4, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%lb: index, %ub: index, %stride: index, %memory: !llvm.ptr):
        %final = scf.for %i = %lb to %ub step %stride
            iter_args(%current = %memory) -> (!llvm.ptr) {
          %next = llvm.getelementptr %current[1]
              : (!llvm.ptr) -> !llvm.ptr, i8
          scf.yield %next : !llvm.ptr
        }
        %loaded = llvm.load %final : !llvm.ptr -> i8
        "loom.spatial_yield"(%loaded)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (i8) -> ()
    }) {graph_name = "pointer_carry_graph", source_maps = []} :
        (index, index, index, !llvm.ptr) -> i8
    dataflow.thread.yield
  }
}
