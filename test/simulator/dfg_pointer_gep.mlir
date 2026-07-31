// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// The typed SimulationWorkload/RuntimeInput execution path is covered by the
// pointer service-boundary and DFG pointer-execution anchors. This test owns
// only the mechanical lowering contract.

// CHECK-COUNT-2: dataflow.memory.service
// CHECK-LABEL: dataflow.graph private @offset_carry_graph
// CHECK-COUNT-2: llvm.getelementptr
// CHECK: dataflow.load {{%.*}}[{{%.*}}] {{%.*}} : memref<?xf32>, !llvm.ptr
// CHECK: dataflow.store {{%.*}}[{{%.*}}] {{%.*}} {{%.*}} : memref<?xf32>, !llvm.ptr
// CHECK-NOT: builtin.unrealized_conversion_cast

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.thread private @offset_carry
      domain(#dataflow.thread_domain<dense>)(
          %src: !llvm.ptr, %dst: !llvm.ptr,
          %lb: i32, %ub: i32, %step: i32, %bias: f32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%lb, %ub, %step, %bias, %src, %dst)
        <{operandSegmentSizes = array<i32: 6, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%lower: i32, %upper: i32, %stride: i32, %addend: f32,
           %source: !llvm.ptr, %destination: !llvm.ptr):
        scf.for %i = %lower to %upper step %stride : i32 {
          %src_at = llvm.getelementptr %source[%i]
              : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %dst_at = llvm.getelementptr %destination[%i]
              : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %data = llvm.load %src_at : !llvm.ptr -> f32
          %sum = arith.addf %data, %addend : f32
          llvm.store %sum, %dst_at : f32, !llvm.ptr
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "offset_carry_graph", source_maps = []} :
        (i32, i32, i32, f32, !llvm.ptr, !llvm.ptr) -> ()
    dataflow.thread.yield
  }
}
