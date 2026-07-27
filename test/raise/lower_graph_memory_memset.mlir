// RUN: env LOOM_INDEX_WIDTH=64 loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir

// Memset is byte-defined even when a later access uses a wider element type.
// The dynamic byte value and exact byte count therefore enter one structured
// byte-store loop. The recursive region lowerer owns repetition and carries
// the loop's final memory frontier into the following load.

// CHECK-LABEL: dataflow.graph private @dynamic_byte_memset(
// CHECK-SAME: %[[START:.*]]: none, %[[COUNT:.*]]: i64, %[[FILL:.*]]: i8, %[[BASE:.*]]: !llvm.ptr)
// CHECK-DAG: builtin.unrealized_conversion_cast %[[BASE]] : !llvm.ptr to memref<?xi8>
// CHECK-DAG: builtin.unrealized_conversion_cast %[[BASE]] : !llvm.ptr to memref<?xi32>
// CHECK-DAG: arith.index_cast %[[COUNT]] : i64 to index
// CHECK-DAG: dataflow.load {{.*}} [[LOAD_CTRL:%[0-9]+]]#0 : memref<?xi32>
// CHECK: dataflow.stream
// CHECK: dataflow.invariant {{.*}}, %[[FILL]] : i8
// CHECK: dataflow.gate
// CHECK: dataflow.store {{.*}} : memref<?xi8>
// CHECK: [[LOAD_CTRL]]:2 = dataflow.sync
// CHECK-NOT: llvm.intr.memset
// CHECK-NOT: scf.for
// CHECK: dataflow.graph.return

module {
  dataflow.graph private @dynamic_byte_memset(
      %start: none, %byte_count: i64, %fill: i8, %base: !llvm.ptr) -> (i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    "llvm.intr.memset"(%base, %fill, %byte_count)
      <{arg_attrs = [{llvm.align = 1 : i64}, {}, {}],
         isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %value = llvm.load %base : !llvm.ptr -> i32
    dataflow.graph.return %start, %value : none, i32
  }
}
