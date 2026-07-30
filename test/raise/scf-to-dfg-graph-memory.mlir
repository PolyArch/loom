// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/pointer-chain.mlir \
// RUN:   | FileCheck %s --check-prefix=CHAIN
// RUN: loom-raise-opt --loom-lower-graph-memory %t.dir/ranked.mlir \
// RUN:   | FileCheck %s --check-prefix=RANKED
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/dynamic-rank.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=DYNAMIC-RANK

// One chain of pointer arithmetic becomes one integer access function over a
// launch-owned typed view. Intermediate pointers never become graph entities.

// CHAIN-LABEL: dataflow.thread private @pointer_chain
// CHAIN: dataflow.graph.launch @pointer_chain_graph
// CHAIN-SAME: memories(%arg0)
// CHAIN-LABEL: dataflow.graph private @pointer_chain_graph(
// CHAIN-SAME: [[MEM:%[^, )]+]]: memref<?xf32>)
// CHAIN: %[[OUTER:.*]] = arith.index_cast %arg1 : i64 to index
// CHAIN: %[[MIDDLE:.*]] = arith.index_cast %arg2 : i64 to index
// CHAIN: %[[FIRST:.*]] = arith.addi %[[OUTER]], %[[MIDDLE]] : index
// CHAIN: %[[INNER:.*]] = arith.index_cast %arg3 : i64 to index
// CHAIN: %[[ADDRESS:.*]] = arith.addi %[[FIRST]], %[[INNER]] : index
// CHAIN: dataflow.load [[MEM]][%[[ADDRESS]]]
// CHAIN-NOT: builtin.unrealized_conversion_cast
// CHAIN-NOT: llvm.getelementptr
// CHAIN-NOT: llvm.load

//--- pointer-chain.mlir
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @pointer_chain
      domain(#dataflow.thread_domain<dense>)(
          %base: !llvm.ptr, %outer: i64, %middle: i64, %inner: i64)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%outer, %middle, %inner, %base)
        <{operandSegmentSizes = array<i32: 3, 0, 1, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%i: i64, %j: i64, %k: i64, %memory: !llvm.ptr):
        %first = llvm.getelementptr inbounds %memory[%i]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %second = llvm.getelementptr inbounds %first[%j]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %third = llvm.getelementptr inbounds %second[%k]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %loaded = llvm.load %third : !llvm.ptr -> f32
        "loom.spatial_yield"(%loaded)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "pointer_chain_graph", source_maps = []} :
        (i64, i64, i64, !llvm.ptr) -> f32
    dataflow.thread.yield
  }
}

// Static identity-layout memrefs flatten row-major indices mechanically.

// RANKED-LABEL: dataflow.graph private @rank3_row_major(
// RANKED: %[[D1:.*]] = dataflow.constant %arg0 {const_value = 5 : index} : index
// RANKED: %[[M1:.*]] = arith.muli %arg1, %[[D1]] : index
// RANKED: %[[A1:.*]] = arith.addi %[[M1]], %arg2 : index
// RANKED: %[[D2:.*]] = dataflow.constant %arg0 {const_value = 7 : index} : index
// RANKED: %[[M2:.*]] = arith.muli %[[A1]], %[[D2]] : index
// RANKED: %[[ADDRESS:.*]] = arith.addi %[[M2]], %arg3 : index
// RANKED: dataflow.load %arg4[%[[ADDRESS]]]
// RANKED: dataflow.store %arg4[{{%.*}}]

//--- ranked.mlir
module {
  dataflow.graph private @rank3_row_major(
      %start: none, %i: index, %j: index, %k: index,
      %memory: memref<3x5x7xf32>) -> ()
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value = memref.load %memory[%i, %j, %k] : memref<3x5x7xf32>
    memref.store %value, %memory[%i, %j, %k] : memref<3x5x7xf32>
    dataflow.graph.return %start : none
  }
}

// A dynamic multidimensional shape has no exact static row-major projection.

// DYNAMIC-RANK: memref.load requires an identity-layout memref whose shape is static when rank exceeds one

//--- dynamic-rank.mlir
module {
  dataflow.graph private @dynamic_rank(
      %start: none, %i: index, %j: index,
      %memory: memref<?x?xf32>) -> ()
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value = memref.load %memory[%i, %j] : memref<?x?xf32>
    dataflow.graph.return %start : none
  }
}
