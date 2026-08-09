// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/pointer-chain.mlir \
// RUN:   | FileCheck %s --check-prefix=CHAIN
// RUN: loom-raise-opt --loom-lower-graph-memory %t.dir/ranked.mlir \
// RUN:   | FileCheck %s --check-prefix=RANKED
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/dynamic-rank.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=DYNAMIC-RANK

// A source pointer remains first-class graph data. The enclosing thread
// explicitly acquires one object-scoped memory service, while the complete
// typed GEP chain remains the pointer-addressed access function.

// CHAIN-LABEL: dataflow.thread private @pointer_chain
// CHAIN: %[[SERVICE:.*]] = dataflow.memory.service %arg0 : !llvm.ptr -> memref<?xf32>
// CHAIN: dataflow.graph.launch @pointer_chain_graph
// CHAIN-SAME: values(%arg1, %arg2, %arg3, %arg0)
// CHAIN-SAME: memories(%[[SERVICE]])
// CHAIN-LABEL: dataflow.graph private @pointer_chain_graph(
// CHAIN-SAME: %[[BASE:[^, )]+]]: !llvm.ptr
// CHAIN-SAME: [[MEM:%[^, )]+]]: memref<?xf32>)
// CHAIN: %[[FIRST:.*]] = llvm.getelementptr inbounds %[[BASE]]
// CHAIN: %[[SECOND:.*]] = llvm.getelementptr inbounds %[[FIRST]]
// CHAIN: %[[ADDRESS:.*]] = llvm.getelementptr inbounds %[[SECOND]]
// CHAIN: dataflow.load [[MEM]][%[[ADDRESS]]]
// CHAIN-NOT: builtin.unrealized_conversion_cast
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
        <{operandSegmentSizes = array<i32: 4, 0, 0, 0>,
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

// Static ranked memrefs use their exact offset and per-dimension strides.

// RANKED-LABEL: dataflow.graph private @rank3_row_major(
// RANKED: %[[S0:.*]] = dataflow.constant %arg0 {const_value = 35 : index} : index
// RANKED: %[[M0:.*]] = arith.muli %arg1, %[[S0]] : index
// RANKED: %[[S1:.*]] = dataflow.constant %arg0 {const_value = 7 : index} : index
// RANKED: %[[M1:.*]] = arith.muli %arg2, %[[S1]] : index
// RANKED: %[[A1:.*]] = arith.addi %[[M0]], %[[M1]] : index
// RANKED: %[[ADDRESS:.*]] = arith.addi %[[A1]], %arg3 : index
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

// A dynamic multidimensional shape has no exact static stride projection.

// DYNAMIC-RANK: memref.load has no exactly addressable ranked layout
// DYNAMIC-RANK-SAME: dynamic shape is not exact for this ranked layout

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
