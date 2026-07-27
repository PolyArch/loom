// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/candidate.mlir -o %t.once.mlir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.once.mlir -o %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/mixed.mlir | FileCheck %s --check-prefix=MIXED --implicit-check-not=loom.spatial_region

// Publication owns the transition from spatial candidate to canonical
// Dataflow. A module that already holds finalized graphs has nothing left to
// publish, so running the pipeline again is an identity. Re-finalizing a graph
// would drive its canonical dataflow.load/store back through the graph-memory
// owner and rebuild a memory-event network that already exists.

// A module may hold both. The finalized graph keeps the exact ctrl/done
// network it was published with, and the spatial candidate beside it is
// published and finalized in the same transaction.

// MIXED-LABEL: dataflow.graph private @existing_graph
// MIXED: %[[READ_CTRL:.*]]:2 = dataflow.sync %{{.*}}, %{{.*}} : (none, none) -> (none, none)
// MIXED: %[[DATA:.*]], %[[READ_DONE:.*]] = dataflow.load %arg1[%{{.*}}] %[[READ_CTRL]]#0 : memref<8xi32>
// MIXED: %[[WRITE_CTRL:.*]]:2 = dataflow.sync %{{.*}}, %[[READ_DONE]] : (none, none) -> (none, none)
// MIXED: dataflow.store %arg2[%{{.*}}] %[[DATA]] %[[WRITE_CTRL]]#0 : memref<8xi32>
// MIXED: dataflow.graph.return values() streams() memories() complete(%{{.*}}, %{{.*}} : none, none)

// MIXED-LABEL: dataflow.graph private @new_graph
// MIXED: %[[NEW_DATA:.*]], %[[NEW_DONE:.*]] = dataflow.load %arg1[%{{.*}}] %arg0 : memref<1xi32>
// MIXED: dataflow.store %arg2[%{{.*}}] %[[NEW_DATA]] %[[NEW_DONE]] : memref<1xi32>

//--- candidate.mlir
dataflow.thread private @spatial_copy domain(#dataflow.thread_domain<dense>)(
    %src: memref<8xi32>, %dst: memref<8xi32>) ctrl (%ctrl: none) {
  "loom.spatial_region"(%src, %dst)
      <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%source: memref<8xi32>, %target: memref<8xi32>):
      memref.copy %source, %target : memref<8xi32> to memref<8xi32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "copy_graph", source_maps = []} :
      (memref<8xi32>, memref<8xi32>) -> ()
  dataflow.thread.yield
}

//--- mixed.mlir
dataflow.thread private @existing_thread domain(#dataflow.thread_domain<dense>)(
    %src: memref<8xi32>, %dst: memref<8xi32>) ctrl (%ctrl: none) {
  %done = dataflow.graph.launch @existing_graph deps(%ctrl) values()
      stream_inputs() memories(%src, %dst) stream_outputs()
      : (none, memref<8xi32>, memref<8xi32>) -> none
  dataflow.thread.yield %done : none
}

dataflow.graph private @existing_graph(
    %arg0: none, %arg1: memref<8xi32>, %arg2: memref<8xi32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %0 = dataflow.constant %arg0 {const_value = 0 : index} : index
  %1 = dataflow.constant %arg0 {const_value = 8 : index} : index
  %2 = dataflow.constant %arg0 {const_value = 1 : index} : index
  %3 = arith.index_cast %0 : index to i32
  %4 = arith.index_cast %1 : index to i32
  %5 = arith.index_cast %2 : index to i32
  %iv, %phase = dataflow.stream %3, %4, %5 step add while slt : i32
  %6 = arith.index_cast %iv : i32 to index
  %7 = dataflow.carry %phase, %arg0, %8#1 : none
  %8:2 = dataflow.demux %phase, %7 : (i1, none) -> (none, none)
  %9 = dataflow.carry %phase, %arg0, %15 : none
  %10 = dataflow.carry %phase, %arg0, %15 : none
  %11:2 = dataflow.demux %phase, %9 : (i1, none) -> (none, none)
  %12:2 = dataflow.demux %phase, %10 : (i1, none) -> (none, none)
  %13:2 = dataflow.sync %8#1, %11#1 : (none, none) -> (none, none)
  %data, %done = dataflow.load %arg1[%6] %13#0 : memref<8xi32>
  %14:2 = dataflow.sync %12#1, %done : (none, none) -> (none, none)
  %15 = dataflow.store %arg2[%6] %data %14#0 : memref<8xi32>
  dataflow.graph.return values() streams() memories() complete(%8#0, %12#0 : none, none)
}

dataflow.thread private @new_thread domain(#dataflow.thread_domain<dense>)(
    %src: memref<1xi32>, %dst: memref<1xi32>) ctrl (%ctrl: none) {
  "loom.spatial_region"(%src, %dst)
      <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%source: memref<1xi32>, %target: memref<1xi32>):
      memref.copy %source, %target : memref<1xi32> to memref<1xi32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "new_graph", source_maps = []} :
      (memref<1xi32>, memref<1xi32>) -> ()
  dataflow.thread.yield
}
