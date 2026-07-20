// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/thread.mlir | FileCheck %s --check-prefix=THREAD
// RUN: loom %t.dir/thread.mlir | loom | FileCheck %s --check-prefix=THREAD-ROUNDTRIP
// RUN: loom %t.dir/thread.mlir --emit-bytecode | loom | FileCheck %s --check-prefix=THREAD-ROUNDTRIP
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/same-root-cast.mlir | FileCheck %s --check-prefix=SAME-ROOT
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/unknown-root.mlir | FileCheck %s --check-prefix=UNKNOWN-ROOT

//--- thread.mlir
module {
  dataflow.thread private @source_noalias(
      %left: memref<?xi32> {llvm.noalias, test.arg = "left"},
      %index: index {test.arg = "index"},
      %right: memref<?xi32> {llvm.noalias, test.arg = "right"},
      %value: i32 {test.arg = "value"}) ctrl (%ctrl: none) {
    "loom.spatial_region"(%index, %value, %left, %right)
        <{operandSegmentSizes = array<i32: 2, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%position: index, %payload: i32,
           %left_memory: memref<?xi32>, %right_memory: memref<?xi32>):
        memref.store %payload, %left_memory[%position] : memref<?xi32>
        %loaded = memref.load %right_memory[%position] : memref<?xi32>
        memref.store %loaded, %left_memory[%position] : memref<?xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "g_source_noalias_0", source_maps = []} :
        (index, i32, memref<?xi32>, memref<?xi32>) -> ()
    dataflow.thread.yield
  }
}

// THREAD-LABEL: dataflow.thread private @source_noalias(
// THREAD-SAME: %{{.*}}: memref<?xi32> {llvm.noalias, test.arg = "left"},
// THREAD-SAME: %{{.*}}: index {test.arg = "index"},
// THREAD-SAME: %{{.*}}: memref<?xi32> {llvm.noalias, test.arg = "right"},
// THREAD-SAME: %{{.*}}: i32 {test.arg = "value"})
// THREAD-LABEL: dataflow.graph private @g_source_noalias_0(
// THREAD-SAME: %[[START:.*]]: none,
// THREAD-SAME: %[[INDEX:.*]]: index {test.arg = "index"},
// THREAD-SAME: %[[VALUE:.*]]: i32 {test.arg = "value"},
// THREAD-SAME: %[[LEFT:.*]]: memref<?xi32> {llvm.noalias, test.arg = "left"},
// THREAD-SAME: %[[RIGHT:.*]]: memref<?xi32> {llvm.noalias, test.arg = "right"})
// THREAD: %{{.*}} = dataflow.store %[[LEFT]][%[[INDEX]]] %[[VALUE]] %[[START]] : memref<?xi32>
// THREAD: %{{.*}}, %{{.*}} = dataflow.load %[[RIGHT]][%[[INDEX]]] %[[START]] : memref<?xi32>
// THREAD: %{{.*}} = dataflow.store %[[LEFT]][%[[INDEX]]] %{{.*}} %{{.*}} : memref<?xi32>

// THREAD-ROUNDTRIP-LABEL: dataflow.thread private @source_noalias(
// THREAD-ROUNDTRIP-SAME: %{{.*}}: memref<?xi32> {llvm.noalias, test.arg = "left"},
// THREAD-ROUNDTRIP-SAME: %{{.*}}: index {test.arg = "index"},
// THREAD-ROUNDTRIP-SAME: %{{.*}}: memref<?xi32> {llvm.noalias, test.arg = "right"},
// THREAD-ROUNDTRIP-SAME: %{{.*}}: i32 {test.arg = "value"})

//--- same-root-cast.mlir
module {
  dataflow.thread private @same_root_cast(
      %memory: memref<4xi32> {llvm.noalias},
      %limit: index, %seed: i32) ctrl (%ctrl: none) {
    %view = memref.cast %memory : memref<4xi32> to memref<?xi32>
    "loom.spatial_region"(%limit, %seed, %memory, %view)
        <{operandSegmentSizes = array<i32: 2, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%end: index, %initial: i32,
           %target: memref<4xi32>, %alias: memref<?xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %result = scf.for %i = %c0 to %end step %c1
            iter_args(%value = %initial) -> (i32) {
          memref.store %value, %target[%i] : memref<4xi32>
          %loaded = memref.load %alias[%i] : memref<?xi32>
          scf.yield %loaded : i32
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "g_same_root_cast_0", source_maps = []} :
        (index, i32, memref<4xi32>, memref<?xi32>) -> ()
    dataflow.thread.yield
  }
}

// SAME-ROOT-LABEL: dataflow.graph private @g_same_root_cast_0
// SAME-ROOT: %[[STORE_DONE:.*]] = dataflow.store
// SAME-ROOT: %{{.*}}, %{{.*}} = dataflow.load {{.*}} %[[STORE_DONE]]

//--- unknown-root.mlir
module {
  dataflow.thread private @unknown_root(
      %direct: memref<?xi32> {llvm.noalias, test.arg = "direct"},
      %other: memref<?xi32>, %limit: index, %seed: i32)
      ctrl (%ctrl: none) {
    %unknown = builtin.unrealized_conversion_cast %direct, %other
        : memref<?xi32>, memref<?xi32> to memref<?xi32>
    "loom.spatial_region"(%limit, %seed, %direct, %unknown)
        <{operandSegmentSizes = array<i32: 2, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%end: index, %initial: i32,
           %target: memref<?xi32>, %alias: memref<?xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %result = scf.for %i = %c0 to %end step %c1
            iter_args(%value = %initial) -> (i32) {
          memref.store %value, %target[%i] : memref<?xi32>
          %loaded = memref.load %alias[%i] : memref<?xi32>
          scf.yield %loaded : i32
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "g_unknown_root_0", source_maps = []} :
        (index, i32, memref<?xi32>, memref<?xi32>) -> ()
    dataflow.thread.yield
  }
}

// UNKNOWN-ROOT-LABEL: dataflow.graph private @g_unknown_root_0(
// UNKNOWN-ROOT-SAME: memref<?xi32> {test.arg = "direct"}
// UNKNOWN-ROOT-NOT: llvm.noalias
// UNKNOWN-ROOT: %[[UNKNOWN_STORE_DONE:.*]] = dataflow.store
// UNKNOWN-ROOT: %{{.*}}, %{{.*}} = dataflow.load {{.*}} %[[UNKNOWN_STORE_DONE]]
