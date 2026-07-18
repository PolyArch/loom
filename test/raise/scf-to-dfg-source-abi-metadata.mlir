// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/function.mlir | FileCheck %s --check-prefix=FUNCTION
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/thread.mlir | FileCheck %s --check-prefix=THREAD
// RUN: loom %t.dir/thread.mlir | loom | FileCheck %s --check-prefix=THREAD-ROUNDTRIP
// RUN: loom %t.dir/thread.mlir --emit-bytecode | loom | FileCheck %s --check-prefix=THREAD-ROUNDTRIP
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/same-root-cast.mlir | FileCheck %s --check-prefix=SAME-ROOT

//--- function.mlir
module {
  func.func private @source_interface_metadata(
      %memory: !llvm.ptr {llvm.noalias, test.arg = "memory"},
      %bias: i32 {test.arg = "bias"})
      -> (!llvm.ptr {test.result = "memory"},
          i32 {test.result = "sum"}) {
    %loaded = llvm.load %memory : !llvm.ptr -> i32
    %sum = llvm.add %loaded, %bias : i32
    return %memory, %sum : !llvm.ptr, i32
  }
}

// FUNCTION-LABEL: dataflow.graph private @g_source_interface_metadata_0(
// FUNCTION-SAME: %{{.*}}: none,
// FUNCTION-SAME: %{{.*}}: i32 {test.arg = "bias"},
// FUNCTION-SAME: %{{.*}}: !llvm.ptr {llvm.noalias, test.arg = "memory"})
// FUNCTION-SAME: -> (i32 {test.result = "sum"}, !llvm.ptr {test.result = "memory"})

//--- thread.mlir
module {
  dataflow.thread private @source_noalias(
      %left: memref<?xi32> {llvm.noalias, test.arg = "left"},
      %index: index {test.arg = "index"},
      %right: memref<?xi32> {llvm.noalias, test.arg = "right"},
      %value: i32 {test.arg = "value"}) ctrl (%ctrl: none) {
    memref.store %value, %left[%index] : memref<?xi32>
    %loaded = memref.load %right[%index] : memref<?xi32>
    memref.store %loaded, %left[%index] : memref<?xi32>
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %limit step %c1
        iter_args(%value = %seed) -> (i32) {
      memref.store %value, %memory[%i] : memref<4xi32>
      %loaded = memref.load %view[%i] : memref<?xi32>
      scf.yield %loaded : i32
    }
    dataflow.thread.yield
  }
}

// SAME-ROOT-LABEL: dataflow.graph private @g_same_root_cast_0
// SAME-ROOT: %[[STORE_DONE:.*]] = dataflow.store
// SAME-ROOT: %{{.*}}, %{{.*}} = dataflow.load {{.*}} %[[STORE_DONE]]
