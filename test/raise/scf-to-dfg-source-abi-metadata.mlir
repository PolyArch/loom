// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/function.mlir | FileCheck %s --check-prefix=FUNCTION
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/thread.mlir | FileCheck %s --check-prefix=THREAD

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
