// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/post-done.mlir --graph post_done_state --output %t.post.json 2>&1 | FileCheck %s --check-prefix=POST
// RUN: not loom-dfg-sim %t.dir/dynamic.mlir --graph invalid_dynamic_memory_export --arg 0=13 --output %t.dynamic.json 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not loom-dfg-sim %t.dir/import.mlir --graph invalid_memory_export --output %t.import.json 2>&1 | FileCheck %s --check-prefix=IMPORT
// RUN: not loom-dfg-sim %t.dir/pointer.mlir --graph invalid_fresh_pointer_export --output %t.pointer.json 2>&1 | FileCheck %s --check-prefix=POINTER

// POST: retirement frontier does not cover close/reset of 'dataflow.stream'
// DYNAMIC: memref.alloc dynamic extent must be a graph value input
// IMPORT: nontrivial graph uses raw start as a retirement completion witness
// POINTER: fresh memory export must use a memref result

//--- post-done.mlir
module {
  dataflow.graph private @post_done_state(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %c0 = dataflow.constant %start {const_value = 0 : i32} : i32
    %c1 = dataflow.constant %start {const_value = 1 : i32} : i32
    %c2 = dataflow.constant %start {const_value = 2 : i32} : i32
    %iv, %phase = dataflow.stream %c0, %c2, %c1
        step add while slt : i32
    %published:2 = dataflow.sync %start, %c0
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- dynamic.mlir
module {
  dataflow.graph private @invalid_dynamic_memory_export(
      %start: none, %value: i32) -> memref<?xi32>
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %extent = dataflow.constant %start {const_value = 1 : index} : index
    %slot = memref.alloc(%extent) : memref<?xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %done = dataflow.store %slot[%index] %value %start : memref<?xi32>
    dataflow.graph.return values() streams()
        memories(%slot : memref<?xi32>) complete(%done : none)
  }
}

//--- import.mlir
module {
  dataflow.graph private @invalid_memory_export(
      %start: none, %memory: memref<?xi32>) -> memref<?xi32>
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    %unused = dataflow.constant %start {const_value = 1 : i32} : i32
    dataflow.graph.return values() streams()
        memories(%memory : memref<?xi32>) complete(%start : none)
  }
}

//--- pointer.mlir
module {
  dataflow.graph private @invalid_fresh_pointer_export(%start: none)
      -> !llvm.ptr
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc() : memref<1xi32>
    %raw = builtin.unrealized_conversion_cast %slot : memref<1xi32> to !llvm.ptr
    dataflow.graph.return values() streams()
        memories(%raw : !llvm.ptr) complete(%start : none)
  }
}
