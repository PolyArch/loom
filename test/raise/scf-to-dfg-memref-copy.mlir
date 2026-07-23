// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/layout.mlir -o %t.layout.mlir
// RUN: FileCheck %s --check-prefix=EXPAND --implicit-check-not=memref.copy --implicit-check-not=linalg. --implicit-check-not=memref.dma --implicit-check-not=dataflow.transfer < %t.layout.mlir
// RUN: loom-dfg-sim %t.layout.mlir --graph copy_graph --memref 0=1,2,3,4,5,6,7 --memref 1=0,0,0,0,0,0,0 --output %t.seven.json 2>&1
// RUN: FileCheck %s --check-prefix=SEVEN < %t.seven.json
// RUN: env LOOM_INDEX_WIDTH=4 loom-raise-opt --loom-lower-scf-to-dfg %t.dir/override.mlir -o %t.override.mlir
// RUN: env LOOM_INDEX_WIDTH=4 loom-dfg-sim %t.override.mlir --graph override_graph --memref 0=1,2,3,4,5,6,7,8 --memref 1=0,0,0,0,0,0,0,0 --output %t.eight.json
// RUN: FileCheck %s --check-prefix=EIGHT < %t.eight.json
// RUN: not loom-raise-opt --loom-lower-scf-to-dfg -split-input-file --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/unusable.mlir 2>&1 | FileCheck %s --check-prefix=REJECT --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// A SpatialCore-owned memref.copy carries no independent Dataflow semantics.
// It is expanded into a structured element loop inside the publication
// transaction, so the canonical graph-memory owner derives the ordinary
// dataflow.load/store pair and its ctrl/done memory-event network. The store
// must observe the load's completion through the read frontier rather than
// merely appear after it in program order.
//
// The expanded loop is an ordinary structured loop, so it shares one canonical
// set of index constants with the hand-written loop beside it instead of
// seeding a second set of constant sources.

// EXPAND-LABEL: dataflow.graph private @copy_graph
// EXPAND: %[[IV:.*]], %{{.*}} = dataflow.stream
// EXPAND: %[[ADDR:.*]] = arith.index_cast %[[IV]]
// EXPAND: %[[DATA:.*]], %[[READ_DONE:.*]] = dataflow.load %arg1[%[[ADDR]]] %{{.*}} : memref<7xi32>
// EXPAND: %[[WRITE_CTRL:.*]]:2 = dataflow.sync %{{.*}}, %[[READ_DONE]] : (none, none) -> (none, none)
// EXPAND: dataflow.store %arg2[%[[ADDR]]] %[[DATA]] %[[WRITE_CTRL]]#0 : memref<7xi32>
// EXPAND: dataflow.graph.return

// An expanded loop is an ordinary structured loop beside the hand-written one,
// so the two share one canonical set of index constants.

// EXPAND-LABEL: dataflow.graph private @shared_constants_graph
// EXPAND-COUNT-3: dataflow.constant
// EXPAND-NOT: dataflow.constant
// EXPAND: dataflow.stream

// A one-element copy needs no stream, and an empty one needs no access at all.
// Both still complete the graph.

// EXPAND-LABEL: dataflow.graph private @unit_graph
// EXPAND: %[[UNIT_DATA:.*]], %[[UNIT_DONE:.*]] = dataflow.load %arg1[%{{.*}}] %arg0 : memref<1xi32>
// EXPAND: dataflow.store %arg2[%{{.*}}] %[[UNIT_DATA]] %[[UNIT_DONE]] : memref<1xi32>
// EXPAND-LABEL: dataflow.graph private @empty_graph
// EXPAND-NEXT: dataflow.graph.return %arg0 : none

// The closest enclosing data layout owns the index width, so the declared four
// bit index governs admissibility even though the process default is wider.
// Seven elements fit that signed domain and every one of them moves.

// SEVEN: "arg1": [
// SEVEN-NEXT: "i32:1",
// SEVEN-NEXT: "i32:2",
// SEVEN-NEXT: "i32:3",
// SEVEN-NEXT: "i32:4",
// SEVEN-NEXT: "i32:5",
// SEVEN-NEXT: "i32:6",
// SEVEN-NEXT: "i32:7"
// SEVEN: "dataflow.load": 7
// SEVEN: "dataflow.store": 7

// A declared width also overrides the configured one, so eight elements are
// admissible under a declared sixty-four bit index even with a four bit
// environment override, and the destination really changes.

// EIGHT: "arg1": [
// EIGHT-NEXT: "i32:1",
// EIGHT-NEXT: "i32:2",
// EIGHT-NEXT: "i32:3",
// EIGHT-NEXT: "i32:4",
// EIGHT-NEXT: "i32:5",
// EIGHT-NEXT: "i32:6",
// EIGHT-NEXT: "i32:7",
// EIGHT-NEXT: "i32:8"
// EIGHT: "dataflow.load": 8
// EIGHT: "dataflow.store": 8

// The expansion materializes its own loop bounds, so it may only do so when the
// index width resolved at that graph can represent them. Eight elements are not
// expressible in a four bit signed stream, and a width with no fixed
// representation is not a width at all: the expansion resolves it before
// expanding that graph's copies, so the checked error is reported by the
// expansion rather than caught later by the graph-memory owner. Each failure
// leaves its live module exactly as written, with no graph published.

// REJECT: error: 'memref.copy' op loom-expand-graph-memref-copy: cannot expand memref.copy into a structured load/store loop; bound 8 is not representable in the graph's resolved signed index domain 'i4'
// REJECT: dataflow.thread private @too_wide
// REJECT: loom.spatial_region
// REJECT: memref.copy %{{.*}}, %{{.*}} : memref<8xi32> to memref<8xi32>

// REJECT: error: loom-expand-graph-memref-copy: index bit width must be nonzero
// REJECT: dataflow.thread private @unusable_width
// REJECT: loom.spatial_region
// REJECT: memref.copy %{{.*}}, %{{.*}} : memref<4xi32> to memref<4xi32>

//--- layout.mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 4>>
} {
  dataflow.thread private @spatial_copy(
      %src: memref<7xi32>, %dst: memref<7xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<7xi32>, %target: memref<7xi32>):
        memref.copy %source, %target : memref<7xi32> to memref<7xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "copy_graph", source_maps = []} :
        (memref<7xi32>, memref<7xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @shared_constants(
      %loop_src: memref<7xi32>, %loop_dst: memref<7xi32>,
      %copy_src: memref<7xi32>, %copy_dst: memref<7xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%loop_src, %loop_dst, %copy_src, %copy_dst)
        <{operandSegmentSizes = array<i32: 0, 0, 4, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%from: memref<7xi32>, %to: memref<7xi32>,
           %source: memref<7xi32>, %target: memref<7xi32>):
        %zero = arith.constant 0 : index
        %seven = arith.constant 7 : index
        %one = arith.constant 1 : index
        scf.for %index = %zero to %seven step %one {
          %element = memref.load %from[%index] : memref<7xi32>
          memref.store %element, %to[%index] : memref<7xi32>
        }
        memref.copy %source, %target : memref<7xi32> to memref<7xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "shared_constants_graph", source_maps = []} :
        (memref<7xi32>, memref<7xi32>, memref<7xi32>, memref<7xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @unit_copy(
      %src: memref<1xi32>, %dst: memref<1xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<1xi32>, %target: memref<1xi32>):
        memref.copy %source, %target : memref<1xi32> to memref<1xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "unit_graph", source_maps = []} :
        (memref<1xi32>, memref<1xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @empty_copy(
      %src: memref<0xi32>, %dst: memref<0xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<0xi32>, %target: memref<0xi32>):
        memref.copy %source, %target : memref<0xi32> to memref<0xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "empty_graph", source_maps = []} :
        (memref<0xi32>, memref<0xi32>) -> ()
    dataflow.thread.yield
  }
}

//--- override.mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @spatial_copy(
      %src: memref<8xi32>, %dst: memref<8xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<8xi32>, %target: memref<8xi32>):
        memref.copy %source, %target : memref<8xi32> to memref<8xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "override_graph", source_maps = []} :
        (memref<8xi32>, memref<8xi32>) -> ()
    dataflow.thread.yield
  }
}

//--- unusable.mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 4>>
} {
  dataflow.thread private @too_wide(
      %src: memref<8xi32>, %dst: memref<8xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<8xi32>, %target: memref<8xi32>):
        memref.copy %source, %target : memref<8xi32> to memref<8xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "too_wide_graph", source_maps = []} :
        (memref<8xi32>, memref<8xi32>) -> ()
    dataflow.thread.yield
  }
}

// -----

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 0>>
} {
  dataflow.thread private @unusable_width(
      %src: memref<4xi32>, %dst: memref<4xi32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%src, %dst)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%source: memref<4xi32>, %target: memref<4xi32>):
        memref.copy %source, %target : memref<4xi32> to memref<4xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "unusable_graph", source_maps = []} :
        (memref<4xi32>, memref<4xi32>) -> ()
    dataflow.thread.yield
  }
}
