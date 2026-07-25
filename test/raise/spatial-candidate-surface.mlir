// RUN: split-file %s %t
// RUN: loom-raise-opt --loom-llvm-cf-to-cf --loom-lift-cf-to-scf %t/unselected.mlir | FileCheck %s --check-prefix=UNSELECTED
// RUN: not loom-raise-opt --loom-lower-for-to-graph %t/weighted-selected.mlir 2>&1 | FileCheck %s --check-prefix=WEIGHTED --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-for-to-graph %t/scalable-selected.mlir 2>&1 | FileCheck %s --check-prefix=SCALABLE --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// Rejection belongs to the candidate that selects a SpatialCore boundary, not
// to ordinary module compilation. The same construct that is legal S0 while
// unselected cannot be presented as the contents of a selected
// `loom.spatial_region`.

// No structured operation states a branch probability, so a weighted branch
// keeps the control it came in with and its imported profile stays where it
// was. Preservation is a disposition of that one callable: the structurable
// callable beside it is still recovered, an unweighted switch beside a
// weighted one still respells, and the pass succeeds.
// UNSELECTED-LABEL: llvm.func @structured_sibling
// UNSELECTED: scf.if
// UNSELECTED-NOT: cf.cond_br
// UNSELECTED-LABEL: llvm.func @weighted_branch
// UNSELECTED: cf.cond_br %arg0 weights([1, 9])
// UNSELECTED-NOT: scf.if
// UNSELECTED-LABEL: llvm.func @plain_switch
// UNSELECTED: scf.index_switch
// UNSELECTED-LABEL: llvm.func @weighted_switch
// UNSELECTED: llvm.switch
// UNSELECTED: branch_weights = array<i32: 1, 9>
// UNSELECTED-NOT: scf.index_switch

// A selected SpatialCore boundary holds one structured block, and the weighted
// CFG can never be structured without losing its weights. The candidate is
// rejected whole and no graph or launch is published.
// WEIGHTED: 'loom.spatial_region' op expects region #0 to have 0 or 1 blocks
// WEIGHTED: cf.cond_br
// WEIGHTED-SAME: branch_weights = array<i32: 1, 9>

// A scalable vector is a legal S0 value, but its element count is a runtime
// `vscale` multiple that the fixed-rank Canonical Dataflow contract has no
// meaning for. A typed structured transform must materialize it as
// fixed-width chunks, loops, and masks or tails first; until then the
// candidate that selected this region cannot finalize.
// SCALABLE: 'loom.spatial_region' op holds scalable vector type 'vector<[4]xf32>'
// SCALABLE-SAME: must be materialized as fixed-width chunks, loops, and masks or tails

//--- unselected.mlir
llvm.func @structured_sibling(%c: i1, %p: !llvm.ptr) {
  %z = llvm.mlir.constant(0 : i32) : i32
  llvm.cond_br %c, ^yes, ^exit
^yes:
  llvm.store %z, %p : i32, !llvm.ptr
  llvm.br ^exit
^exit:
  llvm.return
}

llvm.func @weighted_branch(%c: i1) -> i32 {
  %z = llvm.mlir.constant(0 : i32) : i32
  llvm.cond_br %c weights([1, 9]), ^yes, ^no
^yes:
  llvm.return %z : i32
^no:
  llvm.return %z : i32
}

llvm.func @plain_switch(%v: i32) -> i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.switch %v : i32, ^bb_default [
    0: ^bb0
  ]
^bb_default:
  llvm.return %c0 : i32
^bb0:
  llvm.return %v : i32
}

llvm.func @weighted_switch(%v: i32) -> i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.switch %v : i32, ^bb_default [
    0: ^bb0
  ] {branch_weights = array<i32: 1, 9>}
^bb_default:
  llvm.return %c0 : i32
^bb0:
  llvm.return %v : i32
}

//--- weighted-selected.mlir
dataflow.thread private @selected_weighted(%c: i1, %v: i32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%c, %v)
      <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%cond: i1, %value: i32):
      cf.cond_br %cond weights([1, 9]), ^yes, ^no
    ^yes:
      "loom.spatial_yield"(%value)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
    ^no:
      "loom.spatial_yield"(%value)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
  }) {graph_name = "selected_weighted_graph", source_maps = []} :
      (i1, i32) -> i32
  dataflow.thread.yield
}

//--- scalable-selected.mlir
dataflow.thread private @selected_scalable(%v: vector<[4]xf32>)
    ctrl (%start: none) {
  %result = "loom.spatial_region"(%v)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%lane: vector<[4]xf32>):
      %sum = arith.addf %lane, %lane : vector<[4]xf32>
      "loom.spatial_yield"(%sum)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (vector<[4]xf32>) -> ()
  }) {graph_name = "selected_scalable_graph", source_maps = []} :
      (vector<[4]xf32>) -> vector<[4]xf32>
  dataflow.thread.yield
}
