// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s --check-prefix=STRUCT
// RUN: loom-raise-opt --loom-lower-for-to-graph --mlir-pass-statistics \
// RUN:   --mlir-pass-statistics-display=list %s 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=WORK

// STRUCT-LABEL: dataflow.thread private @parallel_groups domain(#dataflow.thread_domain<dense>)
// STRUCT-COUNT-2: dataflow.sync
// STRUCT: dataflow.thread.yield

// WORK: LowerForToGraphPass
// WORK: (S) 2 parallel-completion-candidate-inspections

dataflow.thread private @parallel_groups domain(#dataflow.thread_domain<dense>)() ctrl (%start: none) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    "loom.spatial_region"()
        <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0:
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "parallel_group_a", source_maps = []} : () -> ()
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    "loom.spatial_region"()
        <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0:
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "parallel_group_b", source_maps = []} : () -> ()
    scf.reduce
  }
  dataflow.thread.yield
}
