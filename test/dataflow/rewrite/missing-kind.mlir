// The typed rewrite kind is the whole decision the pass makes, so it must be
// stated explicitly. Silently defaulting to one catalog entry would let a
// caller apply a rewrite it never selected.

// RUN: not loom-raise-opt --dataflow-rewrite %s -o %t.missing.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-KIND
// RUN: test ! -s %t.missing.mlir

// An explicitly selected kind is accepted and leaves a non-matching program
// exactly as it was.
// RUN: loom-raise-opt --dataflow-rewrite=kind=pack-unpack-round-trip-eliminate %s -o %t.explicit.mlir
// RUN: FileCheck %s --check-prefix=EXPLICIT < %t.explicit.mlir

// MISSING-KIND: error: dataflow-rewrite requires an explicit 'kind' option

// EXPLICIT-LABEL: dataflow.graph private @no_matching_rewrite
// EXPLICIT: %[[VALUE:[^ ]*]] = dataflow.constant %[[CTRL:[^ ]*]] {const_value = 7 : i32} : i32
// EXPLICIT: dataflow.sync %[[CTRL]], %[[VALUE]] :

module {
  dataflow.graph private @no_matching_rewrite(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    %retired:2 = dataflow.sync %start, %value : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
}
