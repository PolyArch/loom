// RUN: loom-synth-base-test --make anchor | FileCheck %s

// The factory recognises `anchor` (one of the four canonical strategy
// names). The real anchor strategy short-circuits an empty input
// subgraph list to `invalid_input` (the helper passes no subgraphs);
// stub strategies (mcs / incremental / incremental_random) still
// report `topology_mismatch` until their respective tasks land.

// CHECK: result: success=false reason=invalid_input
// CHECK-NEXT: note: anchor: no input subgraphs in synth group
