// RUN: loom-synth-base-test --make anchor | FileCheck %s

// The factory recognises `anchor` (one of the four canonical strategy
// names). Until the real strategy lands, the dispatch returns a stub
// that immediately reports a TopologyMismatch failure with a single
// note explaining the strategy is unimplemented.

// CHECK: result: success=false reason=topology_mismatch
// CHECK-NEXT: note: strategy anchor not yet implemented
