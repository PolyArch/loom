// Documentation-only test: the B2 (inner fabric.fu fails verification)
// failure path of the tightened idempotence precheck in
// `loom-generalize-subgraphs-to-fu` is not constructible as plain MLIR
// at the lit level. The MLIR parser runs verification during parsing
// and rejects every fabric.fu / fabric.op / fabric.yield malformedness
// the check could observe (cross-share-group op_list, body-without-
// fabric.op, fabric.fu operand/block-arg type mismatch, fabric.yield
// arity / type mismatch); a B2 failure can therefore only arise when a
// previously verified wrapper is mutated post-parse (e.g. by a buggy
// upstream pass), which a single-input lit test cannot synthesize
// without also bypassing parser verification.
//
// B1 (`idempotent_resynth_b1_fail.mlir`) and B3
// (`idempotent_resynth_b3_signature_mismatch.mlir`) cover the failure
// paths that ARE constructible. The B2 logic itself reuses the
// `mlir::verify` machinery exercised by every other FU-emitting test
// in the synthesizer suite (and the C++ implementation captures the
// diagnostic via a `ScopedDiagnosticHandler`; see
// `validateMarkerWrapper` in
// `lib/Fabric/Tech/Synthesizer/GeneralizeSubgraphsToFuPass.cpp`).

// RUN: echo "B2-not-constructible" | FileCheck %s
// CHECK: B2-not-constructible
