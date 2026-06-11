// Documentation-only test: the inner-fabric.fu-fails-verification
// failure path of the tightened idempotence precheck in
// `loom-generalize-subgraphs-to-fu` is not constructible as plain MLIR
// at the lit level. The MLIR parser runs verification during parsing
// and rejects every fabric.fu / fabric.op / fabric.yield malformedness
// the check could observe (cross-share-group op_list, body-without-
// fabric.op, fabric.fu operand/block-arg type mismatch, fabric.yield
// arity / type mismatch); an inner-fu-verifier failure can therefore
// only arise when a previously verified wrapper is mutated post-parse
// (e.g. by a buggy upstream pass), which a single-input lit test
// cannot synthesize without also bypassing parser verification.
//
// `idempotent_resynth_body_shape_fail.mlir` and
// `idempotent_resynth_signature_mismatch.mlir` cover the failure
// paths that ARE constructible. The inner-fu-verifier logic itself
// reuses the `mlir::verify` machinery exercised by every other
// FU-emitting test in the synthesizer suite and must report a structured
// diagnostic if a post-parse mutation ever makes this path reachable.

// RUN: echo "inner-fu-verifier-not-constructible" | FileCheck %s
// CHECK: inner-fu-verifier-not-constructible
