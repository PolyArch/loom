// RUN: loom-synth-base-test --make incremental | FileCheck %s

// `--make incremental` runs the IncrementalSynthesizer on an empty
// SynthInputs (no input subgraphs). The strategy short-circuits to
// `invalid_input` because every left-fold needs at least one input to
// seed the trivial FU. The default-constructed CoverageReport is never
// touched by this path; this test pins the contract that an empty
// input is rejected at the strategy boundary (not inside the verifier),
// which preserves the documented vacuous-coverage semantics on
// `CoverageReport::allCovered` (true for an empty matchIndex).

// CHECK: result: success=false reason=invalid_input
// CHECK-NEXT: note: incremental: no input subgraphs in synth group
