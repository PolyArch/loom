// RUN: loom-synth-base-test --make incremental | FileCheck %s

// `--make incremental` runs the stub Synthesizer on an empty
// SynthInputs (no input subgraphs). The default-constructed
// CoverageReport has an empty `matchIndex`, so `allCovered()` returns
// `true` (vacuous coverage) -- the helper does not surface that field
// directly, but the stub failure path proves the SynthResult round-trip
// works without ever touching the coverage report. The chosen semantic
// for empty `matchIndex` is documented on `CoverageReport::allCovered`:
// zero inputs are trivially covered.

// CHECK: result: success=false reason=topology_mismatch
// CHECK-NEXT: note: strategy incremental not yet implemented
