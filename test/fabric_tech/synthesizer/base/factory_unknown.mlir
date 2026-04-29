// RUN: loom-synth-base-test --make wibble | FileCheck %s

// Unknown strategy names produce a null Synthesizer from the factory;
// the helper prints `factory: nullptr`. The pass propagates this as
// `invalid_input` on the input function.

// CHECK: factory: nullptr
