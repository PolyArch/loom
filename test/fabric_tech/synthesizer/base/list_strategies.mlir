// RUN: loom-synth-base-test --list-strategies | FileCheck %s

// The factory dispatches on `SynthConfig.strategy`. The four canonical
// names are documented in `docs/spec-generalize-subgraphs-to-fu.md`'s
// "Configuration Surface" section and listed by the helper in spec order.

// CHECK: anchor
// CHECK-NEXT: mcs
// CHECK-NEXT: incremental
// CHECK-NEXT: incremental_random
