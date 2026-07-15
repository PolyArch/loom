// RUN: loom-synth-base-test --make mcs | FileCheck %s --check-prefix=MCS
// RUN: loom-synth-base-test --make incremental | FileCheck %s --check-prefix=INC
// RUN: loom-synth-base-test --make incremental_random | FileCheck %s --check-prefix=RAND

// MCS: factory: nullptr
// INC: factory: nullptr
// RAND: factory: nullptr
