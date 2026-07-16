// RUN: loom-synth-base-test --synthesize-fixed-vector | FileCheck %s

// CHECK: synthesis=success
// CHECK-NEXT: input_width=128
// CHECK-NEXT: encodings=1
// CHECK-NEXT: covered=2
