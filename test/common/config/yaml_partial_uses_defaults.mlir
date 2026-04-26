// RUN: loom-config-test %p/yaml_partial_uses_defaults.yaml | FileCheck %s

// Only `algorithm` is set; remaining keys must keep documented defaults.

// CHECK: algorithm=list
// CHECK-NEXT: alpha=1.000000e+00
// CHECK-NEXT: beta=1.000000e+00
// CHECK-NEXT: gamma=5.000000e-01
// CHECK-NEXT: beam_width=4
// CHECK-NEXT: sa_steps=1000
// CHECK-NEXT: sa_seed=49374
// CHECK-NEXT: threads=0
