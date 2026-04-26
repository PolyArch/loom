// RUN: loom-config-test %p/yaml_full_ilp.yaml | FileCheck %s

// The YAML loader accepts "ilp" as a valid algorithm name.

// CHECK: algorithm=ilp
// CHECK-NEXT: alpha=1.000000e+00
// CHECK-NEXT: beta=1.000000e+00
// CHECK-NEXT: gamma=5.000000e-01
// CHECK-NEXT: beam_width=4
// CHECK-NEXT: sa_steps=1000
// CHECK-NEXT: sa_seed=49374
// CHECK-NEXT: threads=1
