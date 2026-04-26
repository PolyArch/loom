// RUN: loom-config-test %p/yaml_full.yaml | FileCheck %s

// All keys explicitly set; defaults are overridden.

// CHECK: algorithm=beam
// CHECK-NEXT: alpha=2.000000e+00
// CHECK-NEXT: beta=5.000000e-01
// CHECK-NEXT: gamma=2.500000e-01
// CHECK-NEXT: beam_width=8
// CHECK-NEXT: sa_steps=500
// CHECK-NEXT: sa_seed=42
// CHECK-NEXT: threads=4
