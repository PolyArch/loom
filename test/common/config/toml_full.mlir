// RUN: loom-config-test %p/toml_full.toml | FileCheck %s

// TOML mirror of yaml_full.yaml with the same key set; output must match.

// CHECK: algorithm=sa
// CHECK-NEXT: alpha=1.500000e+00
// CHECK-NEXT: beta=0.000000e+00
// CHECK-NEXT: gamma=5.000000e-01
// CHECK-NEXT: beam_width=16
// CHECK-NEXT: sa_steps=2000
// CHECK-NEXT: sa_seed=7
// CHECK-NEXT: threads=0
