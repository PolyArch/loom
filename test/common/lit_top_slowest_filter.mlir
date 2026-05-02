// RUN: sed -n 's#^// IN: *##p' %s | %python %S/../lit_top_slowest.py | FileCheck %s

// CHECK: -- Testing: 6 tests, 1 workers --
// CHECK: Testing:  0.. 10.. 20..
// CHECK: FAIL: LOOM :: example/fail.mlir
// CHECK: diagnostic: expected failure text
// CHECK: Slowest 5 Tests:
// CHECK: 9.00s: LOOM :: slow_0.mlir
// CHECK: 8.00s: LOOM :: slow_1.mlir
// CHECK: 7.00s: LOOM :: slow_2.mlir
// CHECK: 6.00s: LOOM :: slow_3.mlir
// CHECK: 5.00s: LOOM :: slow_4.mlir
// CHECK-NOT: 4.00s: LOOM :: slow_5.mlir
// CHECK-NOT: Tests Times:
// CHECK-NOT: histogram bucket
// CHECK: Failed Tests (1):

// IN: -- Testing: 6 tests, 1 workers --
// IN: Testing:  0.. 10.. 20..
// IN: FAIL: LOOM :: example/fail.mlir
// IN: diagnostic: expected failure text
// IN: Slowest Tests:
// IN: --------------------------------------------------------------------------
// IN: 9.00s: LOOM :: slow_0.mlir
// IN: 8.00s: LOOM :: slow_1.mlir
// IN: 7.00s: LOOM :: slow_2.mlir
// IN: 6.00s: LOOM :: slow_3.mlir
// IN: 5.00s: LOOM :: slow_4.mlir
// IN: 4.00s: LOOM :: slow_5.mlir
// IN: --------------------------------------------------------------------------
// IN:
// IN: Tests Times:
// IN: --------------------------------------------------------------------------
// IN: histogram bucket
// IN:
// IN: Failed Tests (1):
