// RUN: loom-hwsg-test find arith.addi -- find arith.subi | FileCheck %s

// Both arith.addi and arith.subi belong to the same multi-member group, so
// findShareGroup must return the same numeric index for both.

// CHECK: find arith.addi=[[GRP:[0-9]+]]
// CHECK-NEXT: find arith.subi=[[GRP]]
