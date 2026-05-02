// RUN: loom-hwsg-test same arith.addi arith.subi | FileCheck %s

// arith.addi and arith.subi share the same multi-member hardware group.

// CHECK: same arith.addi arith.subi=true
