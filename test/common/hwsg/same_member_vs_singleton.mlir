// RUN: loom-hwsg-test same arith.addi arith.muli | FileCheck %s

// A multi-member group entry never shares hardware with a singleton.

// CHECK: same arith.addi arith.muli=false
