// RUN: loom-hwsg-test same arith.addi arith.divsi | FileCheck %s

// Two members of distinct multi-member groups (arith.addi in {arith.addi,
// arith.subi} and arith.divsi in {arith.divsi, arith.remsi}) must not be
// considered to share hardware. This exercises the *ga != *gb branch of
// sameShareGroup.

// CHECK: same arith.addi arith.divsi=false
