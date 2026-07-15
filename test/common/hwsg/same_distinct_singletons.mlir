// RUN: loom-hwsg-test same arith.muli math.absf | FileCheck %s

// Two distinct true singletons (neither arith.muli nor math.absf appears in
// any multi-member group) must not be considered to share hardware. This
// exercises the !ga && !gb branch of sameShareGroup with differing names.

// CHECK: same arith.muli math.absf=false
