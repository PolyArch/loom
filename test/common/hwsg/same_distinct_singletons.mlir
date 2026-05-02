// RUN: loom-hwsg-test same arith.muli arith.cmpi | FileCheck %s

// Two distinct true singletons (neither arith.muli nor arith.cmpi appears in
// any multi-member group) must not be considered to share hardware. This
// exercises the !ga && !gb branch of sameShareGroup with differing names.

// CHECK: same arith.muli arith.cmpi=false
