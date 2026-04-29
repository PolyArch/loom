// RUN: loom-hwsg-test same arith.muli math.sin | FileCheck %s

// Two distinct singletons (note: math.sin is in a multi-member group, but
// arith.muli is not) must not be considered to share hardware. Even when
// both names happen to be implicit singletons, distinct names never share.

// CHECK: same arith.muli math.sin=false
