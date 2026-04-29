// RUN: loom-hwsg-test same arith.muli arith.muli | FileCheck %s

// A singleton trivially shares with itself.

// CHECK: same arith.muli arith.muli=true
