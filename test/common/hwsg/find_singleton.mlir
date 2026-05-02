// RUN: loom-hwsg-test find arith.muli | FileCheck %s

// arith.muli is not in any multi-member group; findShareGroup must report
// it as a singleton (std::nullopt -> "none").

// CHECK: find arith.muli=none
