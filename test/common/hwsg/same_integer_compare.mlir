// RUN: loom-hwsg-test same arith.cmpi llvm.icmp | FileCheck %s

// Integer arith and LLVM comparisons use one predicate-controlled comparator
// family. Exact mode types and attributes remain independently verified.

// CHECK: same arith.cmpi llvm.icmp=true
