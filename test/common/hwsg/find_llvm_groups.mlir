// RUN: loom-hwsg-test find llvm.arm.qadd16 -- find llvm.arm.qsub16 | FileCheck %s --check-prefix=SAT16
// RUN: loom-hwsg-test find llvm.trunc -- find llvm.sext -- find llvm.zext | FileCheck %s --check-prefix=CAST

// SAT16: find llvm.arm.qadd16=[[SAT:[0-9]+]]
// SAT16-NEXT: find llvm.arm.qsub16=[[SAT]]

// CAST: find llvm.trunc=[[CAST_GROUP:[0-9]+]]
// CAST-NEXT: find llvm.sext=[[CAST_GROUP]]
// CAST-NEXT: find llvm.zext=[[CAST_GROUP]]
