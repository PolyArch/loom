// RUN: loom-hwsg-test find llvm.arm.qadd16 -- find llvm.arm.sadd16 -- find llvm.arm.qsub16 -- find llvm.arm.qsub8 | FileCheck %s --check-prefix=SAT
// RUN: loom-hwsg-test find llvm.trunc -- find llvm.sext -- find llvm.zext | FileCheck %s --check-prefix=CAST

// SAT: find llvm.arm.qadd16=[[SAT_GROUP:[0-9]+]]
// SAT-NEXT: find llvm.arm.sadd16=[[SAT_GROUP]]
// SAT-NEXT: find llvm.arm.qsub16=[[SAT_GROUP]]
// SAT-NEXT: find llvm.arm.qsub8=[[SAT_GROUP]]

// CAST: find llvm.trunc=[[CAST_GROUP:[0-9]+]]
// CAST-NEXT: find llvm.sext=[[CAST_GROUP]]
// CAST-NEXT: find llvm.zext=[[CAST_GROUP]]
