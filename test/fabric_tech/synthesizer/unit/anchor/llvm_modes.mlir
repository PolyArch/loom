// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// CHECK: remark: {{.*}}synth-stat group=llvm_icmp strategy=anchor reason=success
// CHECK-SAME: covered=1/1 nodes=1/0/0 encodings=1
// CHECK: remark: {{.*}}synth-stat group=llvm_trunc strategy=anchor reason=success
// CHECK-SAME: covered=1/1 nodes=1/0/0 encodings=1
// CHECK: fabric.module @fu_llvm_icmp
// CHECK: fabric.op [@llvm.icmp]
// CHECK: fabric.module @fu_llvm_trunc
// CHECK: fabric.op [@llvm.trunc]

func.func @pat_llvm_trunc(%value: i64) -> i32
    attributes {loom.synth_group = "llvm_trunc"} {
  %narrow = llvm.trunc %value : i64 to i32
  return %narrow : i32
}

func.func @pat_llvm_icmp(%lhs: i32, %rhs: i32) -> i1
    attributes {loom.synth_group = "llvm_icmp"} {
  %predicate = llvm.icmp "slt" %lhs, %rhs : i32
  return %predicate : i1
}
