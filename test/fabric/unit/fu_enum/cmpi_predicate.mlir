// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU containing a single arith.cmpi fabric.op whose hardware supports three
// predicate values. The enumerator should produce three dataflow.subgraphs,
// one per predicate, each with a properly typed CmpIPredicateAttr.

// CHECK-LABEL: @fu_cmpi
func.func @fu_cmpi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<1> {
    %k = fabric.op [@arith.cmpi] (%x, %y)
         {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }

  // CHECK-DAG: "op#0{predicate=eq}"
  // CHECK-DAG: "op#0{predicate=slt}"
  // CHECK-DAG: "op#0{predicate=sgt}"

  // CHECK-DAG: arith.cmpi eq, %{{.*}}, %{{.*}} : i32
  // CHECK-DAG: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
  // CHECK-DAG: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32

  return
}
