// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Combined FU: a multi-member arith group fed by a 2-input mux, with a
// downstream cmpi whose hardware-supported predicates are a 3-set. The
// total support set is 2 (mux.sel) x 2 (op_sel) x 3 (predicate) = 12.

// CHECK-LABEL: @fu_combined
func.func @fu_combined(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                       %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<1> {
    %m = fabric.mux %w, %x : !fabric.bits<32>
    %s = fabric.op [@arith.addi, @arith.subi] (%m, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %k = fabric.op [@arith.cmpi] (%s, %z)
         {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }

  // 12 enumerated subgraphs total. Spot-check three corners.
  // CHECK-DAG: "mux#0{sel=0,discard=false,disconnect=false}; op#0{op_sel=arith.addi}; op#1{predicate=eq}"
  // CHECK-DAG: "mux#0{sel=1,discard=false,disconnect=false}; op#0{op_sel=arith.subi}; op#1{predicate=sgt}"
  // CHECK-DAG: "mux#0{sel=0,discard=false,disconnect=false}; op#0{op_sel=arith.subi}; op#1{predicate=slt}"

  // Body of one specific config (mux.sel=1, addi, slt):
  // CHECK-DAG: arith.cmpi slt, %{{.*}}, %{{.*}} : i32

  return
}
