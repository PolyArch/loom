// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Combined FU: a multi-member arith group fed by a 2-input mux, with a
// downstream cmpi whose hardware-supported predicates are a 3-set. The
// total support set is 2 (mux.sel) x 2 (op_sel) x 3 (predicate) = 12.
//
// To satisfy the pe uniform-W rule the FU exposes bits<1>
// throughout (cmpi's TypeParam(0) inputs accept any width and its
// Fixed(1) output already matches the boundary).

// CHECK-LABEL: fabric.module @fu_combined
fabric.module @fu_combined(%a : !fabric.bits<1>, %b : !fabric.bits<1>, %c : !fabric.bits<1>, %d : !fabric.bits<1>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>,
                    %pc = %c : !fabric.bits<1>,
                    %pd = %d : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%w = %pa : !fabric.bits<1>,
              %x = %pb : !fabric.bits<1>,
              %y = %pc : !fabric.bits<1>,
              %z = %pd : !fabric.bits<1>) -> !fabric.bits<1> {
      %m = fabric.mux %w, %x : !fabric.bits<1>
      %s = fabric.op [@arith.addi, @arith.subi] (%m, %y)
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      %k = fabric.op [@arith.cmpi] (%s, %z)
           {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}

// After dedup (mux.sel=0 and sel=1 produce graph-isomorphic subgraphs
// -- the only difference is which FU input port feeds the addi/subi),
// the cardinality drops from 12 to 6: 2 op_sel x 3 predicate, each
// anchored on the lex-smallest mux.sel=0.
// CHECK-DAG: "mux#0{sel=0,discard=false,disconnect=false}; op#0{op_sel=arith.addi}; op#1{predicate=eq}"
// CHECK-DAG: "mux#0{sel=0,discard=false,disconnect=false}; op#0{op_sel=arith.subi}; op#1{predicate=sgt}"
// CHECK-DAG: "mux#0{sel=0,discard=false,disconnect=false}; op#0{op_sel=arith.subi}; op#1{predicate=slt}"

// Body of one specific config (e.g. addi, slt):
// CHECK-DAG: arith.cmpi slt, %{{.*}}, %{{.*}} : i1
