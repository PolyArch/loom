// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph pkhbt_graph --hardware-mlir %s --hardware arm_inline_asm_adg --workload pkhbt_graph --output %t.csv --artifact %t.json
// RUN: FileCheck %s < %t.csv

// CHECK: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CHECK-NEXT: pkhbt_graph,arm_inline_asm_adg,pkhbt_graph__pkhbt_graph__arm_inline_asm_adg,2,1,0,0,pass

module {
  dataflow.graph.func private @pkhbt_graph(%ctrl: none, %lhs: i32, %rhs: i32,
                                           %amount: i32) -> (none, i32) {
    %result = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "pkhbt $0, $1, $2, lsl $3", "=r,r,r,I" %lhs, %rhs, %amount
        : (i32, i32, i32) -> i32
    dataflow.graph.return %ctrl, %result : none, i32
  }

  fabric.module @arm_inline_asm_adg(%ctrl : !fabric.bits<0>,
                                    %a : !fabric.bits<32>,
                                    %b : !fabric.bits<32>,
                                    %c : !fabric.bits<32>) {
    %result = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                                   %pb = %b : !fabric.bits<32>,
                                   %pc = %c : !fabric.bits<32>,
                                   %pd = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %pa : !fabric.bits<32>,
                %fb = %pb : !fabric.bits<32>,
                %fc = %pc : !fabric.bits<32>,
                %token = %pd : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@llvm.arm.pkhbt] (%fa, %fb, %fc)
            : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
            -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
  }
}
