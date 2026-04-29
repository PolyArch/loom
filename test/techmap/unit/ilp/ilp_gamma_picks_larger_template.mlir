// The cost-model gamma term rewards bound blocks whose ops fully utilize
// the largest available template for that root. The MIP encodes this as a
// per-block density-deficit penalty (1 - K_t/M_t) scaled by gamma. With
// alpha = 0 the |blocks_with_template| term is neutralized, leaving gamma
// as the dominant tie-breaker between the 1-op covering (where the addi
// singleton has deficit 0.5) and the 2-op covering (deficit 0). With
// gamma = 0 the two partitions tie; with gamma = 10 the 2-op covering
// strictly dominates.

// RUN: echo "techmap:" > %t.g0.yaml
// RUN: echo "  algorithm: ilp" >> %t.g0.yaml
// RUN: echo "  alpha: 0" >> %t.g0.yaml
// RUN: echo "  beta: 0" >> %t.g0.yaml
// RUN: echo "  gamma: 0" >> %t.g0.yaml
// RUN: echo "techmap:" > %t.g10.yaml
// RUN: echo "  algorithm: ilp" >> %t.g10.yaml
// RUN: echo "  alpha: 0" >> %t.g10.yaml
// RUN: echo "  beta: 0" >> %t.g10.yaml
// RUN: echo "  gamma: 10" >> %t.g10.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.g0.yaml" \
// RUN:   | FileCheck --check-prefix=G0 %s
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.g10.yaml" \
// RUN:   | FileCheck --check-prefix=G10 %s

fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @fu_muli_addi(%cast0_fu_muli_addi : !fabric.bits<32>, %cast1_fu_muli_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli_addi : !fabric.bits<32>, %b = %cast1_fu_muli_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.addi] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  }
  fabric.yield
}


// With gamma = 0 the per-block deficit penalty vanishes; ILP returns the
// two-singleton partition that HiGHS produces under tie-breaking.
// G0-LABEL: @graph_chain
// G0: dataflow.subgraph
// G0-NEXT: arith.muli
// G0-NEXT: dataflow.yield
// G0: dataflow.subgraph
// G0-NEXT: arith.addi
// G0-NEXT: dataflow.yield

// With gamma = 10 the addi singleton's 0.5 deficit (M_addi = 2) costs 5;
// the 2-op binding (deficit 0 for both blocks involved) wins.
// G10-LABEL: @graph_chain
// G10: dataflow.subgraph
// G10-NEXT: arith.muli
// G10-NEXT: arith.addi
// G10-NEXT: dataflow.yield
// G10-NOT: dataflow.subgraph
func.func @graph_chain(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
