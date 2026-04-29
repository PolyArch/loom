// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU offers cmpi with a 3-predicate hardware support set. Patterns ask
// for predicates within and outside the support set. To satisfy the
// spatial_pe uniform-W rule we expose the FU at bits<1> throughout
// (cmpi's TypeParam(0) inputs accept any width).

fabric.module @hw_cmpi(%a : !fabric.bits<1>, %b : !fabric.bits<1>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %pa : !fabric.bits<1>, %y = %pb : !fabric.bits<1>)
                  -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_cmpi_eq
func.func @pat_cmpi_eq(%x: i1, %y: i1) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{predicate=eq}"
  %r = dataflow.subgraph(%a = %x : i1, %b = %y : i1) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi eq, %a, %b : i1
    dataflow.yield %k : i1
  }
  return %r : i1
}

// CHECK-LABEL: @pat_cmpi_slt
func.func @pat_cmpi_slt(%x: i1, %y: i1) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{predicate=slt}"
  %r = dataflow.subgraph(%a = %x : i1, %b = %y : i1) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi slt, %a, %b : i1
    dataflow.yield %k : i1
  }
  return %r : i1
}

// "ne" is NOT in the hw_params predicate set; should not match.
// CHECK-LABEL: @pat_cmpi_ne_unmatched
func.func @pat_cmpi_ne_unmatched(%x: i1, %y: i1) -> i1 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i1, %b = %y : i1) -> i1
       attributes {loom.is_pattern} {
    %k = arith.cmpi ne, %a, %b : i1
    dataflow.yield %k : i1
  }
  return %r : i1
}
