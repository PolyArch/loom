// RUN: echo "techmap:" > %t.greedy.yaml
// RUN: echo "  algorithm: greedy" >> %t.greedy.yaml
// RUN: echo "techmap:" > %t.list.yaml
// RUN: echo "  algorithm: list" >> %t.list.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.greedy.yaml" > %t.greedy.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.list.yaml" > %t.list.mlir
// RUN: grep -c "dataflow.subgraph" %t.greedy.mlir > %t.greedy.count
// RUN: grep -c "dataflow.subgraph" %t.list.mlir > %t.list.count
// Greedy and list must agree on subgraph count for this single-op-template
// library (no template fuses across ops, so any topo-respecting order
// emits the same number of singleton blocks).
// RUN: diff %t.greedy.count %t.list.count
// RUN: FileCheck %s < %t.greedy.mlir

// Stress: 30-op dataflow.graph mixing arith.{addi, subi, muli, andi, ori,
// xori, cmpi} and math.{sin, cos, sqrt}. The FU library covers each kind
// with a single-op template (logic ops share one FU via the
// {andi, ori, xori} hardware-share group; trig shares the {sin, cos}
// group). Every op is supported, so every op is wrapped in a singleton
// dataflow.subgraph. Total: 30 subgraphs.

// CHECK-LABEL: @fu_addi
fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_subi
fabric.module @fu_subi(%cast0_fu_subi : !fabric.bits<32>, %cast1_fu_subi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_subi : !fabric.bits<32>, %b = %cast1_fu_subi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_muli
fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_logic
fabric.module @fu_logic(%cast0_fu_logic : !fabric.bits<32>, %cast1_fu_logic : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_logic : !fabric.bits<32>, %b = %cast1_fu_logic : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.andi, @arith.ori, @arith.xori] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_cmpi
fabric.module @fu_cmpi(%cast0_fu_cmpi : !fabric.bits<1>, %cast1_fu_cmpi : !fabric.bits<1>) {
  fabric.pe [spatial] (%a = %cast0_fu_cmpi : !fabric.bits<1>, %b = %cast1_fu_cmpi : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %a : !fabric.bits<1>, %y = %b : !fabric.bits<1>)
                  -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_trig
fabric.module @fu_trig(%cast0_fu_trig : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_trig : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    %k = fabric.op [@math.sin, @math.cos] (%x)
         : (!fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}

// CHECK-LABEL: @fu_sqrt
fabric.module @fu_sqrt(%cast0_fu_sqrt : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_sqrt : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    %k = fabric.op [@math.sqrt] (%x)
         : (!fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// Spot-check that representative subgraphs of every kind appear. A
// stricter ordered count is enforced by the diff-against-list run line.
// CHECK-LABEL: @graph_thirty
// CHECK: dataflow.graph
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.subi
// CHECK-DAG: arith.muli
// CHECK-DAG: arith.andi
// CHECK-DAG: arith.ori
// CHECK-DAG: arith.xori
// CHECK-DAG: arith.cmpi eq
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi sgt
// CHECK-DAG: math.sin
// CHECK-DAG: math.cos
// CHECK-DAG: math.sqrt
// CHECK: dataflow.yield
func.func @graph_thirty(%a: i32, %b: i32, %c: i32, %f: f32) -> (i32, i1, f32) {
  %r:3 = dataflow.graph(%x = %a : i32, %y = %b : i32, %z = %c : i32,
                        %ff = %f : f32) -> (i32, i1, f32) {
    %t0  = arith.addi %x, %y : i32
    %t1  = arith.subi %t0, %z : i32
    %t2  = arith.muli %t1, %z : i32
    %t3  = arith.andi %t2, %y : i32
    %t4  = arith.ori  %t3, %y : i32
    %t5  = arith.xori %t4, %y : i32
    %t6  = arith.addi %t5, %z : i32
    %t7  = arith.subi %t6, %x : i32
    %t8  = arith.muli %t7, %x : i32
    %t9  = arith.andi %t8, %y : i32
    %t10 = arith.ori  %t9, %z : i32
    %t11 = arith.xori %t10, %x : i32
    %t12 = arith.addi %t11, %y : i32
    %t13 = arith.subi %t12, %z : i32
    %t14 = arith.muli %t13, %y : i32
    %t15 = arith.addi %t14, %x : i32
    %t16 = arith.subi %t15, %y : i32
    %t17 = arith.muli %t16, %z : i32
    %t18 = arith.addi %t17, %x : i32
    %t19 = arith.muli %t18, %y : i32
    %p0 = arith.cmpi eq, %t1, %t6 : i32
    %p1 = arith.cmpi slt, %t9, %t14 : i32
    %p2 = arith.cmpi sgt, %t11, %t19 : i32
    %s0 = math.sin %ff : f32
    %s1 = math.cos %s0 : f32
    %s2 = math.sin %s1 : f32
    %s3 = math.cos %s2 : f32
    %q0 = math.sqrt %s3 : f32
    %q1 = math.sqrt %q0 : f32
    %q2 = math.sqrt %q1 : f32
    dataflow.yield %t19, %p0, %q2 : i32, i1, f32
  }
  return %r#0, %r#1, %r#2 : i32, i1, f32
}
