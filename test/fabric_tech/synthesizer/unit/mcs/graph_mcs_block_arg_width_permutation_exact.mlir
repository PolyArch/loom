// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// The two inputs differ only by block-argument order, and the permuted
// arguments have different widths. This must bypass positional wrapper-port
// assumptions before the shared arith.select can be materialized.

// CHECK: remark: {{.*}}synth-stat group=block_arg_width_permutation strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: nodes=1/0/0
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.select]

func.func @pat_select_cxy(%cond: i1, %a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "block_arg_width_permutation"} {
  %r = dataflow.subgraph(%c = %cond : i1, %x = %a : i32,
                         %y = %b : i32) -> i32 {
    %out = arith.select %c, %x, %y : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_select_xyc(%cond: i1, %a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "block_arg_width_permutation"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %c = %cond : i1) -> i32 {
    %out = arith.select %c, %x, %y : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
