// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @subgraph_empty
func.func @subgraph_empty() {
  // CHECK: dataflow.subgraph() -> ()
  dataflow.subgraph() -> () {
  }
  return
}

// CHECK-LABEL: @subgraph_arith_math
func.func @subgraph_arith_math(%a: i32, %b: i32, %f: f32) -> (i32, f32) {
  // CHECK: %{{.*}}:2 = dataflow.subgraph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : f32) -> (i32, f32)
  %r:2 = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %f : f32) -> (i32, f32) {
    // CHECK: arith.addi
    %s = arith.addi %x, %y : i32
    // CHECK: math.absf
    %m = math.absf %z : f32
    dataflow.yield %s, %m : i32, f32
  }
  return %r#0, %r#1 : i32, f32
}

// CHECK-LABEL: @subgraph_dataflow_pipeline
func.func @subgraph_dataflow_pipeline(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: %{{.*}}:2 = dataflow.subgraph
  %i, %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32) -> (i32, i1) {
    %ix, %rwc = dataflow.stream %l, %u, %s step add while slt : i32
    dataflow.yield %ix, %rwc : i32, i1
  }
  return %i, %r : i32, i1
}

// CHECK-LABEL: @subgraph_feedback
func.func @subgraph_feedback(%cond: i1, %init: i32) -> i32 {
  // CHECK: dataflow.subgraph
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32) -> i32 {
    %out = dataflow.carry %c, %i, %out : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
