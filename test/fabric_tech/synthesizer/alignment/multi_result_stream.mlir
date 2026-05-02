// RUN: loom-alignment-test %s | FileCheck %s

// dataflow.stream produces two results: %idx (index port) and %rwc
// (i1 trailing sentinel). The Source must address each by resultIndex.
// Two funcs cover both pickings and assert the correct anchor / bit-width.

func.func @stream_yield_idx(%lb: i32, %ub: i32, %step: i32) -> i32 {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32) -> i32 {
    %i, %r2 = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i32
    dataflow.yield %i : i32
  }
  return %r : i32
}

func.func @stream_yield_rwc(%lb: i32, %ub: i32, %step: i32) -> i1 {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32) -> i1 {
    %i, %r2 = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i32
    dataflow.yield %r2 : i1
  }
  return %r : i1
}

// CHECK: func @stream_yield_idx:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:dataflow.stream#0
// CHECK-NEXT:   signature[0]=dataflow.stream;-;bw=32;arity=3;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=0

// CHECK: func @stream_yield_rwc:
// CHECK-NEXT:   yield-anchors=1
// CHECK-NEXT:   anchor[0]=BodyOp:dataflow.stream#1
// CHECK-NEXT:   signature[0]=dataflow.stream;-;bw=1;arity=3;ohash=0x{{[0-9a-f]+}}
// CHECK-NEXT:   backedges=0
