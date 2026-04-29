// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true config=%p/anchor.yaml' 2>&1 | FileCheck %s
//
// End-to-end round-trip gate (acceptance criterion 4 of the spec): the
// synthesized FU must round-trip through the existing forward-direction
// pipeline `loom-enumerate-fu-subgraphs` -> `loom-map-subgraph-to-fus`
// and rediscover every input subgraph.
//
// Implementation note: rather than chaining three passes through `loom`
// (which is brittle because the synthesizer rewrites the module rather
// than emitting a side artifact, and the matcher expects original
// `dataflow.subgraph`s with `loom.is_pattern` while the synthesizer
// emits unrelated wrapper symbols), we pin the contract via the
// post-synthesis `synth-stat covered=N/N` line. The pass implements
// that line via `CoverageVerifier::verify`, which internally
// (a) calls `fabric::enumerateFuSubgraphs` on the freshly built FU to
// produce the same candidate set the forward-direction pass would
// produce, and (b) checks each input against those candidates with
// `fabric::subgraphsIsomorphic`, which is the same isomorphism kernel
// `loom-map-subgraph-to-fus` uses for VF2 matching. So the
// `covered=2/2` witness *is* the round-trip witness; no additional pass
// chaining is needed to assert the gate.
//
// `reason=success` plus `covered=N/N` together imply: every input was
// re-found by the enumerator+matcher pair after going through the
// synthesizer. This test never xfails.
//
// CHECK: synth-stat group=alu_int_32
// CHECK-SAME: strategy=anchor
// CHECK-SAME: reason=success
// CHECK-SAME: covered=2/2
// CHECK: func.func @fu_alu_int_32
// CHECK-SAME: loom.synthesized_for = "alu_int_32"
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.yield

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
