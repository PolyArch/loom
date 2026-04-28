// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: a single FU exposing three independent sw_config axes:
//   * input fabric.mux (sel + discard + disconnect),
//   * output fabric.demux (sel + discard + disconnect),
//   * a fabric.op[@arith.cmpi] with a hw_params predicate set.
// The Cartesian product is the cross product of every axis, and
// effective-config dedup collapses isomorphic templates. We don't pin
// the exact count (which depends on dedup heuristics), only that:
//   * every advertised predicate appears at least once,
//   * mux/demux configurations expose at least both sel paths.

// CHECK-LABEL: @fu_cartesian
func.func @fu_cartesian(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                        %c: !fabric.bits<32>) {
  %x, %y = fabric.fu(%aa = %a : !fabric.bits<32>,
                     %bb = %b : !fabric.bits<32>,
                     %cc = %c : !fabric.bits<32>)
                    -> (!fabric.bits<1>, !fabric.bits<1>) {
    %m = fabric.mux %aa, %bb : !fabric.bits<32>
    %k = fabric.op [@arith.cmpi] (%m, %cc)
         {hw_params = [{predicate = ["eq", "ne"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    %o0, %o1 = fabric.demux %k : !fabric.bits<1> -> 2
    fabric.yield %o0, %o1 : !fabric.bits<1>, !fabric.bits<1>
  }
  // Both predicates must appear among the materialized templates.
  // CHECK-DAG: arith.cmpi eq
  // CHECK-DAG: arith.cmpi ne
  return
}
