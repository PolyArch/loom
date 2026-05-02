// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/cost_baseline.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BASE
// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/cost_random.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RAND

// Acceptance criterion 2 (incremental_random): on a workload where
// input ordering matters, the chosen FU has cost <= the cost of
// `incremental` with the default `largest_first` heuristic.
//
// We pick a tier-B-friendly workload (mixed prefix lengths sharing the
// arith.addi/subi share group) that the deterministic Incremental
// already synthesizes successfully. The reference cost is taken from
// the BASE run; the random run with restarts=8 and seed=42 must report
// the same magnitude (any cost-equivalent best-cost permutation is
// fine, since the `<=` bound is trivially satisfied by equality). The
// magnitude `194` mirrors the canonical wrapper produced by the
// largest_first heuristic on this input; see
// `IncrementalRandomSynthesizer::run` for the cost ranking rule.

// BASE: synth-stat group=cost_demo strategy=incremental reason=success
// BASE-SAME: cost=1.940000e+02

// RAND: synth-stat group=cost_demo strategy=incremental_random reason=success
// RAND-SAME: cost=1.940000e+02

func.func @cost_pat_add(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "cost_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @cost_pat_sub(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "cost_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @cost_pat_add_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "cost_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}

func.func @cost_pat_sub_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "cost_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.subi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}
