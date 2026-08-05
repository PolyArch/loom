// Anchors for the ActivationPreservingConstantFold typed rewrite. A foldable
// single-result pure canonical Compute actor collapses into one exact typed
// `dataflow.constant` when its operands are `dataflow.constant` results that
// are triggered by the exact same ctrl SSA value and have no users outside
// that candidate actor. The activation stream is preserved, never merged and
// never made timeless.

// RUN: loom-dfg-sim %s --graph same_control_fold --output %t.src-fold.json
// RUN: FileCheck %s --check-prefix=SUM < %t.src-fold.json
// RUN: loom-dfg-sim %s --graph repeated_operand_fold --output %t.src-repeated.json
// RUN: FileCheck %s --check-prefix=DOUBLED < %t.src-repeated.json
// RUN: loom-dfg-sim %s --graph graph_order_fold --output %t.src-graph-order.json
// RUN: FileCheck %s --check-prefix=SUM < %t.src-graph-order.json
// RUN: loom-dfg-sim %s --graph distinct_control --output %t.src-distinct.json
// RUN: FileCheck %s --check-prefix=SUM < %t.src-distinct.json

// RUN: loom-raise-opt --dataflow-rewrite=kind=activation-preserving-constant-fold %s -o %t.opt.mlir
// RUN: FileCheck %s --check-prefix=OPT < %t.opt.mlir
// RUN: loom-dfg-sim %t.opt.mlir --graph same_control_fold --output %t.opt-fold.json
// RUN: FileCheck %s --check-prefix=SUM < %t.opt-fold.json
// RUN: loom-dfg-sim %t.opt.mlir --graph repeated_operand_fold --output %t.opt-repeated.json
// RUN: FileCheck %s --check-prefix=DOUBLED < %t.opt-repeated.json
// RUN: loom-dfg-sim %t.opt.mlir --graph graph_order_fold --output %t.opt-graph-order.json
// RUN: FileCheck %s --check-prefix=SUM < %t.opt-graph-order.json
// RUN: loom-dfg-sim %t.opt.mlir --graph distinct_control --output %t.opt-distinct.json
// RUN: FileCheck %s --check-prefix=SUM < %t.opt-distinct.json
// RUN: loom-dfg-sim %t.opt.mlir --graph selector_actor --output %t.opt-selector.json
// RUN: FileCheck %s --check-prefix=SELECTED < %t.opt-selector.json

// SUM-DAG: "i32:7"
// SUM-DAG: "status": "pass"

// DOUBLED-DAG: "i32:10"
// DOUBLED-DAG: "status": "pass"

// The selector still routes lane 1, so the observed payload stays 4.
// SELECTED-DAG: "i32:4"
// SELECTED-DAG: "status": "pass"

module {
  // OPT-LABEL: dataflow.graph private @same_control_fold
  // OPT-NOT: arith.addi
  // OPT-NOT: const_value = 3 : i32
  // OPT-NOT: const_value = 4 : i32
  // OPT: %[[FOLDED:[^ ]*]] = dataflow.constant %[[CTRL:[^ ]*]] {const_value = 7 : i32} : i32
  // OPT: dataflow.sync %[[CTRL]], %[[FOLDED]] :
  dataflow.graph private @same_control_fold(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %start {const_value = 3 : i32} : i32
    %rhs = dataflow.constant %start {const_value = 4 : i32} : i32
    %sum = arith.addi %lhs, %rhs : i32
    %retired:2 = dataflow.sync %start, %sum : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  // One source constant legally feeds both operands of the single candidate
  // actor. Every use still belongs to that actor, so the constant fold is
  // exact and the constant is erased once, not once per operand.
  // OPT-LABEL: dataflow.graph private @repeated_operand_fold
  // OPT-NOT: arith.addi
  // OPT-NOT: const_value = 5 : i32
  // OPT: %[[TWICE:[^ ]*]] = dataflow.constant %[[REPEAT_CTRL:[^ ]*]] {const_value = 10 : i32} : i32
  // OPT: dataflow.sync %[[REPEAT_CTRL]], %[[TWICE]] :
  dataflow.graph private @repeated_operand_fold(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 5 : i32} : i32
    %sum = arith.addi %value, %value : i32
    %retired:2 = dataflow.sync %start, %sum : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  // A dataflow.graph is an MLIR Graph region: textual operation order does
  // not establish SSA dominance. A rewrite may erase source actors that occur
  // later in the block, so traversal must resolve each original actor through
  // its stable canonical identity instead of retaining raw operation pointers.
  // OPT-LABEL: dataflow.graph private @graph_order_fold
  // OPT-NOT: arith.addi
  // OPT-NOT: const_value = 3 : i32
  // OPT-NOT: const_value = 4 : i32
  // OPT: dataflow.constant %{{[^ ]*}} {const_value = 7 : i32} : i32
  dataflow.graph private @graph_order_fold(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i32
    %lhs = dataflow.constant %start {const_value = 3 : i32} : i32
    %rhs = dataflow.constant %start {const_value = 4 : i32} : i32
    %retired:2 = dataflow.sync %start, %sum : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  // The two source constants are triggered by different ctrl SSA values.
  // Folding them would merge two activation streams that are not proven equal.
  // OPT-LABEL: dataflow.graph private @distinct_control
  // OPT: %[[OTHER:[^ ]*]] = dataflow.sync %{{[^ ]*}} : (none) -> none
  // OPT: dataflow.constant %[[OTHER]] {const_value = 4 : i32} : i32
  // OPT: arith.addi
  dataflow.graph private @distinct_control(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %other = dataflow.sync %start : (none) -> none
    %lhs = dataflow.constant %start {const_value = 3 : i32} : i32
    %rhs = dataflow.constant %other {const_value = 4 : i32} : i32
    %sum = arith.addi %lhs, %rhs : i32
    %retired:2 = dataflow.sync %start, %sum : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  // A source constant with an external consumer keeps that consumer's token.
  // OPT-LABEL: dataflow.graph private @external_constant_use
  // OPT: %[[LHS:[^ ]*]] = dataflow.constant %{{[^ ]*}} {const_value = 3 : i32} : i32
  // OPT: arith.addi %[[LHS]]
  // OPT: dataflow.sync %{{[^,]*}}, %{{[^,]*}}, %[[LHS]] :
  dataflow.graph private @external_constant_use(%start: none) -> (i32, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %lhs = dataflow.constant %start {const_value = 3 : i32} : i32
    %rhs = dataflow.constant %start {const_value = 4 : i32} : i32
    %sum = arith.addi %lhs, %rhs : i32
    %retired:3 = dataflow.sync %start, %sum, %lhs
        : (none, i32, i32) -> (none, i32, i32)
    dataflow.graph.return values(%retired#1, %retired#2 : i32, i32)
        streams() memories() complete(%retired#0 : none)
  }

  // A selector actor is canonically classified as control, not compute, so it
  // is outside this rule even though every operand is a same-ctrl constant.
  // OPT-LABEL: dataflow.graph private @selector_actor
  // OPT: dataflow.mux
  dataflow.graph private @selector_actor(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sel = dataflow.constant %start {const_value = true} : i1
    %lhs = dataflow.constant %start {const_value = 3 : i32} : i32
    %rhs = dataflow.constant %start {const_value = 4 : i32} : i32
    %picked = dataflow.mux %sel, %lhs, %rhs : (i1, i32, i32) -> i32
    %retired:2 = dataflow.sync %start, %picked : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }

  // Signed division by zero has no exact total value, so the actor survives.
  // OPT-LABEL: dataflow.graph private @non_total_division
  // OPT: arith.divsi
  dataflow.graph private @non_total_division(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %numerator = dataflow.constant %start {const_value = 12 : i32} : i32
    %denominator = dataflow.constant %start {const_value = 0 : i32} : i32
    %quotient = arith.divsi %numerator, %denominator : i32
    %retired:2 = dataflow.sync %start, %quotient : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
}
