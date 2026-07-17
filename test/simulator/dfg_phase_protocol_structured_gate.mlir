// RUN: loom-dfg-sim %s --graph structured_gate_reentry --arg 0=none --output %t.structured-gate.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-GATE < %t.structured-gate.json
// RUN: loom-dfg-sim %s --graph structured_gate_reentry_no_phase --arg 0=none --output %t.structured-gate-no-phase.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-GATE-NO-PHASE < %t.structured-gate-no-phase.json

// STRUCTURED-GATE-DAG: "graph": "structured_gate_reentry"
// STRUCTURED-GATE-DAG: "status": "pass"
// STRUCTURED-GATE-DAG: "dataflow.gate": 6
// STRUCTURED-GATE-DAG: "arith.addi": 6
// STRUCTURED-GATE-DAG: "arith.select": 3
// STRUCTURED-GATE-DAG: "index:128"

// STRUCTURED-GATE-NO-PHASE-DAG: "graph": "structured_gate_reentry_no_phase"
// STRUCTURED-GATE-NO-PHASE-DAG: "status": "blocked"
// STRUCTURED-GATE-NO-PHASE-DAG: "structured scf.index_switch failed to execute arith.index_castui"

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @structured_gate_reentry(%ctrl: none)
      -> (none, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %result = scf.for %i = %c0 to %c6 step %c1
        iter_args(%acc = %c0) -> (index) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %is1 = arith.cmpi eq, %i, %c1 : index
      %is3 = arith.cmpi eq, %i, %c3 : index
      %is4 = arith.cmpi eq, %i, %c4 : index
      %is1or3 = arith.ori %is1, %is3 : i1
      %phase = arith.ori %is1or3, %is4 : i1
      %after_cond, %after_value = dataflow.gate %phase, %i : index
      %next = scf.index_switch %i -> index
      case 0 {
        scf.yield %acc : index
      }
      case 1 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 2 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      case 3 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 4 {
        %with_value = arith.addi %acc, %after_value : index
        %bonus = arith.select %after_cond, %c100, %c0 : index
        %sum = arith.addi %with_value, %bonus : index
        scf.yield %sum : index
      }
      case 5 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      default {
        scf.yield %acc : index
      }
      scf.yield %next : index
    }
    dataflow.graph.return %ctrl, %result : none, index
  }

  dataflow.graph.func private @structured_gate_reentry_no_phase(%ctrl: none)
      -> (none, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %i = %c0 to %c3 step %c1
        iter_args(%acc = %c0) -> (index) {
      %phase = arith.cmpi ne, %i, %c1 : index
      %after_cond, %after_value = dataflow.gate %phase, %i : index
      %next = scf.index_switch %i -> index
      case 0 {
        %sum = arith.addi %acc, %after_value : index
        scf.yield %sum : index
      }
      case 1 {
        %bonus = arith.select %after_cond, %c0, %c10 : index
        %sum = arith.addi %acc, %bonus : index
        scf.yield %sum : index
      }
      case 2 {
        %with_value = arith.addi %acc, %after_value : index
        %unexpected_phase = arith.index_castui %after_cond : i1 to index
        %sum = arith.addi %with_value, %unexpected_phase : index
        scf.yield %sum : index
      }
      default {
        scf.yield %acc : index
      }
      scf.yield %next : index
    }
    dataflow.graph.return %ctrl, %result : none, index
  }
}
