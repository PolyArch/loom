// RUN: loom-dfg-sim %s --graph structured_gate_reentry --arg 0=false --arg 0=true --arg 0=false --arg 0=true --arg 0=true --arg 0=false --arg 1=0 --arg 1=1 --arg 1=2 --arg 1=3 --arg 1=4 --arg 1=5 --arg 2=0 --arg 2=1 --arg 2=10 --arg 2=3 --arg 2=104 --arg 2=10 --arg 3=true --arg 3=true --arg 3=true --arg 3=true --arg 3=true --arg 3=true --arg 3=false --arg 4=none --arg 4=none --arg 4=none --arg 4=none --arg 4=none --arg 4=none --arg 4=none --arg 5=false --arg 5=false --arg 5=false --arg 5=false --arg 5=false --arg 5=false --arg 5=true --arg 6=none --arg 6=none --arg 6=none --arg 6=none --arg 6=none --arg 6=none --arg 7=false --arg 7=false --arg 7=true --arg 8=0 --arg 8=100 --arg 8=0 --arg 9=10 --arg 9=0 --arg 9=10 --arg 10=false --arg 10=false --arg 10=true --output %t.structured-gate.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-GATE < %t.structured-gate.json

// STRUCTURED-GATE-DAG: "graph": "structured_gate_reentry"
// STRUCTURED-GATE-DAG: "status": "pass"
// STRUCTURED-GATE-DAG: "dataflow.gate": 6
// STRUCTURED-GATE-DAG: "arith.addi": 6
// STRUCTURED-GATE-DAG: "arith.select": 3
// STRUCTURED-GATE-DAG: "index:128"

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph.func private @structured_gate_reentry(
      %ctrl: none, %gate_phase: i1, %gate_value: index,
      %contribution: index, %acc_phase: i1, %acc_unit: none,
      %acc_last: i1, %gate_unit: none, %gate_close_last: i1,
      %bonus_true: index, %bonus_false: index, %bonus_last: i1)
      -> (none, index)
      attributes {input_segments = array<i32: 0, 11, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %after_cond, %after_value = dataflow.gate %gate_phase, %gate_value : index
    %bonus = arith.select %after_cond, %bonus_true, %bonus_false : index

    %zero = arith.constant 0 : index
    %acc = dataflow.carry %acc_phase, %zero, %next : index
    %next = arith.addi %acc, %contribution : index

    %acc_close:2 = dataflow.demux %acc_phase, %acc_unit
        : (i1, none) -> (none, none)
    %acc_final:2 = dataflow.demux %acc_last, %acc
        : (i1, index) -> (index, index)

    %gate_closes:2 = dataflow.demux %gate_phase, %gate_unit
        : (i1, none) -> (none, none)
    %gate_close_pair:2 = dataflow.sync %gate_closes#0, %gate_close_last
        : (none, i1) -> (none, i1)
    %gate_complete:2 = dataflow.demux %gate_close_pair#1, %gate_close_pair#0
        : (i1, none) -> (none, none)

    %bonus_final:2 = dataflow.demux %bonus_last, %bonus
        : (i1, index) -> (index, index)
    %retired:4 = dataflow.sync %gate_complete#1, %acc_close#0,
        %acc_final#1, %bonus_final#1
        : (none, none, index, index) -> (none, none, index, index)
    dataflow.graph.return %retired#0, %retired#2 : none, index
  }
}
