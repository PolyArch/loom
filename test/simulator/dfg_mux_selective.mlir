// RUN: loom-dfg-sim %s --graph mux_false_lane --arg 0=none --arg 1=false --arg 2=7 --arg 3=99 --output %t.false.json
// RUN: FileCheck %s --check-prefix=FALSE < %t.false.json
// RUN: loom-dfg-sim %s --graph mux_true_lane --arg 0=none --arg 1=true --arg 2=7 --arg 3=99 --output %t.true.json
// RUN: FileCheck %s --check-prefix=TRUE < %t.true.json
// RUN: loom-dfg-sim %s --graph demux_false_lane --arg 0=none --arg 1=false --arg 2=7 --output %t.demux_false.json
// RUN: FileCheck %s --check-prefix=DEMUX-FALSE < %t.demux_false.json
// RUN: loom-dfg-sim %s --graph demux_true_lane --arg 0=none --arg 1=true --arg 2=99 --output %t.demux_true.json
// RUN: FileCheck %s --check-prefix=DEMUX-TRUE < %t.demux_true.json
// RUN: loom-dfg-sim %s --graph computed_i1_selectors --arg 0=none --arg 1=true --arg 2=11 --arg 3=22 --output %t.computed_i1.json
// RUN: FileCheck %s --check-prefix=COMPUTED-I1 < %t.computed_i1.json
// RUN: loom-dfg-sim %s --graph computed_i1_arith_select --arg 0=none --arg 1=true --arg 2=11 --arg 3=22 --output %t.computed_select_true.json
// RUN: FileCheck %s --check-prefix=COMPUTED-SELECT-TRUE < %t.computed_select_true.json
// RUN: loom-dfg-sim %s --graph computed_i1_arith_select --arg 0=none --arg 1=false --arg 2=11 --arg 3=22 --output %t.computed_select_false.json
// RUN: FileCheck %s --check-prefix=COMPUTED-SELECT-FALSE < %t.computed_select_false.json
// RUN: loom-dfg-sim %s --graph structured_mux_loop --arg 0=none --arg 1=true --arg 2=11 --arg 3=22 --output %t.structured_mux.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-MUX < %t.structured_mux.json

// FALSE-DAG: "status": "pass"
// FALSE-DAG: "dataflow.mux": 1
// FALSE-DAG: "i64:7"
// FALSE-NOT: "incomplete final outputs"

// TRUE-DAG: "status": "pass"
// TRUE-DAG: "dataflow.mux": 1
// TRUE-DAG: "i64:99"
// TRUE-NOT: "incomplete final outputs"

// DEMUX-FALSE-DAG: "status": "pass"
// DEMUX-FALSE-DAG: "dataflow.demux": 1
// DEMUX-FALSE-DAG: "i64:7"
// DEMUX-FALSE-NOT: "incomplete final outputs"

// DEMUX-TRUE-DAG: "status": "pass"
// DEMUX-TRUE-DAG: "dataflow.demux": 1
// DEMUX-TRUE-DAG: "i64:99"
// DEMUX-TRUE-NOT: "incomplete final outputs"

// COMPUTED-I1-DAG: "status": "pass"
// COMPUTED-I1-DAG: "dataflow.demux": 1
// COMPUTED-I1-DAG: "dataflow.mux": 1
// COMPUTED-I1-DAG: "i64:22"
// COMPUTED-I1-NOT: "selector is out of range"

// COMPUTED-SELECT-TRUE-DAG: "status": "pass"
// COMPUTED-SELECT-TRUE-DAG: "i64:22"
// COMPUTED-SELECT-TRUE-NOT: "i64:11"

// COMPUTED-SELECT-FALSE-DAG: "status": "pass"
// COMPUTED-SELECT-FALSE-DAG: "i64:11"

// STRUCTURED-MUX-DAG: "status": "pass"
// STRUCTURED-MUX-DAG: "dataflow.mux": 1
// STRUCTURED-MUX-DAG: "i64:22"
// STRUCTURED-MUX-NOT: "unsupported op: dataflow.mux"

module {
  dataflow.graph.func private @mux_false_lane(%ctrl: none, %sel: i1,
                                              %false_value: i64,
                                              %true_value: i64)
      -> (none, i64) {
    %after_cond, %true_lane = dataflow.gate %sel, %true_value : i64
    %out = dataflow.mux %sel, %false_value, %true_lane : (i1, i64, i64) -> i64
    dataflow.graph.return %ctrl, %out : none, i64
  }

  dataflow.graph.func private @mux_true_lane(%ctrl: none, %sel: i1,
                                             %false_value: i64,
                                             %true_value: i64)
      -> (none, i64) {
    %after_cond, %true_lane = dataflow.gate %sel, %true_value : i64
    %out = dataflow.mux %sel, %false_value, %true_lane : (i1, i64, i64) -> i64
    dataflow.graph.return %ctrl, %out : none, i64
  }

  dataflow.graph.func private @demux_false_lane(%ctrl: none, %sel: i1,
                                                %value: i64)
      -> (none, i64) {
    %false_lane, %true_lane = dataflow.demux %sel, %value : (i1, i64) -> (i64, i64)
    dataflow.graph.return %ctrl, %false_lane : none, i64
  }

  dataflow.graph.func private @demux_true_lane(%ctrl: none, %sel: i1,
                                               %value: i64)
      -> (none, i64) {
    %false_lane, %true_lane = dataflow.demux %sel, %value : (i1, i64) -> (i64, i64)
    dataflow.graph.return %ctrl, %true_lane : none, i64
  }

  dataflow.graph.func private @computed_i1_selectors(%ctrl: none, %sel: i1,
                                                     %false_value: i64,
                                                     %true_value: i64)
      -> (none, i64) {
    %computed_sel = arith.andi %sel, %sel : i1
    %false_lane, %true_lane = dataflow.demux %computed_sel, %false_value : (i1, i64) -> (i64, i64)
    %out = dataflow.mux %computed_sel, %false_lane, %true_value : (i1, i64, i64) -> i64
    dataflow.graph.return %ctrl, %out : none, i64
  }

  dataflow.graph.func private @computed_i1_arith_select(%ctrl: none, %sel: i1,
                                                        %false_value: i64,
                                                        %true_value: i64)
      -> (none, i64) {
    %computed_sel = arith.andi %sel, %sel : i1
    %out = arith.select %computed_sel, %true_value, %false_value : i64
    dataflow.graph.return %ctrl, %out : none, i64
  }

  dataflow.graph.func private @structured_mux_loop(%ctrl: none, %sel: i1,
                                                   %false_value: i64,
                                                   %true_value: i64)
      -> (none, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%carry = %false_value) -> (i64) {
      %selected = dataflow.mux %sel, %carry, %true_value : (i1, i64, i64) -> i64
      scf.yield %selected : i64
    }
    dataflow.graph.return %ctrl, %out : none, i64
  }
}
