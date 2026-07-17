// RUN: loom-dfg-sim %s --graph mux_false_lane --arg 0=false --arg 1=7 --arg 2=99 --output %t.false.json
// RUN: FileCheck %s --check-prefix=FALSE < %t.false.json
// RUN: loom-dfg-sim %s --graph mux_true_lane --arg 0=true --arg 1=7 --arg 2=99 --output %t.true.json
// RUN: FileCheck %s --check-prefix=TRUE < %t.true.json
// RUN: loom-dfg-sim %s --graph demux_false_lane --arg 0=false --arg 1=7 --output %t.demux_false.json
// RUN: FileCheck %s --check-prefix=DEMUX-FALSE < %t.demux_false.json
// RUN: loom-dfg-sim %s --graph demux_true_lane --arg 0=true --arg 1=99 --output %t.demux_true.json
// RUN: FileCheck %s --check-prefix=DEMUX-TRUE < %t.demux_true.json
// RUN: loom-dfg-sim %s --graph computed_i1_selectors --arg 0=true --arg 1=11 --arg 2=22 --output %t.computed_i1.json
// RUN: FileCheck %s --check-prefix=COMPUTED-I1 < %t.computed_i1.json
// RUN: loom-dfg-sim %s --graph computed_i1_arith_select --arg 0=true --arg 1=11 --arg 2=22 --output %t.computed_select_true.json
// RUN: FileCheck %s --check-prefix=COMPUTED-SELECT-TRUE < %t.computed_select_true.json
// RUN: loom-dfg-sim %s --graph computed_i1_arith_select --arg 0=false --arg 1=11 --arg 2=22 --output %t.computed_select_false.json
// RUN: FileCheck %s --check-prefix=COMPUTED-SELECT-FALSE < %t.computed_select_false.json
// RUN: loom-dfg-sim %s --graph structured_mux_loop --arg 0=true --arg 1=11 --arg 2=22 --output %t.structured_mux.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-MUX < %t.structured_mux.json
// RUN: loom-dfg-sim %s --graph structured_demux_store_noop --arg 0=false --arg 1=99 --memref 2=7 --output %t.demux_store_noop.json
// RUN: FileCheck %s --check-prefix=DEMUX-STORE-NOOP < %t.demux_store_noop.json
// RUN: loom-dfg-sim %s --graph structured_demux_store_noop --arg 0=true --arg 1=99 --memref 2=7 --output %t.demux_store_active.json
// RUN: FileCheck %s --check-prefix=DEMUX-STORE-ACTIVE < %t.demux_store_active.json

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

// DEMUX-STORE-NOOP-DAG: "status": "pass"
// DEMUX-STORE-NOOP-DAG: "arg2": [
// DEMUX-STORE-NOOP-DAG: "i64:7"
// DEMUX-STORE-NOOP-DAG: "dataflow.demux": 3
// DEMUX-STORE-NOOP-NOT: "dataflow.store"

// DEMUX-STORE-ACTIVE-DAG: "status": "pass"
// DEMUX-STORE-ACTIVE-DAG: "arg2": [
// DEMUX-STORE-ACTIVE-DAG: "i64:99"
// DEMUX-STORE-ACTIVE-DAG: "dataflow.demux": 3
// DEMUX-STORE-ACTIVE-DAG: "dataflow.store": 1

module {
  dataflow.graph.func private @mux_false_lane(%ctrl: none, %sel: i1,
                                              %false_value: i64,
                                              %true_value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %unused, %true_lane = dataflow.demux %sel, %true_value
        : (i1, i64) -> (i64, i64)
    %out = dataflow.mux %sel, %false_value, %true_lane : (i1, i64, i64) -> i64
    %published:2 = dataflow.sync %ctrl, %out
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @mux_true_lane(%ctrl: none, %sel: i1,
                                             %false_value: i64,
                                             %true_value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %unused, %true_lane = dataflow.demux %sel, %true_value
        : (i1, i64) -> (i64, i64)
    %out = dataflow.mux %sel, %false_value, %true_lane : (i1, i64, i64) -> i64
    %published:2 = dataflow.sync %ctrl, %out
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @demux_false_lane(%ctrl: none, %sel: i1,
                                                %value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %false_lane, %true_lane = dataflow.demux %sel, %value : (i1, i64) -> (i64, i64)
    %published:2 = dataflow.sync %ctrl, %false_lane
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @demux_true_lane(%ctrl: none, %sel: i1,
                                               %value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %false_lane, %true_lane = dataflow.demux %sel, %value : (i1, i64) -> (i64, i64)
    %published:2 = dataflow.sync %ctrl, %true_lane
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @computed_i1_selectors(%ctrl: none, %sel: i1,
                                                     %false_value: i64,
                                                     %true_value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %computed_sel = arith.andi %sel, %sel : i1
    %false_lane, %true_lane = dataflow.demux %computed_sel, %false_value : (i1, i64) -> (i64, i64)
    %out = dataflow.mux %computed_sel, %false_lane, %true_value : (i1, i64, i64) -> i64
    %published:2 = dataflow.sync %ctrl, %out
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @computed_i1_arith_select(%ctrl: none, %sel: i1,
                                                        %false_value: i64,
                                                        %true_value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %computed_sel = arith.andi %sel, %sel : i1
    %out = arith.select %computed_sel, %true_value, %false_value : i64
    %published:2 = dataflow.sync %ctrl, %out
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @structured_mux_loop(%ctrl: none, %sel: i1,
                                                   %false_value: i64,
                                                   %true_value: i64)
      -> (none, i64)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %out = dataflow.mux %sel, %false_value, %true_value
        : (i1, i64, i64) -> i64
    %published:2 = dataflow.sync %ctrl, %out
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph.func private @structured_demux_store_noop(
      %ctrl: none, %sel: i1, %value: i64, %mem: memref<?xi64>) -> none
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %addr = dataflow.constant %ctrl {const_value = 0 : index} : index
    %addr_false, %addr_true = dataflow.demux %sel, %addr
        : (i1, index) -> (index, index)
    %data_false, %data_true = dataflow.demux %sel, %value
        : (i1, i64) -> (i64, i64)
    %ctrl_false, %ctrl_true = dataflow.demux %sel, %ctrl
        : (i1, none) -> (none, none)
    %stored = dataflow.store %mem[%addr_true] %data_true %ctrl_true
        : memref<?xi64>
    %complete = dataflow.mux %sel, %ctrl_false, %stored
        : (i1, none, none) -> none
    dataflow.graph.return %complete : none
  }
}
