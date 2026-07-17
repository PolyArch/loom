// RUN: not loom-dfg-sim %s --graph opposite_demux_completion --arg 0=true --output %t.sim.json 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: not loom-pnr-map --dfg-mlir %s --graph opposite_demux_completion --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload opposite_demux_completion --output %t.mapping.csv --artifact %t.mapping.json 2>&1 | FileCheck %s --check-prefix=PNR
// RUN: not loom-dfg-sim %s --graph opposite_mux_value_completion --arg 0=false --arg 1=7 --output %t.mux-value.json 2>&1 | FileCheck %s --check-prefix=MUX-VALUE
// RUN: not loom-dfg-sim %s --graph opposite_mux_close_completion --arg 0=false --arg 1=0 --arg 2=2 --arg 3=1 --output %t.mux-close.json 2>&1 | FileCheck %s --check-prefix=MUX-CLOSE
// RUN: rm -f %t.published.mlir
// RUN: not loom-lower %s -o %t.published.mlir 2>&1 | FileCheck %s --check-prefix=LOWER
// RUN: test ! -e %t.published.mlir

// SIM: retirement frontier does not causally cover stream output #0
// PNR: retirement frontier does not causally cover stream output #0
// MUX-VALUE: retirement frontier does not causally cover value output #0
// MUX-CLOSE: retirement frontier does not cover close/reset of 'dataflow.stream'
// LOWER: final Dataflow validation failed for @opposite_demux_completion

// The two demux outputs are selected alternatives. Completing the false lane
// cannot retire the stream exported on the true lane merely because both
// values were produced by one operation.
module {
  dataflow.graph.func private @opposite_demux_completion(
      %start: none, %select: i1) -> (none, none)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %lanes:2 = dataflow.demux %select, %start : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%lanes#1 : none) memories()
        complete(%lanes#0 : none)
  }

  dataflow.graph.func private @opposite_mux_value_completion(
      %start: none, %select: i1, %value: i32) -> (none, i32)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value_lanes:2 = dataflow.demux %select, %value
        : (i1, i32) -> (i32, i32)
    %value_branch_done:2 = dataflow.sync %start, %value_lanes#0
        : (none, i32) -> (none, i32)
    %value_complete = dataflow.mux %select, %start, %value_branch_done#0
        : (i1, none, none) -> none
    dataflow.graph.return values(%value_lanes#0 : i32) streams() memories()
        complete(%value_complete : none)
  }

  dataflow.graph.func private @opposite_mux_close_completion(
      %start: none, %select: i1, %lower: i64, %upper: i64, %step: i64)
      -> none
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %close_index, %close_phase = dataflow.stream %lower, %upper, %step
        step add while slt : i64
    %close_tokens = dataflow.invariant %close_phase, %start : none
    %closed:2 = dataflow.demux %close_phase, %close_tokens
        : (i1, none) -> (none, none)
    %branches:2 = dataflow.demux %select, %closed#0
        : (i1, none) -> (none, none)
    %close_complete = dataflow.mux %select, %start, %branches#0
        : (i1, none, none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%close_complete : none)
  }
}
