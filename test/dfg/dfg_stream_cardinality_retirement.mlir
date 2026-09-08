// RUN: not loom-dfg-sim %s --graph stream_cardinality_without_close \
// RUN:   --arg 0=0 --arg 1=3 --arg 2=1 --output %t.json 2>&1 | FileCheck %s

// Matching recurrence parameters prove that the carry consumes exactly one
// issue completion per iteration. They do not prove that the distinct drain
// stream has closed when only the issue stream's exit reaches graph return.
// CHECK: retirement frontier does not cover close/reset of 'dataflow.stream'

module {
  dataflow.graph private @stream_cardinality_without_close(
      %start: none, %lb: i16, %ub: i16, %step: i16) -> ()
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %issue_iv, %issue_phase = dataflow.stream %lb, %ub, %step step add while slt : i16
    %done_iv, %done_phase = dataflow.stream %lb, %ub, %step step add while slt : i16
    %issue = dataflow.invariant %issue_phase, %start : none
    %done = dataflow.carry %done_phase, %start, %issue_lane#1 : none
    %issue_lane:2 = dataflow.demux %issue_phase, %issue : (i1, none) -> (none, none)
    %done_lane:2 = dataflow.demux %done_phase, %done : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories() complete(%issue_lane#0 : none)
  }
}
