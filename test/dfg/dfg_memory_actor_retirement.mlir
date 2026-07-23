// RUN: loom-dfg-sim %s --graph fence_retires_stream --arg 0=0 --arg 1=3 --arg 2=1 --output %t.json
// RUN: FileCheck %s < %t.json

// The retirement frontier reaches the stream's close through the new actor's
// control event, so the graph is finalized and only the unmodeled contract
// stops it. A close/reset rule that still enumerated load and store would
// reject this graph before admission instead.
// CHECK-DAG: "status": "unsupported"
// CHECK-DAG: unsupported op: dataflow.fence

module {
  dataflow.graph private @fence_retires_stream(
      %start: none, %lb: i16, %ub: i16, %step: i16) -> ()
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %iv, %phase = dataflow.stream %lb, %ub, %step step add while slt : i16
    %execution = dataflow.carry %phase, %start, %lane#1 : none
    %lane:2 = dataflow.demux %phase, %execution : (i1, none) -> (none, none)
    %done = dataflow.fence %lane#0
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values() streams() memories() complete(%done : none)
  }
}
