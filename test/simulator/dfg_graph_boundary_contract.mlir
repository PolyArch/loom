// RUN: not loom-dfg-sim %s --graph post_done_state --output %t.post.json 2>&1 | FileCheck %s --check-prefix=POST
// RUN: loom-dfg-sim %s --graph empty_stream_export --output %t.empty.json
// RUN: FileCheck %s --check-prefix=EMPTY-STREAM < %t.empty.json
// RUN: loom-dfg-sim %s --graph multi_stream_export --output %t.multi.json
// RUN: FileCheck %s --check-prefix=MULTI-STREAM < %t.multi.json
// RUN: loom-dfg-sim %s --graph memory_export --output %t.memory.json
// RUN: FileCheck %s --check-prefix=MEMORY < %t.memory.json
// RUN: loom-dfg-sim %s --graph invocation_reentry --invocations 3 --arg 0=0 --arg 0=1 --arg 0=2 --memref 1=0,0,0 --output %t.reentry.json
// RUN: FileCheck %s --check-prefix=REENTRY < %t.reentry.json

// POST: retirement frontier does not cover close/reset of 'dataflow.stream'

// EMPTY-STREAM-DAG: "status": "pass"
// EMPTY-STREAM-DAG: "final_stream_outputs": [
// EMPTY-STREAM-DAG: []

// MULTI-STREAM-DAG: "status": "pass"
// MULTI-STREAM-DAG: "final_stream_outputs": [
// MULTI-STREAM-DAG: "i32:0"
// MULTI-STREAM-DAG: "i32:1"

// MEMORY: memory export simulation is unsupported
// MEMORY: "status": "unsupported"

// REENTRY-DAG: "status": "pass"
// REENTRY-DAG: "dataflow.load": 3
// REENTRY-DAG: "dataflow.store": 3
// REENTRY-DAG: "arg1": [
// REENTRY-DAG: "i32:1"
// REENTRY-DAG: "i32:1"
// REENTRY-DAG: "i32:1"

module {
  dataflow.graph.func private @post_done_state(%start: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %c0 = dataflow.constant %start {const_value = 0 : i32} : i32
    %c1 = dataflow.constant %start {const_value = 1 : i32} : i32
    %c2 = dataflow.constant %start {const_value = 2 : i32} : i32
    %iv, %phase = dataflow.stream %c0, %c2, %c1
        step add while slt : i32
    %published:2 = dataflow.sync %start, %iv
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @empty_stream_export(%start: none)
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %c0 = dataflow.constant %start {const_value = 0 : i32} : i32
    %c1 = dataflow.constant %start {const_value = 1 : i32} : i32
    %iv, %phase = dataflow.stream %c0, %c0, %c1
        step add while slt : i32
    %tokens = dataflow.invariant %phase, %start : none
    %complete:2 = dataflow.demux %phase, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%complete#0 : none)
  }

  dataflow.graph.func private @multi_stream_export(%start: none)
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %c0 = dataflow.constant %start {const_value = 0 : i32} : i32
    %c1 = dataflow.constant %start {const_value = 1 : i32} : i32
    %c2 = dataflow.constant %start {const_value = 2 : i32} : i32
    %iv, %phase = dataflow.stream %c0, %c2, %c1
        step add while slt : i32
    %tokens = dataflow.invariant %phase, %start : none
    %complete:2 = dataflow.demux %phase, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%complete#0 : none)
  }

  dataflow.graph.func private @memory_export(
      %start: none, %memory: memref<?xi32>) -> (none, memref<?xi32>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams()
        memories(%memory : memref<?xi32>) complete(%start : none)
  }

  dataflow.graph.func private @invocation_reentry(
      %start: none, %index: index, %memory: memref<?xi32>) -> none
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %value, %loaded = dataflow.load %memory[%index] %start : memref<?xi32>
    %incremented = arith.addi %value, %one : i32
    %stored = dataflow.store %memory[%index] %incremented %loaded
        : memref<?xi32>
    dataflow.graph.return values() streams() memories()
        complete(%stored : none)
  }
}
