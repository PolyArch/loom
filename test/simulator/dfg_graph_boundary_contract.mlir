// RUN: loom-dfg-sim %s --graph empty_stream_export --output %t.empty.json
// RUN: FileCheck %s --check-prefix=EMPTY-STREAM < %t.empty.json
// RUN: loom-dfg-sim %s --graph multi_stream_export --output %t.multi.json
// RUN: FileCheck %s --check-prefix=MULTI-STREAM < %t.multi.json
// RUN: loom-dfg-sim %s --graph memory_export --memref 0=2,5 --output %t.memory.json
// RUN: FileCheck %s --check-prefix=MEMORY < %t.memory.json
// RUN: loom-dfg-sim %s --graph view_memory_export --memref 0=2,5 --output %t.view.json
// RUN: FileCheck %s --check-prefix=VIEW < %t.view.json
// RUN: loom-dfg-sim %s --graph fresh_memory_export --arg 0=7 --output %t.fresh.json
// RUN: FileCheck %s --check-prefix=FRESH < %t.fresh.json
// RUN: loom-dfg-sim %s --graph fresh_memory_export --invocations 2 --arg 0=7 --arg 0=9 --output %t.fresh-reentry.json
// RUN: FileCheck %s --check-prefix=FRESH-REENTRY < %t.fresh-reentry.json
// RUN: loom-dfg-sim %s --graph fresh_memory_reentry --invocations 2 --arg 0=true --arg 0=false --arg 1=7 --arg 1=9 --output %t.fresh-isolated.json
// RUN: FileCheck %s --check-prefix=FRESH-ISOLATED < %t.fresh-isolated.json
// RUN: loom-dfg-sim %s --graph fresh_dynamic_memory_export --arg 0=1 --arg 1=13 --output %t.dynamic.json
// RUN: FileCheck %s --check-prefix=DYNAMIC < %t.dynamic.json
// RUN: loom-dfg-sim %s --graph invocation_reentry --invocations 3 --arg 0=0 --arg 0=1 --arg 0=2 --memref 1=0,0,0 --output %t.reentry.json
// RUN: FileCheck %s --check-prefix=REENTRY < %t.reentry.json

// EMPTY-STREAM-DAG: "status": "pass"
// EMPTY-STREAM-DAG: "final_stream_outputs": [
// EMPTY-STREAM-DAG: []

// MULTI-STREAM-DAG: "status": "pass"
// MULTI-STREAM-DAG: "final_stream_outputs": [
// MULTI-STREAM-DAG: "i32:0"
// MULTI-STREAM-DAG: "i32:1"

// MEMORY: "arg0": "memory_root0"
// MEMORY: "memory_result0": "memory_root0"
// MEMORY: "arg0": [
// MEMORY-NEXT: "i32:2"
// MEMORY-NEXT: "i32:5"
// MEMORY: "memory_result0": [
// MEMORY-NEXT: "i32:2"
// MEMORY-NEXT: "i32:5"
// MEMORY: "status": "pass"

// VIEW: "arg0": "memory_root0"
// VIEW: "memory_result0": "memory_root0"
// VIEW: "arg0": [
// VIEW-NEXT: "i32:2"
// VIEW-NEXT: "i32:5"
// VIEW: "memory_result0": [
// VIEW-NEXT: "i32:2"
// VIEW-NEXT: "i32:5"
// VIEW: "status": "pass"

// FRESH: "memory_result0": [
// FRESH-NEXT: "i32:7"
// FRESH: "status": "pass"

// The report contains only the final invocation's local alias class; the
// repeated label is not a cross-invocation object identity.
// FRESH-REENTRY: "memory_result0": "memory_root0"
// FRESH-REENTRY: "memory_result0": [
// FRESH-REENTRY-NEXT: "i32:9"
// FRESH-REENTRY: "status": "pass"

// FRESH-ISOLATED: "memory_result0": [
// FRESH-ISOLATED-NEXT: "uninitialized"
// FRESH-ISOLATED: "status": "pass"

// DYNAMIC: "memory_result0": [
// DYNAMIC-NEXT: "i32:13"
// DYNAMIC: "status": "pass"

// REENTRY-DAG: "status": "pass"
// REENTRY-DAG: "dataflow.load": 3
// REENTRY-DAG: "dataflow.store": 3
// REENTRY-DAG: "arg1": [
// REENTRY-DAG: "i32:1"
// REENTRY-DAG: "i32:1"
// REENTRY-DAG: "i32:1"

module {
  dataflow.graph private @empty_stream_export(%start: none)
      -> (i32)
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

  dataflow.graph private @multi_stream_export(%start: none)
      -> (i32)
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

  dataflow.graph private @memory_export(
      %start: none, %memory: memref<?xi32>) -> (memref<?xi32>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams()
        memories(%memory : memref<?xi32>) complete(%start : none)
  }

  dataflow.graph private @view_memory_export(
      %start: none, %memory: memref<2xi32>) -> (memref<?xi32>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    %view = memref.cast %memory : memref<2xi32> to memref<?xi32>
    dataflow.graph.return values() streams()
        memories(%view : memref<?xi32>) complete(%start : none)
  }

  dataflow.graph private @fresh_memory_export(
      %start: none, %value: i32) -> (memref<1xi32>)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc() : memref<1xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %done = dataflow.store %slot[%index] %value %start : memref<1xi32>
    dataflow.graph.return values() streams()
        memories(%slot : memref<1xi32>) complete(%done : none)
  }

  dataflow.graph private @fresh_dynamic_memory_export(
      %start: none, %extent: index, %value: i32) -> (memref<?xi32>)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc(%extent) : memref<?xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %done = dataflow.store %slot[%index] %value %start : memref<?xi32>
    dataflow.graph.return values() streams()
        memories(%slot : memref<?xi32>) complete(%done : none)
  }

  dataflow.graph private @fresh_memory_reentry(
      %start: none, %write: i1, %value: i32) -> (memref<1xi32>)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc() : memref<1xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %skip, %store_ctrl = dataflow.demux %write, %start
        : (i1, none) -> (none, none)
    %stored = dataflow.store %slot[%index] %value %store_ctrl : memref<1xi32>
    %complete = dataflow.mux %write, %skip, %stored
        : (i1, none, none) -> none
    dataflow.graph.return values() streams()
        memories(%slot : memref<1xi32>) complete(%complete : none)
  }

  dataflow.graph private @invocation_reentry(
      %start: none, %index: index, %memory: memref<?xi32>) -> ()
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
