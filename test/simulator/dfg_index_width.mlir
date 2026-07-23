// RUN: env LOOM_INDEX_WIDTH=32 loom-dfg-sim %s --graph index_width_fallback --arg 0=4294967296 --memref 1=10,20,30 --output %t.fallback32.json
// RUN: FileCheck %s --check-prefix=FALLBACK32 < %t.fallback32.json
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_width_fallback --arg 0=4294967296 --memref 1=10,20,30 --output %t.fallback64.json
// RUN: FileCheck %s --check-prefix=FALLBACK64 < %t.fallback64.json
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_width_explicit32 --arg 0=4294967296 --memref 1=10,20,30 --output %t.explicit32.json
// RUN: FileCheck %s --check-prefix=EXPLICIT32 < %t.explicit32.json
// RUN: env LOOM_INDEX_WIDTH=32 loom-dfg-sim %s --graph index_vector_forward \
// RUN:   --arg 0=0x000000060000000500000004000000030000000200000001 \
// RUN:   --output %t.index-vector.json
// RUN: FileCheck %s --check-prefix=INDEX-VECTOR < %t.index-vector.json
// RUN: loom-dfg-sim %s --graph invalid_index_width_with_stream --arg 0=1 --arg 1=0 --arg 2=64 --arg 3=1 --output %t.invalid.json
// RUN: FileCheck %s --check-prefix=INVALID < %t.invalid.json
// RUN: grep -c 'index bit width must be in \[1, 64\], got 128' %t.invalid.json | FileCheck %s --check-prefix=INVALID-COUNT
// RUN: env LOOM_INDEX_WIDTH=128 loom-dfg-sim %s --graph wide_index_load \
// RUN:   --arg 0=2 --memref 1=10,20,30 --output %t.wide-load.json
// RUN: FileCheck %s --check-prefix=WIDE-LOAD < %t.wide-load.json
// RUN: env LOOM_INDEX_WIDTH=128 loom-dfg-sim %s --graph wide_index_load \
// RUN:   --arg 0=18446744073709551616 --memref 1=10,20,30 \
// RUN:   --output %t.wide-range.json
// RUN: FileCheck %s --check-prefix=WIDE-RANGE < %t.wide-range.json
// RUN: env LOOM_INDEX_WIDTH=128 not loom-dfg-sim %s --graph wide_index_load \
// RUN:   --arg 0=340282366920938463463374607431768211456 \
// RUN:   --memref 1=10,20,30 --output %t.wide-overflow.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIDE-OVERFLOW
// RUN: env LOOM_INDEX_WIDTH=128 loom-dfg-sim %s --graph wide_index_gather \
// RUN:   --arg 0=0x0000000000000000000000000000000200000000000000000000000000000001 \
// RUN:   --memref 1=10,20,30 --output %t.wide-gather.json
// RUN: FileCheck %s --check-prefix=WIDE-GATHER < %t.wide-gather.json
// RUN: env LOOM_INDEX_WIDTH=128 loom-dfg-sim %s --graph wide_index_gather \
// RUN:   --arg 0=0x0000000000000000000000000000000100000000000000010000000000000000 \
// RUN:   --memref 1=10,20,30 --output %t.wide-gather-range.json
// RUN: FileCheck %s --check-prefix=WIDE-GATHER-RANGE < %t.wide-gather-range.json
// A memory fixture is an operand of its own graph, so an explicit declaration
// overrides the configured fallback for its element tokens too.
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_memory_load \
// RUN:   --arg 0=0 --memref 1=7 --output %t.index-memory-load.json
// RUN: FileCheck %s --check-prefix=INDEX-MEMORY-LOAD \
// RUN:   < %t.index-memory-load.json
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %s --graph index_memory_store \
// RUN:   --arg 0=0 --arg 1=9 --memref 2=7,8 \
// RUN:   --output %t.index-memory-store.json
// RUN: FileCheck %s --check-prefix=INDEX-MEMORY-STORE \
// RUN:   < %t.index-memory-store.json

// FALLBACK32-DAG: "index:0"
// FALLBACK32-DAG: "i32:20"

// FALLBACK64-DAG: "index:4294967296"
// FALLBACK64-DAG: "i32:30"

// EXPLICIT32-DAG: "index:0"
// EXPLICIT32-DAG: "i32:20"

// A rank-2 index vector token is 6 lanes of the resolved 32-bit index width,
// and the runtime boundary preserves its exact packed bit pattern.
// INDEX-VECTOR-DAG: "status": "pass"
// INDEX-VECTOR-DAG: "vector<2x3xindex>:0x60000000500000004000000030000000200000001"

// INVALID-DAG: "status": "blocked"
// INVALID-DAG: "dataflow.stream": 65
// INVALID-COUNT: 1

// The memory path carries an index at its resolved width, so an arbitrary
// configured 128-bit index addresses memory exactly rather than through a
// host integer, without needing an explicit declaration.
// WIDE-LOAD-DAG: "status": "pass"
// WIDE-LOAD-DAG: "dataflow.load": 1
// WIDE-LOAD-DAG: "i32:30"

// 2^64 is a representable 128-bit index and no element of this memory, so it
// is refused instead of truncating into an in-range host address.
// WIDE-RANGE-DAG: "status": "blocked"
// WIDE-RANGE-DAG: "dataflow.load address is out of range"

// 2^128 has no 128-bit index representation at all, so the runtime boundary
// rejects it exactly instead of wrapping it into one.
// WIDE-OVERFLOW: index argument does not fit its declared bit width

// WIDE-GATHER-DAG: "status": "pass"
// WIDE-GATHER-DAG: "dataflow.load": 1
// WIDE-GATHER-DAG: "vector<2xi32>:0x1E00000014"

// WIDE-GATHER-RANGE-DAG: "status": "blocked"
// WIDE-GATHER-RANGE-DAG: "dataflow.load address is out of range"

// The fixture element is read back at the declared 32-bit width, not the
// 64-bit configured fallback.
// INDEX-MEMORY-LOAD-DAG: "status": "pass"
// INDEX-MEMORY-LOAD-DAG: "dataflow.load": 1
// INDEX-MEMORY-LOAD-DAG: "index:7"

// The written element and the untouched fixture element are both encoded at
// that same declared width.
// INDEX-MEMORY-STORE: "arg2": [
// INDEX-MEMORY-STORE-NEXT: "index:9",
// INDEX-MEMORY-STORE-NEXT: "index:8"
// INDEX-MEMORY-STORE: "dataflow.store": 1
// INDEX-MEMORY-STORE: "status": "pass"

module {
  dataflow.graph private @index_width_fallback(
      %ctrl: none, %value: i64, %base: memref<?xi32>)
      -> (index, i32)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %index = arith.index_cast %value : i64 to index
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    %loaded, %loaded_done = dataflow.load %base[%one] %ctrl : memref<?xi32>
    %published:3 = dataflow.sync %loaded_done, %index, %loaded
        : (none, index, i32) -> (none, index, i32)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, index, i32
  }

  dataflow.graph private @wide_index_load(
      %ctrl: none, %addr: index, %mem: memref<?xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xi32>
    dataflow.graph.return %done, %data : none, i32
  }

  dataflow.graph private @wide_index_gather(
      %ctrl: none, %addresses: vector<2xindex>, %mem: memref<?xi32>)
      -> vector<2xi32>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done = dataflow.load %mem[%addresses] %ctrl
        : memref<?xi32>, vector<2xindex>, vector<2xi32>
    dataflow.graph.return %done, %data : none, vector<2xi32>
  }

  dataflow.graph private @index_vector_forward(
      %ctrl: none, %addresses: vector<2x3xindex>) -> vector<2x3xindex>
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %ctrl, %addresses
        : (none, vector<2x3xindex>) -> (none, vector<2x3xindex>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2x3xindex>
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph private @index_width_explicit32(
        %ctrl: none, %value: i64, %base: memref<?xi32>)
        -> (index, i32)
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 2, 0, 0>} {
      %index = arith.index_cast %value : i64 to index
      %one = dataflow.constant %ctrl {const_value = 1 : index} : index
      %loaded, %loaded_done = dataflow.load %base[%one] %ctrl : memref<?xi32>
      %published:3 = dataflow.sync %loaded_done, %index, %loaded
          : (none, index, i32) -> (none, index, i32)
      dataflow.graph.return %published#0, %published#1, %published#2
          : none, index, i32
    }

    dataflow.graph private @index_memory_load(
        %ctrl: none, %addr: index, %mem: memref<?xindex>) -> index
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xindex>
      dataflow.graph.return %done, %data : none, index
    }

    dataflow.graph private @index_memory_store(
        %ctrl: none, %addr: index, %value: index, %mem: memref<?xindex>)
        attributes {input_segments = array<i32: 2, 0, 1>,
                    result_segments = array<i32: 0, 0, 0>} {
      %done = dataflow.store %mem[%addr] %value %ctrl : memref<?xindex>
      dataflow.graph.return %done : none
    }
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 128>>
  } {
    dataflow.graph private @invalid_index_width_with_stream(
        %ctrl: none, %value: i64, %init: i64, %limit: i64, %step: i64)
        -> (index, i1)
        attributes {input_segments = array<i32: 4, 0, 0>,
                    result_segments = array<i32: 1, 1, 0>} {
      %index = arith.index_cast %value : i64 to index
      %iv, %phase = dataflow.stream %init, %limit, %step
          step add while slt : i64
      %tokens = dataflow.invariant %phase, %ctrl : none
      %closed:2 = dataflow.demux %phase, %tokens
          : (i1, none) -> (none, none)
      %published:2 = dataflow.sync %closed#0, %index
          : (none, index) -> (none, index)
      dataflow.graph.return values(%published#1 : index)
          streams(%phase : i1) memories() complete(%published#0 : none)
    }
  }
}
