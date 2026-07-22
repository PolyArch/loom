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
