// RUN: loom-dfg-sim %s --graph ranked_contiguous_load --arg 0=1 \
// RUN:   --memref 1=1,2,3,4,5,6,7,8 --output %t.rank-load.json
// RUN: FileCheck %s --check-prefix=RANK-LOAD < %t.rank-load.json
// RUN: loom-dfg-sim %s --graph ranked_contiguous_store --arg 0=1 \
// RUN:   --arg 1=6618611909121 --memref 2=0,0,0,0,0,0,0,0 \
// RUN:   --output %t.rank-store.json
// RUN: FileCheck %s --check-prefix=RANK-STORE < %t.rank-store.json
// RUN: loom-dfg-sim %s --graph ranked_gather \
// RUN:   --arg 0=0x00000003000000000000000200000004 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.rank-gather.json
// RUN: FileCheck %s --check-prefix=RANK-GATHER < %t.rank-gather.json
// RUN: loom-dfg-sim %s --graph ranked_scatter \
// RUN:   --arg 0=0x00000003000000000000000200000004 --arg 1=287454020 \
// RUN:   --memref 2=1,2,3,4,5 --output %t.rank-scatter.json
// RUN: FileCheck %s --check-prefix=RANK-SCATTER < %t.rank-scatter.json
// RUN: loom-dfg-sim %s --graph masked_scatter \
// RUN:   --arg 0=0x00000063000000030000006300000001 --arg 1=287454020 \
// RUN:   --arg 2=5 --memref 3=1,2,3,4,5 --output %t.masked-scatter.json
// RUN: FileCheck %s --check-prefix=MASKED-SCATTER < %t.masked-scatter.json
// RUN: loom-dfg-sim %s --graph masked_scatter \
// RUN:   --arg 0=0x00000063000000630000006300000063 --arg 1=287454020 \
// RUN:   --arg 2=0 --memref 3=1,2,3,4,5 --output %t.zero-scatter.json
// RUN: FileCheck %s --check-prefix=ZERO-SCATTER < %t.zero-scatter.json
// RUN: loom-dfg-sim %s --graph ranked_scatter \
// RUN:   --arg 0=0x00000001000000020000000000000002 --arg 1=287454020 \
// RUN:   --memref 2=1,2,3,4,5 --output %t.dup-scatter.json
// RUN: FileCheck %s --check-prefix=DUP-SCATTER < %t.dup-scatter.json
// RUN: loom-dfg-sim %s --graph ranked_scatter \
// RUN:   --arg 0=0x00000002000000630000000100000000 --arg 1=287454020 \
// RUN:   --memref 2=1,2,3,4,5 --output %t.range-scatter.json
// RUN: FileCheck %s --check-prefix=RANGE-SCATTER < %t.range-scatter.json
// RUN: loom-dfg-sim %s --graph wrapped_masked_load --arg 0=4294967294 \
// RUN:   --arg 1=12 --memref 2=10,20,30,40,50 --output %t.wrap-load.json
// RUN: FileCheck %s --check-prefix=WRAP-LOAD < %t.wrap-load.json
// RUN: loom-dfg-sim %s --graph zero_width_masked_load --arg 0=0 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.zero-width.json
// RUN: FileCheck %s --check-prefix=ZERO-WIDTH < %t.zero-width.json
// RUN: loom-dfg-sim %s --graph zero_width_masked_store --arg 0=287454020 \
// RUN:   --arg 1=0 --memref 2=1,2,3,4,5 --output %t.zero-width-store.json
// RUN: FileCheck %s --check-prefix=ZERO-WIDTH-STORE < %t.zero-width-store.json

// A rank-2 contiguous access reads `base + i` for flattened lane `i` in
// row-major order, so lane zero owns the low bit slice.
// RANK-LOAD-DAG: "status": "pass"
// RANK-LOAD-DAG: "event_count": 1
// RANK-LOAD-DAG: "dataflow.load": 1
// RANK-LOAD-DAG: "vector<2x3xi8>:0x70605040302"

// RANK-STORE: "event_count": 2
// RANK-STORE: "arg2": [
// RANK-STORE-NEXT: "i8:0",
// RANK-STORE-NEXT: "i8:1",
// RANK-STORE-NEXT: "i8:2",
// RANK-STORE-NEXT: "i8:3",
// RANK-STORE-NEXT: "i8:4",
// RANK-STORE-NEXT: "i8:5",
// RANK-STORE-NEXT: "i8:6",
// RANK-STORE-NEXT: "i8:0"
// RANK-STORE: "dataflow.store": 1
// RANK-STORE: "status": "pass"

// RANK-GATHER-DAG: "status": "pass"
// RANK-GATHER-DAG: "event_count": 1
// RANK-GATHER-DAG: "dataflow.load": 1
// RANK-GATHER-DAG: "vector<2x2xi8>:0x280A1E32"

// Distinct active scatter destinations execute; lane `i` writes the element
// named by address lane `i`.
// RANK-SCATTER: "event_count": 2
// RANK-SCATTER: "arg2": [
// RANK-SCATTER-NEXT: "i8:34",
// RANK-SCATTER-NEXT: "i8:2",
// RANK-SCATTER-NEXT: "i8:51",
// RANK-SCATTER-NEXT: "i8:17",
// RANK-SCATTER-NEXT: "i8:68"
// RANK-SCATTER: "dataflow.store": 1
// RANK-SCATTER: "status": "pass"

// Inactive lanes evaluate no address, so their out-of-range and duplicated
// addresses neither fault nor collide.
// MASKED-SCATTER: "event_count": 3
// MASKED-SCATTER: "arg3": [
// MASKED-SCATTER-NEXT: "i8:1",
// MASKED-SCATTER-NEXT: "i8:68",
// MASKED-SCATTER-NEXT: "i8:3",
// MASKED-SCATTER-NEXT: "i8:34",
// MASKED-SCATTER-NEXT: "i8:5"
// MASKED-SCATTER: "dataflow.store": 1
// MASKED-SCATTER: "status": "pass"

// An all-zero mask completes the firing without evaluating any address.
// ZERO-SCATTER: "event_count": 3
// ZERO-SCATTER: "arg3": [
// ZERO-SCATTER-NEXT: "i8:1",
// ZERO-SCATTER-NEXT: "i8:2",
// ZERO-SCATTER-NEXT: "i8:3",
// ZERO-SCATTER-NEXT: "i8:4",
// ZERO-SCATTER-NEXT: "i8:5"
// ZERO-SCATTER: "dataflow.store": 1
// ZERO-SCATTER: "status": "pass"

// A plain scatter has no lane order for duplicate active destinations. Only
// the runtime addresses expose that conflict, so the run reports an
// unsupported capability instead of choosing a lane order or witnessing a
// deadlock. The store never fires and Unsupported exports no terminal state.
// DUP-SCATTER: "dataflow.store does not resolve duplicate active addresses"
// DUP-SCATTER-DAG: "final_memory_roots": {}
// DUP-SCATTER-DAG: "final_memory_state": {}
// DUP-SCATTER-DAG: "final_outputs": []
// DUP-SCATTER-NOT: "dataflow.store"
// DUP-SCATTER: "status": "unsupported"

// One out-of-range active lane refuses the whole firing, so the lanes that
// resolved before it leave no partial write behind.
// RANGE-SCATTER: "dataflow.store address is out of range"
// RANGE-SCATTER: "arg2": [
// RANGE-SCATTER-NEXT: "i8:1",
// RANGE-SCATTER-NEXT: "i8:2",
// RANGE-SCATTER-NEXT: "i8:3",
// RANGE-SCATTER-NEXT: "i8:4",
// RANGE-SCATTER-NEXT: "i8:5"
// RANGE-SCATTER: "status": "blocked"

// A contiguous lane ordinal is added at the declared index width, so lanes 2
// and 3 of a 32-bit base 4294967294 address elements 0 and 1.
// WRAP-LOAD-DAG: "status": "pass"
// WRAP-LOAD-DAG: "dataflow.load": 1
// WRAP-LOAD-DAG: "vector<4xi8>:0x140A0000"

// An all-zero mask evaluates no address, but the access still has a structural
// index width, so an unusable declaration is reported rather than skipped.
// ZERO-WIDTH-DAG: "status": "blocked"
// ZERO-WIDTH-DAG: "index bit width must be nonzero"

// A store takes the same structural width, so it cannot retire an all-zero
// mask that a load refuses.
// ZERO-WIDTH-STORE: "index bit width must be nonzero"
// ZERO-WIDTH-STORE: "arg2": [
// ZERO-WIDTH-STORE-NEXT: "i8:1",
// ZERO-WIDTH-STORE-NEXT: "i8:2",
// ZERO-WIDTH-STORE-NEXT: "i8:3",
// ZERO-WIDTH-STORE-NEXT: "i8:4",
// ZERO-WIDTH-STORE-NEXT: "i8:5"
// ZERO-WIDTH-STORE-NOT: "dataflow.store"
// ZERO-WIDTH-STORE: "status": "blocked"

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @ranked_contiguous_load(
      %start: none, %idx: index, %mem: memref<?xi8>) -> vector<2x3xi8>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done =
        dataflow.load %mem[%idx] %start : memref<?xi8>, vector<2x3xi8>
    dataflow.graph.return %done, %data : none, vector<2x3xi8>
  }

  dataflow.graph private @ranked_contiguous_store(
      %start: none, %idx: index, %packed: i48, %mem: memref<?xi8>)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed : i48 -> vector<2x3xi8>
    %done =
        dataflow.store %mem[%idx] %data %start : memref<?xi8>, vector<2x3xi8>
    dataflow.graph.return %done : none
  }

  dataflow.graph private @ranked_gather(
      %start: none, %addresses: vector<2x2xindex>, %mem: memref<?xi8>)
      -> vector<2x2xi8>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done = dataflow.load %mem[%addresses] %start
        : memref<?xi8>, vector<2x2xindex>, vector<2x2xi8>
    dataflow.graph.return %done, %data : none, vector<2x2xi8>
  }

  dataflow.graph private @ranked_scatter(
      %start: none, %addresses: vector<2x2xindex>, %packed: i32,
      %mem: memref<?xi8>)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed : i32 -> vector<2x2xi8>
    %done = dataflow.store %mem[%addresses] %data %start
        : memref<?xi8>, vector<2x2xindex>, vector<2x2xi8>
    dataflow.graph.return %done : none
  }

  dataflow.graph private @masked_scatter(
      %start: none, %addresses: vector<2x2xindex>, %packed: i32,
      %packed_mask: i4, %mem: memref<?xi8>)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed : i32 -> vector<2x2xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<2x2xi1>
    %done = dataflow.store %mem[%addresses] %data %start mask %mask
        : memref<?xi8>, vector<2x2xindex>, vector<2x2xi8>
    dataflow.graph.return %done : none
  }

  dataflow.graph private @wrapped_masked_load(
      %start: none, %idx: index, %packed_mask: i4, %mem: memref<?xi8>)
      -> vector<4xi8>
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %data, %done = dataflow.load %mem[%idx] %start mask %mask
        : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done, %data : none, vector<4xi8>
  }
}

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 0>>
} {
  dataflow.graph private @zero_width_masked_load(
      %start: none, %packed_mask: i4, %mem: memref<?xi8>) -> vector<4xi8>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %data, %done = dataflow.load %mem[%idx] %start mask %mask
        : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done, %data : none, vector<4xi8>
  }

  dataflow.graph private @zero_width_masked_store(
      %start: none, %packed: i32, %packed_mask: i4, %mem: memref<?xi8>)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %data = dataflow.unpack %packed : i32 -> vector<4xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %done = dataflow.store %mem[%idx] %data %start mask %mask
        : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done : none
  }
}
