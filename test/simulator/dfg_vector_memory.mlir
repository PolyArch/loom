// RUN: loom-dfg-sim %s --graph masked_load --arg 0=2 --arg 1=5 \
// RUN:   --memref 2=1,2,3,4,5 --output %t.load.json
// RUN: FileCheck %s --check-prefix=LOAD < %t.load.json
// RUN: loom-dfg-sim %s --graph unmasked_load --arg 0=1 \
// RUN:   --memref 1=1,2,3,4,5 --output %t.unmasked-load.json
// RUN: FileCheck %s --check-prefix=UNMASKED-LOAD < %t.unmasked-load.json
// RUN: loom-dfg-sim %s --graph masked_load --arg 0=99 --arg 1=0 \
// RUN:   --memref 2=1,2,3,4,5 --output %t.zero-load.json
// RUN: FileCheck %s --check-prefix=ZERO-LOAD < %t.zero-load.json
// RUN: loom-dfg-sim %s --graph masked_store --arg 0=2 --arg 1=740365835 \
// RUN:   --arg 2=5 --memref 3=1,2,3,4,5 --output %t.store.json
// RUN: FileCheck %s --check-prefix=STORE < %t.store.json
// RUN: loom-dfg-sim %s --graph masked_store --arg 0=99 --arg 1=740365835 \
// RUN:   --arg 2=0 --memref 3=1,2,3,4,5 --output %t.zero-store.json
// RUN: FileCheck %s --check-prefix=ZERO-STORE < %t.zero-store.json
// RUN: loom-dfg-sim %s --graph gather_load_32 \
// RUN:   --arg 0=0x00000000000000040000000100000004 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.gather32.json
// RUN: FileCheck %s --check-prefix=GATHER32 < %t.gather32.json
// RUN: loom-dfg-sim %s --graph gather_forward_32 --arg 0=0x4 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.forward32.json
// RUN: FileCheck %s --check-prefix=FORWARD32 < %t.forward32.json
// RUN: loom-dfg-sim %s --graph gather_load_64 \
// RUN:   --arg 0=0x0000000000000000000000000000000400000000000000010000000000000004 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.gather64.json
// RUN: FileCheck %s --check-prefix=GATHER64 < %t.gather64.json
// RUN: loom-dfg-sim %s --graph gather_forward_64 --arg 0=0x4 \
// RUN:   --memref 1=10,20,30,40,50 --output %t.forward64.json
// RUN: FileCheck %s --check-prefix=FORWARD64 < %t.forward64.json
// RUN: loom-dfg-sim %s --graph masked_gather_load_32 \
// RUN:   --arg 0=0x00000063000000010000006300000004 --arg 1=5 \
// RUN:   --memref 2=10,20,30,40,50 --output %t.masked-gather.json
// RUN: FileCheck %s --check-prefix=MASKED-GATHER < %t.masked-gather.json
// RUN: loom-dfg-sim %s --graph masked_gather_load_32 \
// RUN:   --arg 0=0x00000063000000630000006300000063 --arg 1=0 \
// RUN:   --memref 2=10,20,30,40,50 --output %t.zero-gather.json
// RUN: FileCheck %s --check-prefix=ZERO-GATHER < %t.zero-gather.json

// LOAD-DAG: "status": "pass"
// LOAD-DAG: "event_count": 2
// LOAD-DAG: "dataflow.load": 1
// LOAD-DAG: "dataflow.unpack": 1
// LOAD-DAG: "vector<4xi8>:0x50003"

// UNMASKED-LOAD-DAG: "status": "pass"
// UNMASKED-LOAD-DAG: "event_count": 1
// UNMASKED-LOAD-DAG: "dataflow.load": 1
// UNMASKED-LOAD-DAG: "vector<4xi8>:0x5040302"

// ZERO-LOAD-DAG: "status": "pass"
// ZERO-LOAD-DAG: "event_count": 2
// ZERO-LOAD-DAG: "dataflow.load": 1
// ZERO-LOAD-DAG: "dataflow.unpack": 1
// ZERO-LOAD-DAG: "vector<4xi8>:0x0"

// STORE: "event_count": 3
// STORE: "arg3": [
// STORE-NEXT: "i8:1",
// STORE-NEXT: "i8:2",
// STORE-NEXT: "i8:11",
// STORE-NEXT: "i8:4",
// STORE-NEXT: "i8:33"
// STORE: "dataflow.store": 1
// STORE-NEXT: "dataflow.unpack": 2
// STORE: "status": "pass"

// ZERO-STORE: "event_count": 3
// ZERO-STORE: "arg3": [
// ZERO-STORE-NEXT: "i8:1",
// ZERO-STORE-NEXT: "i8:2",
// ZERO-STORE-NEXT: "i8:3",
// ZERO-STORE-NEXT: "i8:4",
// ZERO-STORE-NEXT: "i8:5"
// ZERO-STORE: "dataflow.store": 1
// ZERO-STORE-NEXT: "dataflow.unpack": 2
// ZERO-STORE: "status": "pass"

// GATHER32-DAG: "status": "pass"
// GATHER32-DAG: "event_count": 1
// GATHER32-DAG: "dataflow.load": 1
// GATHER32-DAG: "vector<4xi8>:0xA321432"

// FORWARD32-DAG: "status": "pass"
// FORWARD32-DAG: "dataflow.load": 1
// FORWARD32-DAG: "vector<4xindex>:0x4"

// GATHER64-DAG: "status": "pass"
// GATHER64-DAG: "event_count": 1
// GATHER64-DAG: "dataflow.load": 1
// GATHER64-DAG: "vector<4xi8>:0xA321432"

// FORWARD64-DAG: "status": "pass"
// FORWARD64-DAG: "dataflow.load": 1
// FORWARD64-DAG: "vector<4xindex>:0x4"

// MASKED-GATHER-DAG: "status": "pass"
// MASKED-GATHER-DAG: "event_count": 2
// MASKED-GATHER-DAG: "dataflow.load": 1
// MASKED-GATHER-DAG: "dataflow.unpack": 1
// MASKED-GATHER-DAG: "vector<4xi8>:0x140032"

// ZERO-GATHER-DAG: "status": "pass"
// ZERO-GATHER-DAG: "event_count": 2
// ZERO-GATHER-DAG: "dataflow.load": 1
// ZERO-GATHER-DAG: "dataflow.unpack": 1
// ZERO-GATHER-DAG: "vector<4xi8>:0x0"

module {
  dataflow.graph private @masked_load(
      %start: none, %idx: index, %packed_mask: i4, %mem: memref<?xi8>)
      -> vector<4xi8>
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %data, %done = dataflow.load %mem[%idx] %start mask %mask
        : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done, %data : none, vector<4xi8>
  }

  dataflow.graph private @unmasked_load(
      %start: none, %idx: index, %mem: memref<?xi8>) -> vector<4xi8>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done =
        dataflow.load %mem[%idx] %start : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done, %data : none, vector<4xi8>
  }

  dataflow.graph private @masked_store(
      %start: none, %idx: index, %packed_data: i32, %packed_mask: i4,
      %mem: memref<?xi8>)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed_data : i32 -> vector<4xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %done = dataflow.store %mem[%idx] %data %start mask %mask
        : memref<?xi8>, vector<4xi8>
    dataflow.graph.return %done : none
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph private @gather_load_32(
        %start: none, %addresses: vector<4xindex>, %mem: memref<?xi8>)
        -> vector<4xi8>
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data, %done = dataflow.load %mem[%addresses] %start
          : memref<?xi8>, vector<4xindex>, vector<4xi8>
      dataflow.graph.return %done, %data : none, vector<4xi8>
    }

    dataflow.graph private @gather_forward_32(
        %start: none, %addresses: vector<4xindex>, %mem: memref<?xi8>)
        -> vector<4xindex>
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data, %done = dataflow.load %mem[%addresses] %start
          : memref<?xi8>, vector<4xindex>, vector<4xi8>
      dataflow.graph.return %done, %addresses : none, vector<4xindex>
    }

    dataflow.graph private @masked_gather_load_32(
        %start: none, %addresses: vector<4xindex>, %packed_mask: i4,
        %mem: memref<?xi8>) -> vector<4xi8>
        attributes {input_segments = array<i32: 2, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
      %data, %done = dataflow.load %mem[%addresses] %start mask %mask
          : memref<?xi8>, vector<4xindex>, vector<4xi8>
      dataflow.graph.return %done, %data : none, vector<4xi8>
    }
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
  } {
    dataflow.graph private @gather_load_64(
        %start: none, %addresses: vector<4xindex>, %mem: memref<?xi8>)
        -> vector<4xi8>
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data, %done = dataflow.load %mem[%addresses] %start
          : memref<?xi8>, vector<4xindex>, vector<4xi8>
      dataflow.graph.return %done, %data : none, vector<4xi8>
    }

    dataflow.graph private @gather_forward_64(
        %start: none, %addresses: vector<4xindex>, %mem: memref<?xi8>)
        -> vector<4xindex>
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data, %done = dataflow.load %mem[%addresses] %start
          : memref<?xi8>, vector<4xindex>, vector<4xi8>
      dataflow.graph.return %done, %addresses : none, vector<4xindex>
    }
  }

}
