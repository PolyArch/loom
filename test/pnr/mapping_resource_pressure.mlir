// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph two_loads_one_port --hardware-mlir %s --hardware one_load_adg --workload two_loads_one_port --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=CSV < %t.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.json

// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph mixed_addi_pressure --hardware-mlir %s --hardware mixed_addi_adg --workload mixed_addi_pressure --output %t.mixed.csv --artifact %t.mixed.json
// RUN: FileCheck %s --check-prefix=MIXED-CSV < %t.mixed.csv
// RUN: FileCheck %s --check-prefix=MIXED-JSON < %t.mixed.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: two_loads_one_port,one_load_adg,two_loads_one_port__two_loads_one_port__one_load_adg,4,{{[0-9]+}},{{[0-9]+}},1,fail,missing hardware resource for software op dataflow.load
// CSV-SAME: resource pressure: resource_kind=fabric.mem.load operation=dataflow.load required=2 available=1 placed=1 missing=1

// JSON-DAG: "status": "fail"
// JSON-DAG: "unplaced_records": 1
// JSON-DAG: "diagnostics": [
// JSON-DAG: resource pressure: resource_kind=fabric.mem.load operation=dataflow.load required=2 available=1 placed=1 missing=1
// JSON-DAG: "resource_pressure": [
// JSON-DAG: "resource_kind": "fabric.mem.load"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "required": 2
// JSON-DAG: "available": 1
// JSON-DAG: "placed": 1
// JSON-DAG: "missing": 1

// MIXED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// MIXED-CSV-NEXT: mixed_addi_pressure,mixed_addi_adg,mixed_addi_pressure__mixed_addi_pressure__mixed_addi_adg,5,{{[0-9]+}},{{[0-9]+}},1,fail,missing hardware resource for software op arith.addi
// MIXED-CSV-SAME: resource pressure: resource_kind=fabric.op operation=arith.addi required=3 available=2 placed=2 missing=1

// MIXED-JSON-DAG: "status": "fail"
// MIXED-JSON-DAG: "unplaced_records": 1
// MIXED-JSON-DAG: "resource_kind": "fabric.op"
// MIXED-JSON-DAG: "operation": "arith.addi"
// MIXED-JSON-DAG: "required": 3
// MIXED-JSON-DAG: "available": 2
// MIXED-JSON-DAG: "placed": 2
// MIXED-JSON-DAG: "missing": 1

module {
  dataflow.graph private @two_loads_one_port(
      %ctrl: none, %idx: index, %a: memref<?xi32>, %b: memref<?xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 1, 0, 2>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs, %lhs_done = dataflow.load %a[%idx] %ctrl : memref<?xi32>
    %rhs, %rhs_done = dataflow.load %b[%idx] %ctrl : memref<?xi32>
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  fabric.module @one_load_adg(%mgr : memref<?x!fabric.bits<32>>,
                              %idx : !fabric.bits<32>,
                              %ctrl : !fabric.bits<0>,
                              %lhs : !fabric.bits<32>,
                              %rhs : !fabric.bits<32>) {
    %ctrl_to_load, %ctrl_to_sync = fabric.switch [spatial] %ctrl
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
    %load_data, %load_done =
        fabric.mem [spatial] mgr(%mgr) load(%idx, %ctrl_to_load) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    %sum = fabric.pe [spatial] (%pa = %lhs : !fabric.bits<32>,
                                %pb = %rhs : !fabric.bits<32>,
                                %pc = %load_done : !fabric.bits<0> to !fabric.bits<32>,
                                %pd = %ctrl_to_sync : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %pa : !fabric.bits<32>,
                %fb = %pb : !fabric.bits<32>,
                %loaded = %pc : !fabric.bits<32> to !fabric.bits<0>,
                %start = %pd : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %result = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %joined0, %joined1 = fabric.op [@dataflow.sync] (%loaded, %start)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<0>)
              -> (!fabric.bits<0>, !fabric.bits<0>)
        %done, %published = fabric.op [@dataflow.sync] (%joined0, %result)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  dataflow.graph private @mixed_addi_pressure(
      %ctrl: none,
      %i32a: i32, %i32b: i32,
      %i64a: i64, %i64b: i64,
      %i32c: i32, %i32d: i32)
      -> (i32, i64, i32) {
    %sum0 = arith.addi %i32a, %i32b : i32
    %sum1 = arith.addi %i64a, %i64b : i64
    %sum2 = arith.addi %i32c, %i32d : i32
    dataflow.graph.return %ctrl, %sum0, %sum1, %sum2
        : none, i32, i64, i32
  }

  fabric.module @mixed_addi_adg(%ctrl : !fabric.bits<0>,
                                %i32a : !fabric.bits<32>,
                                %i32b : !fabric.bits<32>,
                                %i64a : !fabric.bits<64>,
                                %i64b : !fabric.bits<64>) {
    %ctrl_i32, %ctrl_i64, %ctrl_extra = fabric.switch [spatial] %ctrl
        [{connectivity_table = ["1", "1", "1"]}]
        : (!fabric.bits<0>)
          -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
    %i32a_add, %i32a_extra = fabric.switch [spatial] %i32a
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.pe [spatial] (%pa = %i32a_add : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>,
                         %pc = %ctrl_i32 : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %sum)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i64a : !fabric.bits<64>,
                         %pb = %i64b : !fabric.bits<64>,
                         %pc = %ctrl_i64 : !fabric.bits<0> to !fabric.bits<64>)
        -> !fabric.bits<64> {
      fabric.fu(%lhs = %pa : !fabric.bits<64>,
                %rhs = %pb : !fabric.bits<64>,
                %token = %pc : !fabric.bits<64> to !fabric.bits<0>)
          -> !fabric.bits<64> {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
            : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
        %done, %published = fabric.op [@dataflow.sync] (%token, %sum)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<64>)
              -> (!fabric.bits<0>, !fabric.bits<64>)
        fabric.yield %sum : !fabric.bits<64>
      }
    }
    fabric.pe [spatial] (
        %pc = %ctrl_extra : !fabric.bits<0> to !fabric.bits<32>,
        %pv = %i32a_extra : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(
          %token = %pc : !fabric.bits<32> to !fabric.bits<0>,
          %value = %pv : !fabric.bits<32>) -> !fabric.bits<32> {
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %published : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
