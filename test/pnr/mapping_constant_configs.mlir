// RUN: loom-pnr-map --dfg-mlir %s --graph constant_two --hardware-mlir %s --hardware constant_adg --workload constant_two --output %t.pass.csv --artifact %t.pass.json
// RUN: FileCheck %s --check-prefix=CSV-PASS < %t.pass.csv
// RUN: FileCheck %s --check-prefix=JSON-PASS < %t.pass.json
// RUN: loom-pnr-map --dfg-mlir %s --graph constant_three --hardware-mlir %s --hardware constant_adg --workload constant_three --output %t.fail.csv --artifact %t.fail.json
// RUN: FileCheck %s --check-prefix=CSV-FAIL < %t.fail.csv
// RUN: loom-pnr-map --dfg-mlir %s --graph shared_constant_five --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload shared_constant_five --output %t.shared-five.csv --artifact %t.shared-five.json
// RUN: FileCheck %s --check-prefix=CSV-SHARED-FIVE < %t.shared-five.csv
// RUN: FileCheck %s --check-prefix=JSON-SHARED-FIVE < %t.shared-five.json
// RUN: loom-pnr-map --dfg-mlir %s --graph structured_constant_bounds --hardware-mlir %s --hardware bounds_constant_adg --workload structured_constant_bounds --output %t.bounds.csv --artifact %t.bounds.json
// RUN: FileCheck %s --check-prefix=CSV-BOUNDS < %t.bounds.csv
// RUN: FileCheck %s --check-prefix=JSON-BOUNDS < %t.bounds.json
// RUN: loom-pnr-map --dfg-mlir %s --graph shared_constant_eight --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload shared_constant_eight --output %t.shared-eight.csv --artifact %t.shared-eight.json
// RUN: FileCheck %s --check-prefix=CSV-SHARED-EIGHT < %t.shared-eight.csv
// RUN: FileCheck %s --check-prefix=JSON-SHARED-EIGHT < %t.shared-eight.json
// RUN: loom-pnr-map --dfg-mlir %s --graph wide_constant_thirty_one --hardware-mlir %s --hardware wide_constant_adg --workload wide_constant_thirty_one --output %t.wide.csv --artifact %t.wide.json
// RUN: FileCheck %s --check-prefix=CSV-WIDE < %t.wide.csv
// RUN: FileCheck %s --check-prefix=JSON-WIDE < %t.wide.json

// CSV-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-PASS-NEXT: constant_two,constant_adg,constant_two__constant_two__constant_adg,1,0,0,0,pass

// JSON-PASS-DAG: "register": "sw_configs.const_hex_value"
// JSON-PASS-DAG: "value": "0x00000002"
// JSON-PASS-DAG: "source": "placement:dataflow.constant#0"

// CSV-FAIL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-FAIL-NEXT: constant_three,constant_adg,constant_three__constant_three__constant_adg,0,0,0,1,fail,missing hardware resource for software op dataflow.constant

// CSV-SHARED-FIVE: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-SHARED-FIVE-NEXT: shared_constant_five,shared_reduction_adg,shared_constant_five__shared_constant_five__shared_reduction_adg,5,0,0,0,pass

// JSON-SHARED-FIVE-DAG: "software": "dataflow.constant#4"
// JSON-SHARED-FIVE-DAG: "status": "pass"

// CSV-BOUNDS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-BOUNDS-NEXT: structured_constant_bounds,bounds_constant_adg,structured_constant_bounds__structured_constant_bounds__bounds_constant_adg,2,1,0,0,pass

// JSON-BOUNDS-DAG: "status": "pass"
// JSON-BOUNDS-DAG: "software": "dataflow.constant#0"
// JSON-BOUNDS-DAG: "value": "0x00000008"
// JSON-BOUNDS-DAG: "edge_ref": "dataflow.constant#0.result0->arith.cmpi#0.operand1"
// JSON-BOUNDS-NOT: "software": "dataflow.constant#1"
// JSON-BOUNDS-NOT: "software": "dataflow.constant#2"
// JSON-BOUNDS-NOT: "software": "dataflow.constant#3"
// JSON-BOUNDS-NOT: "missing hardware resource for software op dataflow.constant"

// CSV-SHARED-EIGHT: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-SHARED-EIGHT-NEXT: shared_constant_eight,shared_memory_reduction_adg,shared_constant_eight__shared_constant_eight__shared_memory_reduction_adg,1,0,0,0,pass

// JSON-SHARED-EIGHT-DAG: "software": "dataflow.constant#0"
// JSON-SHARED-EIGHT-DAG: "value": "0x00000008"
// JSON-SHARED-EIGHT-DAG: "status": "pass"

// CSV-WIDE: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-WIDE-NEXT: wide_constant_thirty_one,wide_constant_adg,wide_constant_thirty_one__wide_constant_thirty_one__wide_constant_adg,1,0,0,0,pass

// JSON-WIDE-DAG: "software": "dataflow.constant#0"
// JSON-WIDE-DAG: "value": "0x000000000000001f"
// JSON-WIDE-DAG: "status": "pass"

module {
  dataflow.graph.func private @constant_two(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @constant_three(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @shared_constant_five(%ctrl: none)
      -> (none, i32, i32, i32, i32, i32) {
    %zero0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one0 = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero1 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one1 = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    dataflow.graph.return %ctrl, %zero0, %one0, %two, %zero1, %one1
        : none, i32, i32, i32, i32, i32
  }

  dataflow.graph.func private @structured_constant_bounds(
      %ctrl: none, %data: i32) -> none {
    %limit = dataflow.constant %ctrl {const_value = 8 : i32} : i32
    %lb = dataflow.constant %ctrl {const_value = 0 : i64} : i64
    %step = dataflow.constant %ctrl {const_value = 1 : i64} : i64
    %ub = dataflow.constant %ctrl {const_value = 16 : i64} : i64
    scf.for %i = %lb to %ub step %step : i64 {
      %under_limit = arith.cmpi ult, %data, %limit : i32
      scf.if %under_limit {
      }
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @shared_constant_eight(%ctrl: none)
      -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 8 : i32} : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @wide_constant_thirty_one(%ctrl: none)
      -> (none, i64) {
    %value = dataflow.constant %ctrl {const_value = 31 : i64} : i64
    dataflow.graph.return %ctrl, %value : none, i64
  }

  fabric.module @constant_adg(%ctrl: !fabric.bits<0>) {
    fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%input = %pa : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@dataflow.constant] (%input)
                 {hw_params = [{const_hex_value = ["0x00000002"]}]}
                 : (!fabric.bits<0>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  fabric.module @bounds_constant_adg(%ctrl: !fabric.bits<0>,
                                     %data: !fabric.bits<32>) {
    %limit = fabric.pe [spatial]
        (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%input = %pa : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@dataflow.constant] (%input)
                 {hw_params = [{const_hex_value = ["0x00000008"]}]}
                 : (!fabric.bits<0>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%lhs = %data : !fabric.bits<32>,
                         %rhs = %limit : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %rhs : !fabric.bits<32>) -> () {
        %pred = fabric.op [@arith.cmpi] (%a, %b)
                {hw_params = [{predicate = ["ult"]}]}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
        fabric.yield
      }
    }
    fabric.yield
  }

  fabric.module @wide_constant_adg(%ctrl: !fabric.bits<0>) {
    fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<64>)
        -> !fabric.bits<64> {
      fabric.fu(%input = %pa : !fabric.bits<64> to !fabric.bits<0>)
          -> !fabric.bits<64> {
        %value = fabric.op [@dataflow.constant] (%input)
                 {hw_params = [{const_hex_value = ["0x000000000000001f"]}]}
                 : (!fabric.bits<0>) -> !fabric.bits<64>
        fabric.yield %value : !fabric.bits<64>
      }
    }
    fabric.yield
  }
}
